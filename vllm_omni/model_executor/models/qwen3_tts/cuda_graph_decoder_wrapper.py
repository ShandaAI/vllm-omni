# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

This module provides CUDA Graph acceleration for the speech tokenizer decoder,
reducing kernel launch overhead during inference.
"""

import os

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _bench_timing_enabled() -> bool:
    return bool(os.environ.get("BENCH_CODE2WAV_TIMING"))


def _parse_positive_int_list(value: str) -> list[int]:
    sizes: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        size = int(item)
        if size <= 0:
            raise ValueError(f"Expected positive integer, got {size}")
        sizes.append(size)
    return sorted(set(sizes))


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024**2):.2f} MiB"


def _cuda_memory_snapshot(device: torch.device) -> tuple[int, int, int, int]:
    torch.cuda.synchronize(device)
    return (
        torch.cuda.memory_allocated(device),
        torch.cuda.memory_reserved(device),
        torch.cuda.max_memory_allocated(device),
        torch.cuda.max_memory_reserved(device),
    )


def _log_cuda_memory_delta(
    label: str,
    start: tuple[int, int, int, int],
    end: tuple[int, int, int, int],
) -> None:
    start_allocated, start_reserved, start_max_allocated, start_max_reserved = start
    end_allocated, end_reserved, end_max_allocated, end_max_reserved = end
    logger.info(
        "%s CUDA memory: allocated=%s (%+s), reserved=%s (%+s), "
        "max_allocated=%s (%+s), max_reserved=%s (%+s)",
        label,
        _format_bytes(end_allocated),
        _format_bytes(end_allocated - start_allocated),
        _format_bytes(end_reserved),
        _format_bytes(end_reserved - start_reserved),
        _format_bytes(end_max_allocated),
        _format_bytes(end_max_allocated - start_max_allocated),
        _format_bytes(end_max_reserved),
        _format_bytes(end_max_reserved - start_max_reserved),
    )


class CUDAGraphDecoderWrapper:
    """
    CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

    This wrapper captures the decoder forward pass for fixed input sizes
    and replays them during inference to reduce kernel launch overhead.

    Usage:
        wrapper = CUDAGraphDecoderWrapper(decoder, capture_sizes=[25, 50, 100, 200, 300])
        wrapper.warmup(device)

        # During inference:
        output = wrapper.decode(codes)  # Automatically uses CUDA graph if possible
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_sizes: list[int] | None = None,
        capture_batch_sizes: list[int] | None = None,
        num_quantizers: int = 8,
        enabled: bool = True,
    ):
        self.decoder = decoder
        self._explicit_sizes = capture_sizes is not None
        self.capture_sizes = sorted(capture_sizes) if capture_sizes else []
        self._explicit_batch_sizes = capture_batch_sizes is not None
        self.capture_batch_sizes = (
            sorted({int(size) for size in capture_batch_sizes if int(size) > 0})
            if capture_batch_sizes
            else []
        )
        self.num_quantizers = num_quantizers
        self.enabled = enabled

        self.graphs: dict[tuple[int, int], CUDAGraph] = {}
        self.static_inputs: dict[tuple[int, int], torch.Tensor] = {}
        self.static_outputs: dict[tuple[int, int], torch.Tensor] = {}
        self.max_batch_size = 1

        self._warmed_up = False
        self._device = None

    @staticmethod
    def compute_capture_sizes(
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        codec_streaming: bool = False,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ) -> list[int]:
        """Compute capture sizes from chunking config for high graph hit rate."""
        sizes: set[int] = set()

        if codec_streaming and codec_chunk_frames > 0:
            # Streaming sends one codec window per scheduler step. Capture the
            # initial chunk, the full left-context window, and compact tail/IC
            # buckets below the steady-state window.
            sizes.add(codec_chunk_frames)
            streaming_max = codec_chunk_frames
            if codec_left_context_frames > 0:
                streaming_max = codec_chunk_frames + codec_left_context_frames
                sizes.add(streaming_max)
            for p2 in [2, 4, 8, 16, 32, 64, 128, 256]:
                if p2 < streaming_max:
                    sizes.add(p2)
            return sorted(sizes)

        # Preserve the compatible non-streaming sizing behavior.
        if codec_chunk_frames > 0:
            sizes.add(codec_chunk_frames)
            if codec_left_context_frames > 0:
                sizes.add(codec_chunk_frames + codec_left_context_frames)

        if decode_chunk_size > 0:
            sizes.add(decode_chunk_size)

        non_stream_max = decode_chunk_size + decode_left_context
        sizes.add(non_stream_max)

        for p2 in [2, 4, 8, 16, 32, 64, 128, 256]:
            if p2 <= non_stream_max:
                sizes.add(p2)

        return sorted(sizes)

    @staticmethod
    def compute_capture_batch_sizes(max_batch_size: int) -> list[int]:
        """Compute a compact set of graph batch buckets up to max_batch_size."""
        max_batch_size = max(1, int(max_batch_size))
        preferred = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
        sizes = {size for size in preferred if size <= max_batch_size}
        sizes.add(max_batch_size)
        return sorted(sizes)

    def _get_padded_size(self, actual_size: int) -> int | None:
        for size in self.capture_sizes:
            if actual_size <= size:
                return size
        return None

    def _get_padded_batch_size(self, actual_batch_size: int) -> int | None:
        for size in self.capture_batch_sizes:
            if actual_batch_size <= size:
                return size
        return None

    def warmup(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.long,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        codec_streaming: bool = False,
        max_batch_size: int = 1,
    ):
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return

        self._device = device
        self.decoder.eval()
        self.max_batch_size = max(1, int(max_batch_size))

        if not self._explicit_sizes:
            self.capture_sizes = self.compute_capture_sizes(
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left_context_frames,
                codec_streaming=codec_streaming,
            )
        if not self._explicit_batch_sizes:
            batch_sizes_env = (
                (
                    os.environ.get("CODE2WAV_STREAMING_CUDAGRAPH_BATCH_SIZES")
                    or os.environ.get("CODE2WAV_CUDAGRAPH_BATCH_SIZES")
                )
                if codec_streaming
                else None
            )
            if batch_sizes_env:
                self.capture_batch_sizes = [
                    size
                    for size in _parse_positive_int_list(batch_sizes_env)
                    if size <= self.max_batch_size
                ]
                if self.max_batch_size not in self.capture_batch_sizes:
                    self.capture_batch_sizes.append(self.max_batch_size)
                self.capture_batch_sizes = sorted(set(self.capture_batch_sizes))
            elif codec_streaming:
                self.capture_batch_sizes = self.compute_capture_batch_sizes(self.max_batch_size)
            else:
                self.capture_batch_sizes = list(range(1, self.max_batch_size + 1))
        if not self.capture_batch_sizes:
            self.capture_batch_sizes = [1]
        self.max_batch_size = max(self.capture_batch_sizes)

        total_graphs = len(self.capture_batch_sizes) * len(self.capture_sizes)
        logger.info(
            "Starting CUDA Graph warmup for %d batch buckets x %d seq sizes (%d graphs): batch_sizes=%s seq_lens=%s",
            len(self.capture_batch_sizes),
            len(self.capture_sizes),
            total_graphs,
            self.capture_batch_sizes,
            self.capture_sizes,
        )
        torch.cuda.reset_peak_memory_stats(device)
        warmup_start_mem = _cuda_memory_snapshot(device)
        logger.info(
            "Code2Wav CUDA Graph memory baseline: allocated=%s, reserved=%s",
            _format_bytes(warmup_start_mem[0]),
            _format_bytes(warmup_start_mem[1]),
        )

        # Warmup runs to ensure CUDA memory is allocated
        for batch_size in self.capture_batch_sizes:
            for size in self.capture_sizes:
                dummy = torch.zeros(batch_size, self.num_quantizers, size, dtype=dtype, device=device)
                with torch.no_grad():
                    _ = self.decoder(dummy)

        torch.cuda.synchronize(device)
        eager_warmup_mem = _cuda_memory_snapshot(device)
        _log_cuda_memory_delta("Code2Wav eager warmup", warmup_start_mem, eager_warmup_mem)

        for batch_size in self.capture_batch_sizes:
            for size in self.capture_sizes:
                try:
                    before_capture_mem = _cuda_memory_snapshot(device)
                    self._capture(batch_size, size, device, dtype)
                    after_capture_mem = _cuda_memory_snapshot(device)
                    logger.info("  Captured CUDA Graph for batch=%d size=%d", batch_size, size)
                    _log_cuda_memory_delta(
                        f"  Capture batch={batch_size} size={size}",
                        before_capture_mem,
                        after_capture_mem,
                    )
                except Exception:
                    logger.warning(
                        "  Failed to capture graph for batch=%d size=%d",
                        batch_size,
                        size,
                        exc_info=True,
                    )

        self._warmed_up = True
        final_mem = _cuda_memory_snapshot(device)
        _log_cuda_memory_delta("Code2Wav CUDA Graph total", warmup_start_mem, final_mem)
        logger.info("CUDA Graph warmup complete: %d/%d captured", len(self.graphs), total_graphs)

    def _capture(self, batch_size: int, size: int, device: torch.device, dtype: torch.dtype):
        static_input = torch.zeros(batch_size, self.num_quantizers, size, dtype=dtype, device=device)
        with torch.no_grad():
            _ = self.decoder(static_input)
        torch.cuda.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self.decoder(static_input)

        graph_key = (batch_size, size)
        self.graphs[graph_key] = graph
        self.static_inputs[graph_key] = static_input
        self.static_outputs[graph_key] = static_output

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self._warmed_up:
            return self.decoder(codes)

        # Inner CUDA graph replay is illegal while an outer stream capture is
        # active (e.g. vLLM's cudagraph_mode=FULL warmup on Stage 1). Fall back
        # to eager in that case so the outer capture can complete.
        if torch.cuda.is_current_stream_capturing():
            return self.decoder(codes)

        batch_size = int(codes.shape[0])
        actual_size = codes.shape[-1]
        padded_batch_size = self._get_padded_batch_size(batch_size)
        padded_size = self._get_padded_size(actual_size)
        graph_key = (
            (padded_batch_size, padded_size)
            if padded_batch_size is not None and padded_size is not None
            else None
        )

        if graph_key is None or graph_key not in self.graphs:
            if _bench_timing_enabled():
                logger.info(
                    "[CUDAGraph] op=decode batch=%d padded_batch=%s padded=%s actual=%d hit=false",
                    batch_size,
                    padded_batch_size,
                    padded_size,
                    actual_size,
                )
            return self.decoder(codes)

        if _bench_timing_enabled():
            logger.info(
                "[CUDAGraph] op=decode batch=%d padded_batch=%d padded=%d actual=%d hit=true",
                batch_size,
                padded_batch_size,
                padded_size,
                actual_size,
            )
        self.static_inputs[graph_key].zero_()
        self.static_inputs[graph_key][:batch_size, :, :actual_size] = codes
        self.graphs[graph_key].replay()

        actual_out_len = actual_size * self.decoder.total_upsample
        return self.static_outputs[graph_key][:batch_size, :, :actual_out_len].clone()

    def batched_decode(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        if not codes_list:
            return []

        batch_size = len(codes_list)
        actual_sizes = [int(codes.shape[-1]) for codes in codes_list]
        max_actual_size = max(actual_sizes)
        padded_batch_size = self._get_padded_batch_size(batch_size)
        padded_size = self._get_padded_size(max_actual_size)
        graph_key = (
            (padded_batch_size, padded_size)
            if padded_batch_size is not None and padded_size is not None
            else None
        )

        if self.enabled and self._warmed_up and graph_key in self.graphs:
            if _bench_timing_enabled():
                logger.info(
                    "[CUDAGraph] op=batched_decode batch=%d padded_batch=%d padded=%d actual=%s hit=true",
                    batch_size,
                    padded_batch_size,
                    padded_size,
                    ",".join(str(size) for size in actual_sizes),
                )
            static_input = self.static_inputs[graph_key]
            static_input.zero_()
            for i, codes in enumerate(codes_list):
                static_input[i : i + 1, :, : actual_sizes[i]] = codes
            self.graphs[graph_key].replay()
            output = self.static_outputs[graph_key]
        else:
            if _bench_timing_enabled():
                logger.info(
                    "[CUDAGraph] op=batched_decode batch=%d padded_batch=%s padded=%s actual=%s hit=false",
                    batch_size,
                    padded_batch_size,
                    padded_size,
                    ",".join(str(size) for size in actual_sizes),
                )
            padded_size = padded_size or max_actual_size
            padded_input = torch.zeros(
                batch_size,
                codes_list[0].shape[1],
                padded_size,
                dtype=codes_list[0].dtype,
                device=codes_list[0].device,
            )
            for i, codes in enumerate(codes_list):
                padded_input[i : i + 1, :, : actual_sizes[i]] = codes
            output = self.decoder(padded_input)

        total_upsample = self.decoder.total_upsample
        return [
            output[i : i + 1, :, : actual_size * total_upsample].clone()
            for i, actual_size in enumerate(actual_sizes)
        ]

    def chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        wavs = []
        start_index = 0
        total_len = codes.shape[-1]
        total_upsample = self.decoder.total_upsample

        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index

            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self.decode(codes_chunk)

            wavs.append(wav_chunk[..., context_size * total_upsample :])
            start_index = end_index

        return torch.cat(wavs, dim=-1)
