from __future__ import annotations

import os
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
)
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class _Code2WavBatchPolicy:
    min_padding_efficiency: float
    min_padding_efficiency_b2: float
    b2_high_load_padding_efficiency: float
    high_load_ewma_threshold: float
    replay_overhead_frames: float
    min_cost_saving: float
    b2_min_cost_saving: float
    min_absolute_efficiency: float


def _env_float(name: str, default: str, *, mode: str | None = None) -> float:
    if mode:
        value = os.environ.get(f"CODE2WAV_{mode}_{name}")
        if value is not None:
            return float(value)
    return float(os.environ.get(f"CODE2WAV_{name}", default))


def _make_batch_policy(mode: str | None = None) -> _Code2WavBatchPolicy:
    return _Code2WavBatchPolicy(
        min_padding_efficiency=_env_float("BATCH_MIN_PADDING_EFFICIENCY", "0.80", mode=mode),
        min_padding_efficiency_b2=_env_float("BATCH2_MIN_PADDING_EFFICIENCY", "0.90", mode=mode),
        b2_high_load_padding_efficiency=_env_float(
            "BATCH2_HIGH_LOAD_PADDING_EFFICIENCY",
            "0.80",
            mode=mode,
        ),
        high_load_ewma_threshold=_env_float("BATCH_HIGH_LOAD_EWMA_THRESHOLD", "2.75", mode=mode),
        replay_overhead_frames=_env_float("BATCH_REPLAY_OVERHEAD_FRAMES", "16", mode=mode),
        min_cost_saving=_env_float("BATCH_MIN_COST_SAVING", "0.02", mode=mode),
        b2_min_cost_saving=_env_float("BATCH2_MIN_COST_SAVING", "0.06", mode=mode),
        min_absolute_efficiency=_env_float("BATCH_MIN_ABSOLUTE_EFFICIENCY", "0.50", mode=mode),
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else False
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1)[0].item() if value.numel() > 0 else False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class Qwen3TTSCode2Wav(nn.Module):
    """Stage-1 code2wav model for Qwen3-TTS (GenerationModelRunner).
    Consumes frame-aligned codec tokens from input_ids and decodes waveform
    via the SpeechTokenizer decoder directly (bypassing HF wrapper overhead)."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._decode_chunk_frames = 300
        self._decode_left_context_frames = 25
        self._codec_streaming = False
        self._logged_codec_stats = False
        self._streaming_batch_policy = _make_batch_policy("STREAMING")
        self._observed_batch_size_ewma = 1.0

        # Construct decoder from config so it is visible to vLLM's
        # memory profiler at startup.  Weights are loaded later in
        # load_weights().
        tok_config = Qwen3TTSTokenizerV2Config.from_pretrained(
            self.model_path,
            subfolder="speech_tokenizer",
        )
        dec_config = tok_config.decoder_config
        self.decoder = Qwen3TTSTokenizerV2Decoder._from_config(dec_config)
        self.decoder.eval()
        self._num_quantizers = int(dec_config.num_quantizers)
        self._output_sample_rate = int(tok_config.output_sample_rate)
        self._total_upsample = int(self.decoder.total_upsample)
        self._decoder_sliding_window = int(getattr(dec_config, "sliding_window", 0) or 0)

    def _batch_policy(self) -> _Code2WavBatchPolicy:
        return self._streaming_batch_policy

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def _connector_extra_config(self) -> dict[str, Any]:
        model_cfg = getattr(self.vllm_config, "model_config", None)
        connector_cfg = getattr(model_cfg, "stage_connector_config", None)
        extra_cfg = (
            connector_cfg.get("extra", connector_cfg)
            if isinstance(connector_cfg, dict)
            else getattr(connector_cfg, "extra", None)
        )
        return extra_cfg if isinstance(extra_cfg, dict) else {}

    def _apply_chunk_config(self) -> tuple[int, int]:
        model_cfg = getattr(self.vllm_config, "model_config", None)
        extra_cfg = self._connector_extra_config()
        chunk_frames = int(extra_cfg.get("codec_chunk_frames") or 0)
        left_frames = int(extra_cfg.get("codec_left_context_frames") or 0)
        self._codec_streaming = bool(extra_cfg.get("codec_streaming", False))
        if getattr(model_cfg, "async_chunk", False) and chunk_frames > 0:
            self._decode_chunk_frames = chunk_frames
            self._decode_left_context_frames = left_frames if left_frames > 0 else 0
        return chunk_frames, left_frames

    def _enable_decoder_cudagraph(self, decoder: nn.Module, device: torch.device, chunk_frames: int, left_frames: int):
        if not hasattr(decoder, "enable_cudagraph") or device.type != "cuda":
            return
        try:
            if (
                chunk_frames > 0
                and left_frames > 0
                and self._decoder_sliding_window
                and left_frames < self._decoder_sliding_window
            ):
                logger.warning(
                    "Qwen3-TTS streaming codec_left_context_frames=%d "
                    "is smaller than decoder sliding_window=%d; "
                    "chunk-boundary distortion may occur. "
                    "Increase codec_left_context_frames to at least %d for streaming.",
                    left_frames,
                    self._decoder_sliding_window,
                    self._decoder_sliding_window,
                )

            scheduler_cfg = getattr(self.vllm_config, "scheduler_config", None)
            max_batch_size = int(getattr(scheduler_cfg, "max_num_seqs", 1) or 1)
            decoder.enable_cudagraph(
                device=device,
                codec_chunk_frames=chunk_frames,
                codec_left_context_frames=left_frames,
                codec_streaming=self._codec_streaming,
                max_batch_size=max_batch_size,
            )
            logger.info("Code2Wav decoder CUDA Graph enabled")
        except Exception:
            logger.warning("Failed to enable CUDA Graph for Code2Wav decoder", exc_info=True)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # This stage ignores token embeddings. Keep a stable dummy embedding for vLLM runner.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def _split_request_ids(self, ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        """Split concatenated input_ids into per-request segments.

        Uses seq_token_counts (injected by the runner via model_kwargs) when
        available, falling back to forward-context ubatch_slices when
        micro-batching is active. Returns [ids] for single-request batches.
        """
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    @staticmethod
    def _extract_full_audio_codes(info: dict[str, Any] | None, q: int) -> torch.Tensor:
        if not info or "full_audio_codes" not in info:
            return torch.empty((0, q), dtype=torch.long)

        codes = info["full_audio_codes"]
        if isinstance(codes, torch.Tensor):
            codes_tensor = codes.to(dtype=torch.long).detach().cpu()
        else:
            codes_tensor = torch.tensor(codes, dtype=torch.long)

        if codes_tensor.numel() == 0:
            return torch.empty((0, q), dtype=torch.long)
        if codes_tensor.ndim == 1:
            if codes_tensor.numel() % q != 0:
                logger.warning(
                    "Ignoring malformed full_audio_codes with %d elements not divisible by num_quantizers=%d",
                    codes_tensor.numel(),
                    q,
                )
                return torch.empty((0, q), dtype=torch.long)
            return codes_tensor.reshape(-1, q).contiguous()
        if codes_tensor.ndim == 2:
            return codes_tensor.contiguous()

        logger.warning("Ignoring malformed full_audio_codes shape %s", tuple(codes_tensor.shape))
        return torch.empty((0, q), dtype=torch.long)

    @staticmethod
    def _return_codec_tokens_requested(info: dict[str, Any] | None) -> bool:
        if not info:
            return False
        return _as_bool(info.get("return_codec_tokens", False))

    @staticmethod
    def _decoder_max_batch_size(decoder: nn.Module) -> int:
        wrapper = getattr(decoder, "_cudagraph_wrapper", None)
        if wrapper is None:
            return 1024
        return max(1, int(getattr(wrapper, "max_batch_size", 1) or 1))

    @staticmethod
    def _decoder_padded_size(decoder: nn.Module, actual_size: int) -> int:
        wrapper = getattr(decoder, "_cudagraph_wrapper", None)
        get_padded_size = getattr(wrapper, "_get_padded_size", None)
        if callable(get_padded_size):
            padded = get_padded_size(actual_size)
            if padded is not None:
                return int(padded)
        return int(actual_size)

    @staticmethod
    def _decoder_padded_batch_size(decoder: nn.Module, actual_batch_size: int) -> int:
        wrapper = getattr(decoder, "_cudagraph_wrapper", None)
        get_padded_batch_size = getattr(wrapper, "_get_padded_batch_size", None)
        if callable(get_padded_batch_size):
            padded = get_padded_batch_size(actual_batch_size)
            if padded is not None:
                return int(padded)
        return int(actual_batch_size)

    def _code2wav_work(
        self,
        decoder: nn.Module,
        codes_list: list[torch.Tensor],
        *,
        force_serial: bool,
    ) -> tuple[float, int, int]:
        policy = self._batch_policy()
        active_sizes = [int(codes.shape[-1]) for codes in codes_list]
        actual_work = sum(active_sizes)
        padded_work = 0
        cost = 0.0
        if force_serial:
            for actual_size in active_sizes:
                padded_size = self._decoder_padded_size(decoder, actual_size)
                padded_work += padded_size
                cost += padded_size + policy.replay_overhead_frames
        else:
            padded_size = self._decoder_padded_size(decoder, max(active_sizes))
            padded_batch_size = self._decoder_padded_batch_size(decoder, len(active_sizes))
            padded_work = padded_batch_size * padded_size
            cost = padded_work + policy.replay_overhead_frames
        return cost, actual_work, padded_work

    def _code2wav_group_efficiency(self, decoder: nn.Module, codes_list: list[torch.Tensor]) -> float:
        _, actual_work, padded_work = self._code2wav_work(decoder, codes_list, force_serial=False)
        if padded_work <= 0:
            return 1.0
        return actual_work / float(padded_work)

    def _should_batch_codes(self, decoder: nn.Module, codes_list: list[torch.Tensor]) -> bool:
        batch_size = len(codes_list)
        if batch_size <= 1:
            return False
        policy = self._batch_policy()
        efficiency = self._code2wav_group_efficiency(decoder, codes_list)
        threshold = (
            policy.b2_high_load_padding_efficiency
            if batch_size == 2 and self._observed_batch_size_ewma >= policy.high_load_ewma_threshold
            else policy.min_padding_efficiency_b2
            if batch_size == 2
            else policy.min_padding_efficiency
        )
        if efficiency >= threshold:
            return True
        if efficiency < policy.min_absolute_efficiency:
            return False
        batch_cost = self._code2wav_work(decoder, codes_list, force_serial=False)[0]
        serial_cost = self._code2wav_work(decoder, codes_list, force_serial=True)[0]
        min_saving = policy.b2_min_cost_saving if batch_size == 2 else policy.min_cost_saving
        return batch_cost <= serial_cost * (1.0 - min_saving)

    def _code2wav_bucket_histogram(
        self,
        decoder: nn.Module,
        codes_list: list[torch.Tensor],
        *,
        force_serial: bool,
    ) -> Counter[str]:
        hist: Counter[str] = Counter()
        if force_serial:
            for codes in codes_list:
                hist[f"S1x{self._decoder_padded_size(decoder, int(codes.shape[-1]))}"] += 1
            return hist
        padded_size = self._decoder_padded_size(decoder, max(int(codes.shape[-1]) for codes in codes_list))
        padded_batch_size = self._decoder_padded_batch_size(decoder, len(codes_list))
        hist[f"B{len(codes_list)}>{padded_batch_size}x{padded_size}"] += 1
        return hist

    def _plan_code2wav_bucket_groups(
        self,
        decoder: nn.Module,
        bucket: list[tuple[int, torch.Tensor]],
    ) -> list[list[tuple[int, torch.Tensor]]]:
        if len(bucket) <= 1:
            return [[item] for item in bucket]

        max_batch_size = self._decoder_max_batch_size(decoder)
        candidates = range(2, min(max_batch_size, len(bucket)) + 1)
        ordered = sorted(bucket, key=lambda item: int(item[1].shape[-1]), reverse=True)
        groups: list[list[tuple[int, torch.Tensor]]] = []
        offset = 0
        while offset < len(ordered):
            best_size = 1
            best_saving = 0.0
            for group_size in candidates:
                if offset + group_size > len(ordered):
                    break
                group_codes = [ordered[j][1].unsqueeze(0) for j in range(offset, offset + group_size)]
                if not self._should_batch_codes(decoder, group_codes):
                    continue
                cost = self._code2wav_work(decoder, group_codes, force_serial=False)[0]
                serial_cost = self._code2wav_work(decoder, group_codes, force_serial=True)[0]
                saving = serial_cost - cost
                if saving >= best_saving:
                    best_size = group_size
                    best_saving = saving
            groups.append(ordered[offset : offset + best_size])
            offset += best_size
        return groups

    def _decode_single_code2wav(self, decoder: nn.Module, codes_qf: torch.Tensor) -> torch.Tensor:
        codes_bqf = codes_qf.unsqueeze(0)
        if self._codec_streaming:
            wrapper = getattr(decoder, "_cudagraph_wrapper", None)
            decode = getattr(wrapper, "decode", None)
            wav = decode(codes_bqf) if callable(decode) else decoder(codes_bqf)
            return wav.squeeze(0).squeeze(0)
        try:
            wav = decoder.chunked_decode(
                codes_bqf,
                chunk_size=self._decode_chunk_frames,
                left_context_size=self._decode_left_context_frames,
            )
        except TypeError:
            wav = decoder.chunked_decode(codes_bqf)
        return wav.squeeze(0).squeeze(0)

    def _decode_code2wav_batch_once(
        self,
        decoder: nn.Module,
        codes_bqf_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        wrapper = getattr(decoder, "_cudagraph_wrapper", None)
        batched_decode = getattr(wrapper, "batched_decode", None)
        if callable(batched_decode):
            return batched_decode(codes_bqf_list)
        eager_batched_decode = getattr(decoder, "_batched_decode_eager", None)
        if callable(eager_batched_decode):
            return eager_batched_decode(codes_bqf_list)
        return [decoder(codes_bqf) for codes_bqf in codes_bqf_list]

    def _decode_code2wav_grouped(
        self,
        decoder: nn.Module,
        valid_codes_qf: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        if not self._codec_streaming:
            raise RuntimeError("Grouped Code2Wav decode is only enabled for streaming codec")

        if len(valid_codes_qf) == 1 or not hasattr(decoder, "batched_chunked_decode"):
            wav_tensors = [self._decode_single_code2wav(decoder, codes_qf) for codes_qf in valid_codes_qf]
            bucket_hist = Counter()
            for codes_qf in valid_codes_qf:
                bucket_hist.update(
                    self._code2wav_bucket_histogram(decoder, [codes_qf.unsqueeze(0)], force_serial=True)
                )
            return wav_tensors, {
                "path": "serial",
                "groups": ["S1"],
                "avg_group_batch": 1.0,
                "padding_efficiency": 1.0,
                "bucket_hist": ",".join(f"{k}:{v}" for k, v in sorted(bucket_hist.items())),
                "observed_batch_ewma": self._observed_batch_size_ewma,
            }

        self._observed_batch_size_ewma = 0.95 * self._observed_batch_size_ewma + 0.05 * len(valid_codes_qf)
        bucketed: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for index, codes_qf in enumerate(valid_codes_qf):
            padded_size = self._decoder_padded_size(decoder, int(codes_qf.shape[-1]))
            bucketed.setdefault(padded_size, []).append((index, codes_qf))

        wav_tensors: list[torch.Tensor | None] = [None] * len(valid_codes_qf)
        group_labels: list[str] = []
        group_batch_sizes: list[int] = []
        weighted_efficiency_num = 0.0
        weighted_efficiency_den = 0
        bucket_hist: Counter[str] = Counter()
        batched_groups = 0

        for padded_size in sorted(bucketed):
            for group in self._plan_code2wav_bucket_groups(decoder, bucketed[padded_size]):
                group_indices = [item[0] for item in group]
                group_codes_qf = [item[1] for item in group]
                group_codes_bqf = [codes_qf.unsqueeze(0) for codes_qf in group_codes_qf]
                group_efficiency = self._code2wav_group_efficiency(decoder, group_codes_bqf)
                weighted_efficiency_num += group_efficiency * len(group_codes_bqf)
                weighted_efficiency_den += len(group_codes_bqf)
                if self._should_batch_codes(decoder, group_codes_bqf):
                    bucket_hist.update(self._code2wav_bucket_histogram(decoder, group_codes_bqf, force_serial=False))
                    decoded = self._decode_code2wav_batch_once(decoder, group_codes_bqf)
                    for idx, wav in zip(group_indices, decoded):
                        wav_tensors[idx] = wav.squeeze(0).squeeze(0)
                    batched_groups += 1
                    padded_batch_size = self._decoder_padded_batch_size(decoder, len(group_codes_bqf))
                    group_labels.append(f"B{len(group_codes_bqf)}>{padded_batch_size}@{padded_size}:{group_efficiency:.2f}")
                else:
                    bucket_hist.update(self._code2wav_bucket_histogram(decoder, group_codes_bqf, force_serial=True))
                    for idx, codes_qf in zip(group_indices, group_codes_qf):
                        wav_tensors[idx] = self._decode_single_code2wav(decoder, codes_qf)
                    group_labels.append(f"S{len(group_codes_bqf)}@{padded_size}:{group_efficiency:.2f}")
                group_batch_sizes.append(len(group_codes_qf))

        if any(wav is None for wav in wav_tensors):
            raise RuntimeError("Code2Wav grouped decode produced incomplete outputs")

        return [wav for wav in wav_tensors if wav is not None], {
            "path": "batched" if batched_groups else "serial",
            "groups": group_labels,
            "avg_group_batch": sum(group_batch_sizes) / len(group_batch_sizes) if group_batch_sizes else 1.0,
            "padding_efficiency": weighted_efficiency_num / weighted_efficiency_den if weighted_efficiency_den else 1.0,
            "bucket_hist": ",".join(f"{k}:{v}" for k, v in sorted(bucket_hist.items())),
            "observed_batch_ewma": self._observed_batch_size_ewma,
        }

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode codec codes into audio waveform.

        input_ids layout per request: [codec_context_frames, *flat_codes]
        where flat_codes is codebook-major [q*F].

        Bypasses the HF Qwen3TTSTokenizer.decode() wrapper and calls the
        decoder.chunked_decode() directly to avoid GPU->CPU->GPU round-trips.
        Length management is done here instead of relying on HF's padding=-1
        sentinel logic.
        """
        decoder = self.decoder
        q = int(self._num_quantizers)
        upsample = int(self._total_upsample)
        sr_val = int(self._output_sample_rate)
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty],
                    "sr": [sr_tensor],
                    "audio_codes": [torch.empty((0, q), dtype=torch.long)],
                },
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))

        parsed: list[tuple[int, int]] = []
        valid_codes_qf: list[torch.Tensor] = []
        valid_indices: list[int] = []
        left_context_size = [0] * len(request_ids_list)
        audio_codes = [torch.empty((0, q), dtype=torch.long) for _ in request_ids_list]
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= len(left_context_size):
                    break
                if self._return_codec_tokens_requested(info):
                    audio_codes[i] = self._extract_full_audio_codes(info, q)
                meta = info.get("meta", {})
                if "left_context_size" in meta:
                    # left_context_size may come through serialization as an int, [int], or tensor([int]).
                    value = meta["left_context_size"]
                    if isinstance(value, list):
                        value = value[0] if value else 0
                    if isinstance(value, torch.Tensor):
                        value = value.reshape(-1)[0].item() if value.numel() > 0 else 0
                    left_context_size[i] = int(value)
        for i, req_ids in enumerate(request_ids_list):
            if req_ids.numel() < 1:
                parsed.append((0, 0))
                continue
            ctx_frames = left_context_size[i]
            flat = req_ids
            n = flat.numel()
            if n == 0 or n % q != 0:
                if n == 1 and int(flat.reshape(-1)[0].item()) == 0:
                    parsed.append((0, 0))
                    continue
                if n > 0:
                    logger.warning(
                        "Code2Wav input_ids length %d not divisible by num_quantizers %d; skipping malformed request.",
                        n,
                        q,
                    )
                parsed.append((0, 0))
                continue
            frames = n // q
            # [q*F] -> [Q, F] for direct decoder call (decoder expects [B, Q, F])
            codes_qf = flat.reshape(q, frames)
            parsed.append((ctx_frames, frames))
            valid_codes_qf.append(codes_qf)
            valid_indices.append(i)

        num_req = len(request_ids_list)
        if not valid_codes_qf:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": [sr_tensor] * num_req,
                    "audio_codes": audio_codes,
                },
            )

        if not self._logged_codec_stats:
            self._logged_codec_stats = True
            try:
                c = valid_codes_qf[0]
                logger.info(
                    "Code2Wav codec: frames=%d q=%d uniq=%d range=[%d,%d] batch=%d",
                    c.shape[1],
                    q,
                    int(torch.unique(c).numel()),
                    int(c.min().item()),
                    int(c.max().item()),
                    len(valid_codes_qf),
                )
            except Exception:
                pass

        bench_timing = bool(os.environ.get("BENCH_CODE2WAV_TIMING"))
        timing_start = time.perf_counter() if bench_timing else 0.0
        decode_path = "streaming"
        if self._codec_streaming:
            wav_tensors, batch_metrics = self._decode_code2wav_grouped(decoder, valid_codes_qf)
            if bench_timing:
                device = self._module_device(decoder)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                decode_ms = (time.perf_counter() - timing_start) * 1000
                frames = ",".join(str(int(codes_qf.shape[-1])) for codes_qf in valid_codes_qf)
                total_frames = sum(int(codes_qf.shape[-1]) for codes_qf in valid_codes_qf)
                logger.info(
                    "[Code2Wav decode] mode=streaming batch=%d path=%s frames=[%s] decode_ms=%.3f "
                    "ms_per_frame=%.6f avg_group_batch=%.2f padding_eff=%.3f "
                    "batch_ewma=%.2f buckets=%s groups=%s",
                    len(valid_codes_qf),
                    batch_metrics["path"],
                    frames,
                    decode_ms,
                    decode_ms / max(total_frames, 1),
                    batch_metrics["avg_group_batch"],
                    batch_metrics["padding_efficiency"],
                    batch_metrics["observed_batch_ewma"],
                    batch_metrics["bucket_hist"],
                    ";".join(batch_metrics["groups"]),
                )
        elif len(valid_codes_qf) > 1 and hasattr(decoder, "batched_chunked_decode"):
            decode_path = "nonstream_batched"
            wav_tensors = [
                wav.squeeze(0).squeeze(0)
                for wav in decoder.batched_chunked_decode(
                    [codes_qf.unsqueeze(0) for codes_qf in valid_codes_qf]
                )
            ]
        else:
            decode_path = "nonstream_serial"
            wav_tensors = []
            for codes_qf in valid_codes_qf:
                codes_bqf = codes_qf.unsqueeze(0)  # [1, Q, F]
                try:
                    wav = decoder.chunked_decode(
                        codes_bqf,
                        chunk_size=self._decode_chunk_frames,
                        left_context_size=self._decode_left_context_frames,
                    )  # [1, 1, wav_len]
                except TypeError:
                    # Unit-test fakes and older decoder shims may not accept the
                    # explicit chunk kwargs; production Qwen3-TTS decoders do.
                    wav = decoder.chunked_decode(codes_bqf)  # [1, 1, wav_len]
                wav_tensors.append(wav.squeeze(0).squeeze(0))  # [wav_len]
        if bench_timing and not self._codec_streaming:
            device = self._module_device(decoder)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decode_ms = (time.perf_counter() - timing_start) * 1000
            frames = ",".join(str(int(codes_qf.shape[-1])) for codes_qf in valid_codes_qf)
            total_frames = sum(int(codes_qf.shape[-1]) for codes_qf in valid_codes_qf)
            logger.info(
                "[Code2Wav decode] mode=nonstream batch=%d path=%s frames=[%s] decode_ms=%.3f ms_per_frame=%.6f",
                len(valid_codes_qf),
                decode_path,
                frames,
                decode_ms,
                decode_ms / max(total_frames, 1),
            )

        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        for j, idx in enumerate(valid_indices):
            ctx_frames, actual_frames = parsed[idx]
            wav = wav_tensors[j]
            # Slice on exact codec-frame boundaries instead of proportionally.
            start = max(0, ctx_frames * upsample)
            end = max(start, actual_frames * upsample)
            if start >= wav.shape[0]:
                logger.warning(
                    "Context trim start %d >= decoded length %d; returning empty audio.",
                    start,
                    wav.shape[0],
                )
                continue
            wav = wav[start : min(end, wav.shape[0])]
            if wav.shape[0] > 0:
                audios[idx] = wav.to(dtype=torch.float32).reshape(-1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs, "audio_codes": audio_codes},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput | tuple, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)

        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(
                "Qwen3TTSCode2Wav expected OmniOutput, OmniOutput tuple, "
                f"or (audio_tensor, sr) outputs, got {type(model_outputs)}"
            )

        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audio_tensor,
                "sr": sr,
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The primary weights iterator contains no Code2Wav parameters.
        # Drain it so callers don't hang on an unconsumed generator.
        for _ in weights:
            pass

        # Load decoder weights from the speech_tokenizer/ subfolder
        # via vLLM's weight loader (handles sharded safetensors, index
        # files, and all load formats).  AutoWeightsLoader matches
        # "decoder.*" weights to self.decoder and skips encoder weights.
        model_loader = DefaultModelLoader(self.vllm_config.load_config)
        source = DefaultModelLoader.Source(
            model_or_path=self.model_path,
            revision=self.vllm_config.model_config.revision,
            subfolder="speech_tokenizer",
        )
        subfolder_weights = model_loader._get_weights_iterator(source)
        loaded = AutoWeightsLoader(
            self,
            skip_prefixes=["encoder."],
        ).load_weights(subfolder_weights)

        device = self.vllm_config.device_config.device
        self.decoder.to(device=device, dtype=torch.float32)

        # Precompute SnakeBeta exp caches (benefits both Triton and eager paths)
        if hasattr(self.decoder, "precompute_snake_caches"):
            self.decoder.precompute_snake_caches()

        chunk_frames, left_frames = self._apply_chunk_config()
        self._enable_decoder_cudagraph(self.decoder, device, chunk_frames, left_frames)

        return loaded
