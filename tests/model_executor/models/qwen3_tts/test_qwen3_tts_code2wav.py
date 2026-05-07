# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import (
    Qwen3TTSCode2Wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NUM_QUANTIZERS = 2
_TOTAL_UPSAMPLE = 4
_OUTPUT_SAMPLE_RATE = 24000


class _FakeDecoder(nn.Module):
    def __init__(self, total_upsample: int = _TOTAL_UPSAMPLE):
        super().__init__()
        self.total_upsample = total_upsample

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        frames = codes.shape[-1]
        wav_len = frames * self.total_upsample + 6
        wav = torch.arange(wav_len, dtype=torch.float32)
        return wav.view(1, 1, -1)

    def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self(codes)


class _FakeBatchedDecoder(_FakeDecoder):
    def __init__(self, total_upsample: int = _TOTAL_UPSAMPLE):
        super().__init__(total_upsample=total_upsample)
        self.batched_calls: list[list[int]] = []

    def batched_chunked_decode(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        self.batched_calls.append([int(codes.shape[-1]) for codes in codes_list])
        return [self.chunked_decode(codes) for codes in codes_list]


class _FakeCUDAGraphWrapper:
    def __init__(self, max_batch_size: int, batch_buckets: list[int], size_buckets: list[int] | None = None):
        self.max_batch_size = max_batch_size
        self.capture_batch_sizes = batch_buckets
        self.capture_sizes = size_buckets or []
        self.decode_calls: list[int] = []
        self.batched_decode_calls: list[list[int]] = []

    def _get_padded_size(self, actual_size: int) -> int | None:
        if not self.capture_sizes:
            return actual_size
        for bucket in self.capture_sizes:
            if actual_size <= bucket:
                return bucket
        return None

    def _get_padded_batch_size(self, actual_batch_size: int) -> int | None:
        for bucket in self.capture_batch_sizes:
            if actual_batch_size <= bucket:
                return bucket
        return None

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        frames = int(codes.shape[-1])
        self.decode_calls.append(frames)
        return torch.arange(frames * _TOTAL_UPSAMPLE + 6, dtype=torch.float32).view(1, 1, -1)

    def batched_decode(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        frames = [int(codes.shape[-1]) for codes in codes_list]
        self.batched_decode_calls.append(frames)
        return [
            torch.arange(frame * _TOTAL_UPSAMPLE + 6, dtype=torch.float32).view(1, 1, -1)
            for frame in frames
        ]


def _fake_dec_config():
    return SimpleNamespace(
        num_quantizers=_NUM_QUANTIZERS,
        sliding_window=0,
    )


def _make_model(decoder: nn.Module | None = None) -> Qwen3TTSCode2Wav:
    decoder = decoder or _FakeDecoder()
    dec_config = _fake_dec_config()
    tok_config = SimpleNamespace(
        decoder_config=dec_config,
        output_sample_rate=_OUTPUT_SAMPLE_RATE,
    )
    with (
        patch(
            "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.Qwen3TTSTokenizerV2Config.from_pretrained",
            return_value=tok_config,
        ),
        patch(
            "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.Qwen3TTSTokenizerV2Decoder._from_config",
            return_value=decoder,
        ),
    ):
        model = Qwen3TTSCode2Wav(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(model="unused"),
                device_config=SimpleNamespace(device=torch.device("cpu")),
            )
        )
    return model


def test_forward_trims_context_on_exact_frame_boundaries():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"meta": {"left_context_size": 2}}],
    )

    audio = out.multimodal_outputs["model_outputs"][0]
    expected = torch.arange(8, 24, dtype=torch.float32)
    torch.testing.assert_close(audio, expected)


def test_forward_trims_trailing_padding_without_context():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[{"meta": {"left_context_size": 0}}],
    )

    audio = out.multimodal_outputs["model_outputs"][0]
    expected = torch.arange(24, dtype=torch.float32)
    torch.testing.assert_close(audio, expected)


def test_forward_echoes_full_audio_codes_for_codec_token_return():
    model = _make_model()

    out = model.forward(
        input_ids=torch.arange(12, dtype=torch.long),
        runtime_additional_information=[
            {"full_audio_codes": [[1, 2], [3, 4]]},
            {"full_audio_codes": torch.tensor([[5, 6], [7, 8], [9, 10]])},
        ],
        seq_token_counts=[4, 8],
    )

    audio_codes = out.multimodal_outputs["audio_codes"]
    assert len(audio_codes) == 2
    torch.testing.assert_close(audio_codes[0], torch.tensor([[1, 2], [3, 4]]))
    torch.testing.assert_close(audio_codes[1], torch.tensor([[5, 6], [7, 8], [9, 10]]))


def test_streaming_bucket_histogram_reports_outer_graph_bucket():
    decoder = _FakeBatchedDecoder()
    wrapper = _FakeCUDAGraphWrapper(
        max_batch_size=8,
        batch_buckets=[1, 2, 3, 4, 6, 8],
        size_buckets=[25, 64, 97],
    )
    decoder._cudagraph_wrapper = wrapper
    model = _make_model(decoder)
    model._codec_streaming = True

    _, metrics = model._decode_code2wav_grouped(
        decoder,
        [torch.arange(148, dtype=torch.long).reshape(2, 74) for _ in range(8)],
    )

    assert metrics["bucket_hist"] == "B8>8x97:1"
