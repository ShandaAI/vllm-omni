# Qwen3-TTS GRPO 训练：vLLM Server 配置方案

日期：2026-05-07

## 背景

GRPO 训练时 rollout 会产生大量并发的 TTS 推理请求。我们 fork 版 vllm-omni 在 Code2Wav 阶段支持了 streaming CUDA Graph batching，可以把多条请求拼成一个 batch 一起跑，吞吐提升明显。

本方案的核心决策：

- 不做 stage split（不分卡），全部在单卡上完成 AR + Code2Wav
- 只走 streaming codec 路径，不考虑 non-stream

## 问题：当前配置没吃到 batching 收益

当前 GRPO server 配置（`finetuning/grpo/qwen3_tts_highmem.yaml`）：

```yaml
stage_args:
  - stage_id: 1
    engine_args:
      max_num_seqs: 1

runtime:
  defaults:
    max_inflight: 1
```

Code2Wav 阶段同时只处理 1 条请求，完全串行，等于白白浪费了 batching 能力。

## 建议的正式配置

新建配置文件 `finetuning/grpo/qwen3_tts_grpo_stream_b6.yaml`：

```yaml
async_chunk: true

stage_args:
  - stage_id: 0
    engine_args:
      max_num_seqs: 128
      max_num_batched_tokens: 512
      max_model_len: 2048
      gpu_memory_utilization: 0.8

  - stage_id: 1
    engine_args:
      max_num_seqs: 6
      max_num_batched_tokens: 65536
      max_model_len: 65536
      gpu_memory_utilization: 0.15

runtime:
  defaults:
    max_inflight: 6
  connectors:
    connector_of_shared_memory:
      extra:
        codec_streaming: true
        codec_chunk_frames: 25
        codec_left_context_frames: 72
```

关键参数说明：

- `max_num_seqs: 6` — Code2Wav 同时最多处理 6 条，已在拆分测试中验证稳定
- `max_inflight: 6` — 与 max_num_seqs 一致，控制同时在途请求数
- `codec_streaming: true` — 走流式 codec 路径，支持 CUDA Graph batching

启动时设置环境变量，预编译对应 batch size 的 CUDA Graph：

```bash
export STAGE_CFG=/data/zxsu/Qwen3-TTS/finetuning/grpo/qwen3_tts_grpo_stream_b6.yaml
export CODE2WAV_STREAMING_CUDAGRAPH_BATCH_SIZES=1,2,3,4,6
```

如果显存和稳定性没问题，后续可以尝试 `max_num_seqs=8 / max_inflight=8`，对应：

```bash
export CODE2WAV_STREAMING_CUDAGRAPH_BATCH_SIZES=1,2,3,4,6,8
```

## GRPO 训练参数（暂不需要改）

`finetuning/grpo/config_grpo.json` 中：

```json
"batch_size_per_device": 24,
"num_generations": 4,
"vllm_max_concurrency": 128
```

每轮 rollout 产生 24×4=96 条并发请求，正好是 stream batching 收益明显的区间。三路 `vllm_server_url` 继续保留（对应三张卡各跑一个 server），不改成 stage split。

后续可以评估 `max_new_tokens` 是否从 1100 收到 900 左右（训练数据 `max_audio_length=800`），但必须先确认不会截断有效样本。

## 性能实测数据

streaming c96 拆分测试（xRT = 实时率，越高越好）：

| 测试场景 | 原始 baseline | 原始 stream (seqs=6) | fork stream (seqs=6) | fork 相对提升 |
|---|---:|---:|---:|---:|
| deterministic | 45.48 | 52.64 | 67.08 | 1.27× |
| rollout | 50.84 | 52.84 | 77.95 | 1.48× |

结论：

- 上游 stream 支持多并发，但收益有限（+4%~16%）
- fork 的 CUDA Graph grouped batching 额外带来 27%~48% 的提升
- 如果 GRPO server 继续用 `max_num_seqs=1`，这部分收益全部浪费

## 本轮不做的事情

| 方案 | 为什么不做 |
|---|---|
| non-stream codec | 正式训练只走 stream 路径，non-stream 不作为默认 |
| stage split（分卡） | 约束明确要求不分卡，保留为诊断参考 |
| non-stream microbatch hook | stream 路径用不上 |
| 微等待 batching（等一小段时间凑更大 batch） | 需要单独评估对 latency 和 reward 的影响，暂不引入 |
| CodePredictor compact/tail buckets | 对 stream 路径不是稳定的主要收益来源 |
