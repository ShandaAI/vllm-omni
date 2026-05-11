# Qwen3-TTS GRPO 训练：vLLM Server 配置结论

日期：2026-05-07

## 结论

GRPO 训练的 vLLM TTS server 应改成 **non-stream deferred** 路径，不做 stage split。

原因很简单：之前正式测试里，non-stream 单卡吞吐高于 stream。既然当前目标是 rollout 吞吐，就不应该把 GRPO 默认配置继续放在 stream 上。

## 要改的配置

当前默认配置 `/data/zxsu/Qwen3-TTS/finetuning/grpo/qwen3_tts_highmem.yaml` 是 stream 串行 Stage1：

```yaml
async_chunk: true

stage_args:
  - stage_id: 1
    engine_args:
      max_num_seqs: 1

runtime:
  defaults:
    max_inflight: 1
  connectors:
    connector_of_shared_memory:
      extra:
        codec_streaming: true
```

这不是吞吐最优配置。

建议新增正式 GRPO server 配置：

```yaml
# /data/zxsu/Qwen3-TTS/finetuning/grpo/qwen3_tts_grpo_nonstream_b6.yaml

async_chunk: false

stage_args:
  - stage_id: 0
    stage_type: llm
    is_comprehension: true
    runtime:
      devices: "0"
    engine_args:
      model_stage: qwen3_tts
      max_num_seqs: 128
      model_arch: Qwen3TTSTalkerForConditionalGeneration
      worker_type: ar
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      enforce_eager: false
      trust_remote_code: true
      async_scheduling: true
      enable_prefix_caching: false
      engine_output_type: latent
      gpu_memory_utilization: 0.65
      distributed_executor_backend: "mp"
      max_num_batched_tokens: 512
      max_model_len: 2048
    output_connectors:
      to_stage_1: connector_of_shared_memory
    default_sampling_params:
      temperature: 0.9
      top_k: 50
      max_tokens: 1024
      detokenize: false
      repetition_penalty: 1.05
      stop_token_ids: [2150]

  - stage_id: 1
    stage_type: llm
    runtime:
      devices: "0"
    engine_args:
      model_stage: code2wav
      max_num_seqs: 6
      model_arch: Qwen3TTSCode2Wav
      worker_type: generation
      scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      enforce_eager: true
      trust_remote_code: true
      async_scheduling: true
      enable_prefix_caching: false
      engine_output_type: audio
      gpu_memory_utilization: 0.15
      distributed_executor_backend: "mp"
      max_num_batched_tokens: 65536
      max_model_len: 65536
    engine_input_source: [0]
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_tts.talker2code2wav
    final_output: true
    final_output_type: audio
    input_connectors:
      from_stage_0: connector_of_shared_memory
    tts_args:
      max_instructions_length: 500
    default_sampling_params:
      temperature: 0.0
      top_p: 1.0
      top_k: -1
      max_tokens: 65536
      detokenize: true
      repetition_penalty: 1.0

runtime:
  enabled: true
  defaults:
    window_size: -1
    max_inflight: 6
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        shm_threshold_bytes: 65536
        codec_streaming: false
        connector_get_sleep_s: 0.01
        connector_get_max_wait_first_chunk: 3000
        connector_get_max_wait: 300
        codec_chunk_frames: 300
        codec_left_context_frames: 25
  edges:
    - from: 0
      to: 1
      window_size: -1
```

启动时：

```bash
export STAGE_CFG=/data/zxsu/Qwen3-TTS/finetuning/grpo/qwen3_tts_grpo_nonstream_b6.yaml
```

## GRPO JSON

`/data/zxsu/Qwen3-TTS/finetuning/grpo/config_grpo.json` 先保持：

```json
"batch_size_per_device": 24,
"num_generations": 4,
"vllm_max_concurrency": 128
```

rollout batch 是 `24 * 4 = 96`，正好是之前正式测试的高并发目标区间。三路 `vllm_server_url` 继续对应三个单卡 server，不做 stage split。

## 实测依据

non-stream 单卡正式矩阵：

| concurrency | main xRT | fork xRT | fork / main |
|---:|---:|---:|---:|
| 1 | 6.419 | 7.658 | 1.193x |
| 4 | 11.952 | 25.260 | 2.114x |
| 8 | 12.337 | 43.329 | 3.512x |
| 16 | 12.652 | 63.258 | 5.000x |
| 24 | 14.577 | 76.580 | 5.253x |
| 32 | 15.923 | 84.447 | 5.304x |
| 48 | 27.367 | 91.275 | 3.335x |
| 64 | 41.558 | 94.169 | 2.266x |
| 96 | 58.820 | 94.106 | 1.600x |

汇总：

- 算术均值 ratio: `3.286x`
- 几何均值 ratio: `2.909x`
- c96 单点：`94.106 / 58.820 = 1.600x`

stream c96 对照：

| suite | main baseline xRT | fork stream xRT | fork / main |
|---|---:|---:|---:|
| deterministic | 45.48 | 67.08 | 1.47x |
| rollout | 50.84 | 77.95 | 1.53x |

因此按吞吐目标，正式 GRPO server 应优先使用 non-stream deferred 配置。

## Streaming + Code2Wav multi-batch CUDA Graph

Streaming 模式（`async_chunk=true`、`codec_streaming=true`）下，Code2Wav 支持同时处理多条请求并使用 CUDA Graph。单卡 H200，deterministic。

三组：`main-baseline`（upstream，Stage1 单请求）、`fork-compatible`（fork，Stage1 单请求）、`fork-code2wav-batch`（fork，Stage1 最多 4 请求）。

### 吞吐（3 repeats）

fork 两组 c1 受 `--log-stats` 统计采集开销影响，数据不可用，已标注。

| 并发 | main xRT | fork-compat xRT | fork-batch xRT | batch vs main |
|---:|---:|---:|---:|---:|
| c1 | 6.70 | 4.89 ⚠️ | 4.89 ⚠️ | -27% ⚠️ |
| c4 | 17.99 | 17.72 | 18.03 | +0.2% |
| c8 | 26.48 | 25.06 | 26.33 | -0.6% |
| c16 | 33.71 | 32.71 | 36.47 | +8.2% |
| c24 | 36.12 | 35.81 | 42.75 | +18.4% |
| c32 | 38.61 | 38.02 | 49.15 | +27.3% |
| c48 | 43.37 | 42.71 | 55.96 | +29.0% |
| c64 | 44.13 | 48.12 | 57.41 | +30.1% |
| c96 | 49.32 | 崩溃 | — | — |

数据：`outputs/vllm_tts_ab_benchmark/runs_perf_*`。

修复统计采集 + batch 策略优化后（1 repeat，带 timing）：

| 并发 | fork-compat xRT | fork-batch xRT | batch vs compat |
|---:|---:|---:|---:|
| c1 | 5.30 | 6.32 | +19.1%（timing 开销） |
| c8 | 21.77 | 26.94 | +23.7% |
| c16 | 32.41 | 33.17 | +2.3% |
| c32 | 39.61 | 48.70 | +22.9% |
| c64 | 43.59 | 56.55 | +29.8% |

数据：`outputs/vllm_tts_ab_benchmark/runs_batch_policy3_*`。

### 延迟（3 repeats）

| 并发 | main p50/p95 | batch p50/p95 | p50 改善 | p95 改善 |
|---:|---:|---:|---:|---:|
| c16 | 6247/7546ms | 5373/7170ms | -14% | -5% |
| c32 | 11820/14424ms | 8420/10966ms | -29% | -24% |
| c48 | 14844/18357ms | 11095/14721ms | -25% | -20% |
| c64 | 16962/24133ms | 13648/17099ms | -20% | -29% |

### 单请求开销

同 GPU、不开 timing、依次跑两组 c1：

| 组 | xRT | p50 | p95 |
|---|---:|---:|---:|
| fork-compatible | 6.72 | 2214ms | 2959ms |
| fork-code2wav-batch | 6.62 | 2256ms | 2998ms |

差距 -1.5%，单请求几乎没有额外开销。数据：`outputs/vllm_tts_ab_benchmark/runs_batch_policy3_c1_pair_notiming`。

### Code2Wav 诊断

| 并发 | 平均 batch | CUDA Graph 命中 | padding 利用率 | ms/frame |
|---:|---:|---:|---:|---:|
| c1 | 1.00 | 100% | 89.8% | 0.918 |
| c8 | 1.74 | 100% | 87.6% | 0.717 |
| c16 | 2.28 | 100% | 86.4% | 0.617 |
| c32 | 2.65 | 100% | 85.6% | 0.551 |
| c64 | 3.47 | 100% | 85.8% | 0.359 |

c1 全部逐条 decode，c64 多数走 batch decode，每帧耗时从 0.92ms 降到 0.36ms。

### batch 策略

只有一条请求时直接走 `decoder.chunked_decode()`，不做任何额外计算。两条及以上时，先看整批 padding 利用率（实际 token 数 / padding 后 token 数）是否达标；达标就整批 decode，否则按 padding 后长度分组，每组单独判断。batch=2 在低负载时要求利用率更高（0.90），高负载时放宽到 0.80，由近期实际 batch 深度的滑动均值决定。阈值可通过环境变量调整（`CODE2WAV_BATCH_MIN_PADDING_EFFICIENCY` 等）。

### 结论

- c32+ 吞吐 +23~30%，p50 延迟 -20~29%。
- 单请求无额外开销（-1.5%）。
- CUDA Graph 100% 命中，padding 利用率 85-90%。
- non-stream B1-deferred c96 单卡到 ~62-84 xRT，streaming batch c64 到 ~56.55 xRT。前者减少 Stage1 调度次数，后者减少 Code2Wav 串行时间，解决的是不同问题。

## 不采用的方向

- stage split：有诊断价值，但当前约束是不分卡。
- stream 作为默认：吞吐低于已测 non-stream，不应作为正式 GRPO 默认。
- non-stream microbatch hook：正式 non-stream deferred 配置不依赖它。
- 微等待 batching：本轮不引入，避免影响 rollout 延迟分布。
