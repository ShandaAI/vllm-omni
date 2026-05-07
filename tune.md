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

## 不采用的方向

- stage split：有诊断价值，但当前约束是不分卡。
- stream 作为默认：吞吐低于已测 non-stream，不应作为正式 GRPO 默认。
- non-stream microbatch hook：正式 non-stream deferred 配置不依赖它。
- 微等待 batching：本轮不引入，避免影响 rollout 延迟分布。
