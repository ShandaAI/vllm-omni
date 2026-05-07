# Qwen3-TTS Tuning Notes

日期：2026-05-06

记录口径：保持结论可复查。split 只作为诊断，不计入当前目标；当前目标是不分卡、
non-stream、正式并发矩阵上 `fork / main` 算术均值达到 `2x`。

## Single-GPU Non-Stream 2x 均值矩阵

目标口径：

- 不分卡，Stage0+Stage1 在同一张 GPU；
- codec: non-stream, `codec_chunk_frames=300`, `codec_left_context_frames=25`;
- suite: deterministic;
- concurrency matrix: `1,4,8,16,24,32,48,64,96`;
- requests: each point `128/128`;
- `max_new_tokens=300`;
- baseline: upstream `main-nonstream-mixed`;
- candidate: `fork-nonstream-b12-deferred`;
- 判定：不是要求每个并发点都 `2x`，而是所有并发点 ratio 的算术均值 `>= 2x`。

正式结果路径：

```text
/data/zxsu/Qwen3-TTS/outputs/vllm_tts_ab_benchmark/runs_single_gpu_nonstream_formal_b12_cmatrix_20260506
```

聚合文件：

```text
/data/zxsu/Qwen3-TTS/outputs/vllm_tts_ab_benchmark/runs_single_gpu_nonstream_formal_b12_cmatrix_20260506/summary_aggregate.json
```

| concurrency | main xRT | fork xRT | fork / main | main wall | fork wall |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.419 | 7.658 | 1.193x | 294.4s | 271.7s |
| 4 | 11.952 | 25.260 | 2.114x | 158.1s | 84.2s |
| 8 | 12.337 | 43.329 | 3.512x | 153.6s | 47.2s |
| 16 | 12.652 | 63.258 | 5.000x | 148.6s | 32.8s |
| 24 | 14.577 | 76.580 | 5.253x | 130.4s | 27.0s |
| 32 | 15.923 | 84.447 | 5.304x | 118.8s | 24.3s |
| 48 | 27.367 | 91.275 | 3.335x | 69.4s | 22.4s |
| 64 | 41.558 | 94.169 | 2.266x | 45.5s | 22.2s |
| 96 | 58.820 | 94.106 | 1.600x | 32.1s | 21.9s |

汇总：

- 算术均值 ratio: `3.286x`;
- 几何均值 ratio: `2.909x`;
- 单点未全过 `2x`：c1 为 `1.193x`，c96 为 `1.600x`;
- 按“各并发点均值 2x”口径，当前结果达标。

收益来源：

- Stage1 从 async placeholder 路径改成 deferred/sync `talker2code2wav`，这是 c96 从约 `70.34` 到
  `83.64 xRT` 的主收益；
- non-stream Code2Wav graph 增加 exact `T=300`，避免满 chunk 全部 pad 到 `325`，c96 从
  `83.64` 到 `89.41 xRT`；
- Stage1 `max_num_seqs=12` 让同卡 non-stream 在中高并发下继续提升，c96 达到
  `94.11 xRT`；
- Stage0 CodePredictor tail buckets 覆盖
  `1,2,4,8,16,32,64,80,88,96,128`，减少高并发 tail replay/padding 损耗。

注意：

- 这不是证明每个并发点都 `2x`；
- c1 不适合靠 Code2Wav batching 获得大收益；
- c96 已接近当前单卡同跑下的高并发平台区，继续拉单点需要新的 Stage0 或调度收益。

## Upstream Non-Stream Stage Split 诊断

为判断“分卡收益”是否是 fork 改动造成的，本轮只在 upstream `main`
上做 non-stream codec 对照；Stage1 保持上游默认 `max_num_seqs=1`，
只改变 Stage1 是否独占第二张 GPU。

正式结果路径：

```text
/data/zxsu/Qwen3-TTS/outputs/vllm_tts_ab_benchmark/runs_main_nonstream_stage_split_c96_20260506
```

聚合文件：

```text
/data/zxsu/Qwen3-TTS/outputs/vllm_tts_ab_benchmark/runs_main_nonstream_stage_split_c96_20260506/summary_aggregate.json
```

口径：

- worktree: `/data/zxsu/Qwen3-TTS/vllm-omni-main`
- commit: `33586d84`
- codec: non-stream, `codec_chunk_frames=300`, `codec_left_context_frames=25`
- suite: deterministic
- concurrency: `96`
- requests: `128/128`
- `max_new_tokens=300`
- `--disable-log-stats --quiet-http-logs`

| group | layout | stage1 max_num_seqs | xRT | wall | p50 / p95 |
|---|---|---:|---:|---:|---:|
| `main-nonstream-mixed` | Stage0+Stage1 on one GPU | 1 | 58.52 | 32.40s | 19646 / 30049ms |
| `main-nonstream-split` | Stage0 on GPU0, Stage1 on GPU1 | 1 | 80.50 | 23.58s | 15753 / 20912ms |

结论：

- upstream 上分卡收益为 `80.50 / 58.52 = 1.38x`；
- 这说明 non-stream 单卡 mixed 确实存在明显 Stage0/Stage1 资源竞争；
- 但分卡本身没有达到 `1.5x` 目标；
- 因此不能把“Stage0 已经是唯一瓶颈”当作已证事实。更准确地说，当前证据支持：
  non-stream mixed 的瓶颈是 Stage0 主导 + Stage1/同卡资源竞争，而不是单纯 Code2Wav batch depth。

## Split 结果状态

分卡后追加过一个 fork 诊断：

- 使用 non-stream split 配置；
- Stage1 `max_num_seqs=6`；
- 不恢复 non-stream 专项 planner / policy；
- 只加 Stage0 CodePredictor cudagraph bucket 覆盖：
  `CODE_PREDICTOR_CUDAGRAPH_BATCH_SIZES=1,2,4,8,16,32,64,80,88,96,128`

结果路径：

```text
/data/zxsu/Qwen3-TTS/outputs/vllm_tts_ab_benchmark/runs_fork_nonstream_split_stage0tail_c96_20260506
```

聚合文件：

```text
/data/zxsu/Qwen3-TTS/outputs/vllm_tts_ab_benchmark/runs_fork_nonstream_split_stage0tail_c96_20260506/summary_aggregate.json
```

| baseline / group | xRT | vs upstream mixed |
|---|---:|---:|
| upstream non-stream mixed | 58.52 | 1.00x |
| upstream non-stream split | 80.50 | 1.38x |
| fork non-stream split + Stage0 tail buckets | 92.57 | 1.58x |

结论：

- 这组结果现在只作为诊断，不计入目标；
- split 被标记为不可用后，`92.57 xRT / 1.58x` 不能再宣称达成；
- 该诊断仍说明 non-stream mixed 有明显 Stage0/Stage1 同卡资源竞争。

## Single-GPU Non-Stream 50% 目标

当前有效目标：不分卡，`main-nonstream-mixed` 对比 fork 单卡 non-stream。

共同口径：

- suite: deterministic
- concurrency: `96`
- requests: `128/128`
- `max_new_tokens=300`
- `--disable-log-stats --quiet-http-logs --skip-import-check`
- GPU: single physical GPU1

baseline 路径：

```text
/data/zxsu/Qwen3-TTS/outputs/vllm_tts_ab_benchmark/runs_single_gpu_nonstream_tail_c96_20260506
```

候选最终路径：

```text
/data/zxsu/Qwen3-TTS/outputs/vllm_tts_ab_benchmark/runs_single_gpu_nonstream_deferred_exact300_c96_20260506
```

| group | key change | xRT | wall | p50 / p95 | vs main |
|---|---|---:|---:|---:|---:|
| `main-nonstream-mixed` | upstream main, Stage0+Stage1 same GPU | 59.24 | 32.29s | 21413 / 28691ms | 1.00x |
| `fork-nonstream-b6-mixed` | async_chunk non-stream, Stage0 tail buckets | 70.34 | 29.04s | 19051 / 23607ms | 1.19x |
| `fork-nonstream-b1-mixed` | Stage1 max_num_seqs=1 | 54.66 | 38.05s | 23903 / 34706ms | 0.92x |
| `fork-nonstream-b6-mixed-mem65` | async_chunk, Stage0 mem 0.65 | 73.24 | 28.46s | 18177 / 23923ms | 1.24x |
| `fork-nonstream-b6-deferred` | `async_chunk=false`, sync `talker2code2wav`, no exact 300 graph | 83.64 | 24.47s | 14774 / 18658ms | 1.41x |
| `fork-nonstream-b6-deferred` | deferred + exact Code2Wav `T=300` graph | 89.41 | 22.96s | 14033 / 18471ms | 1.51x |

结论：

- 单卡 non-stream `+50%` 已在 c96/N128 口径下达到：`89.41 / 59.24 = 1.51x`；
- 过线关键不是分卡，也不是更深 Code2Wav B，而是 non-stream 改为 deferred/sync Stage1 输入，避免 async placeholder 路径；
- exact `T=300` Code2Wav graph 把 deferred 从 `83.64` 拉到 `89.41 xRT`，说明 non-stream 满 chunk 不应全部 pad 到 `325`；
- Stage1 B1 更差，说明完全串行 Code2Wav 不是解法；
- 降 Stage0 KV 到 0.65 只有小收益，不能单独过线。

当前保留的 non-stream 修改：

- benchmark config `qwen3_tts_bench_nonstream_b6_deferred.yaml`：non-stream 使用 `async_chunk=false` + `talker2code2wav`；
- `cuda_graph_decoder_wrapper.py`：non-stream capture sizes 显式包含 `decode_chunk_size=300` 和 `300+25=325`；
- runner 组：`fork-nonstream-b6-deferred`。

## 结论

正式矩阵单卡复测下，streaming fork 有稳定收益，但没有达到 2x：

| suite | c96 main xRT | c96 best fork xRT | vs main |
|---|---:|---:|---:|
| deterministic | 46.62 | 71.34 | 1.53x |
| rollout | 47.94 | 72.64 | 1.52x |

最高收益点出现在 rollout c32：

| suite | point | main xRT | best fork xRT | vs main |
|---|---|---:|---:|---:|
| rollout | c32 B6 | 39.00 | 67.92 | 1.74x |

因此当前 stream-only 结论：

- streaming batch/grouping 值得保留；
- 大并发 c96 约 `1.5x`；
- 未达到 2x；
- 后续优化应继续围绕 streaming 的 Stage1 batch 质量、Stage0 tail 调度和 logging overhead。

## 保留的实现方向

- Code2Wav CUDA Graph 支持 compact batch buckets。
- compact batch buckets 默认只用于 `codec_streaming=true`。
- Code2Wav zero-wait grouping 只在 streaming codec 下启用。
- non-stream 不恢复 grouped planner / policy；当前保留的是 deferred/sync Stage1 输入配置，以及 exact `T=300` CUDA Graph capture。

## Stream 正式矩阵复测

复测口径：

- suites: `deterministic,rollout`
- repeats: `3`
- max_new_tokens: `1100`
- concurrency/limit: `c1,c4,c8 limit=32`; `c16 limit=64`; `c24 limit=96`; `c32,c48,c64 limit=128`; `c96 limit=192`
- warmup: `--warmup-limit 32 --warmup-concurrency 16`
- 为避免 `--log-stats`/HTTP access log 污染，复测统一加了 `--disable-log-stats --quiet-http-logs`

结果路径：

```text
/data/zxsu/Qwen3-TTS/outputs/vllm_tts_ab_benchmark/runs_formal_retest_stream_*_20260505
```

| suite | c | main xRT | fork-compatible xRT / vs main | B4 xRT / vs main | B6 xRT / vs main | best fork vs main |
|---|---|---:|---:|---:|---:|---|
| deterministic | c1 | 6.70 | 7.20 / 1.07x | 7.21 / 1.08x | 7.25 / 1.08x | B6 1.08x |
| deterministic | c4 | 18.12 | 21.39 / 1.18x | 21.85 / 1.21x | 21.95 / 1.21x | B6 1.21x |
| deterministic | c8 | 24.97 | 30.46 / 1.22x | 32.07 / 1.28x | 34.15 / 1.37x | B6 1.37x |
| deterministic | c16 | 33.57 | 43.02 / 1.28x | 44.90 / 1.34x | 43.65 / 1.30x | B4 1.34x |
| deterministic | c24 | 35.91 | 54.03 / 1.50x | 58.00 / 1.62x | 52.31 / 1.46x | B4 1.62x |
| deterministic | c32 | 38.27 | 60.05 / 1.57x | 64.16 / 1.68x | 66.00 / 1.72x | B6 1.72x |
| deterministic | c48 | 43.49 | 61.40 / 1.41x | 65.22 / 1.50x | 68.84 / 1.58x | B6 1.58x |
| deterministic | c64 | 46.90 | 63.29 / 1.35x | 69.70 / 1.49x | 66.71 / 1.42x | B4 1.49x |
| deterministic | c96 | 46.62 | 62.88 / 1.35x | 66.28 / 1.42x | 71.34 / 1.53x | B6 1.53x |
| rollout | c1 | 6.71 | 7.11 / 1.06x | 7.10 / 1.06x | 7.16 / 1.07x | B6 1.07x |
| rollout | c4 | 18.06 | 21.14 / 1.17x | 21.13 / 1.17x | 21.43 / 1.19x | B6 1.19x |
| rollout | c8 | 25.81 | 31.54 / 1.22x | 32.43 / 1.26x | 32.67 / 1.27x | B6 1.27x |
| rollout | c16 | 34.38 | 44.93 / 1.31x | 49.32 / 1.43x | 46.19 / 1.34x | B4 1.43x |
| rollout | c24 | 34.97 | 57.55 / 1.65x | 55.06 / 1.57x | 55.08 / 1.58x | fork-compatible 1.65x |
| rollout | c32 | 39.00 | 67.47 / 1.73x | 62.74 / 1.61x | 67.92 / 1.74x | B6 1.74x |
| rollout | c48 | 43.61 | 70.39 / 1.61x | 72.90 / 1.67x | 65.67 / 1.51x | B4 1.67x |
| rollout | c64 | 46.72 | 66.40 / 1.42x | 72.87 / 1.56x | 62.21 / 1.33x | B4 1.56x |
| rollout | c96 | 47.94 | 72.64 / 1.52x | 70.83 / 1.48x | 72.41 / 1.51x | fork-compatible 1.52x |

## Stream Stage1 max_num_seqs 拆分补测

补测原因：

- 之前 stream baseline 是上游默认 `Stage1 max_num_seqs=1`；
- fork 的 B6/B4 结果同时包含了 `Stage1 max_num_seqs>1` 和 fork Code2Wav streaming grouped decode 等改动；
- 因此必须补一个 `upstream main + streaming + Stage1 max_num_seqs=6`，拆出“上游已有并发能力”本身的收益。

补测口径：

- date: 2026-05-07
- suite: deterministic, rollout
- concurrency: `96`
- requests: `192/192`
- `max_new_tokens=1100`
- single GPU1, same run
- groups: `main-baseline`, `main-stream-b6`, `fork-code2wav-batch6`

结果路径：

```text
/data/zxsu/Qwen3-TTS/outputs/vllm_tts_ab_benchmark/runs_stream_main_b6_isolation_c96_20260507
```

聚合文件：

```text
/data/zxsu/Qwen3-TTS/outputs/vllm_tts_ab_benchmark/runs_stream_main_b6_isolation_c96_20260507/summary_aggregate.json
```

| suite | main baseline xRT | main stream B6 xRT / vs baseline | fork B6 xRT / vs baseline | fork B6 / main stream B6 |
|---|---:|---:|---:|---:|
| deterministic | 45.48 | 52.64 / 1.16x | 67.08 / 1.47x | 1.27x |
| rollout | 50.84 | 52.84 / 1.04x | 77.95 / 1.53x | 1.48x |

结论：

- 之前 stream 对比的 baseline 确实是 `Stage1 max_num_seqs=1`；
- 上游 main 本身支持把 Stage1 配成 `max_num_seqs=6`，但在 c96 下收益有限：
  deterministic `+15.8%`，rollout `+3.9%`；
- fork B6 相对更公平的 `main-stream-b6` baseline 仍有明显收益：
  deterministic `+27.4%`，rollout `+47.5%`；
- 因此 stream fork 的收益不能全部归因于 `max_num_seqs>1`，但之前“vs main baseline”的总提升确实混入了这部分，需要按上表拆分。

## 推荐复测命令

```bash
python /data/zxsu/Qwen3-TTS/finetuning/grpo/benchmark/run_vllm_tts_ab.py \
  --output-dir /data/zxsu/Qwen3-TTS/outputs/vllm_tts_ab_benchmark/runs_stream_only_retest \
  --gpu 4 \
  --port 8103 \
  --groups main-baseline,fork-compatible,fork-code2wav-batch,fork-code2wav-batch6 \
  --suites deterministic,rollout \
  --concurrencies 96 \
  --repeats 3 \
  --limit 192 \
  --warmup-limit 32 \
  --warmup-concurrency 16 \
  --disable-log-stats \
  --quiet-http-logs \
  --continue-on-client-errors
```
