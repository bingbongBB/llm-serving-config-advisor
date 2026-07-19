# Week 1 — Environment + FP16 baseline

**Hours spent:** __ / 20
**Status:** ☑ done (batch sweep pending)

## Goals (from PLAN.md)

- [x] WSL2 GPU passthrough verified (`setup/01_check_env.sh`)
- [x] vLLM 0.25.1 + genai-perf 0.0.16 installed, pinned in `requirements.lock`
- [x] Qwen2.5-3B FP16 served, API smoke-tested
- [x] Workloads + SLA defined in `configs/workloads.yaml`
- [x] Interactive concurrency sweep (1→32) complete
- [ ] Batch summarization sweep (in progress)

## Baseline numbers (FP16, interactive workload)

SLA: TTFT p95 < 500 ms, TPOT p95 < 50 ms.

| Concurrency | TTFT p95 (ms) | TPOT p95 (ms) | Out tok/s | Within SLA? |
|---|---|---|---|---|
| 1 | 89 | 26.9 | 38 | yes |
| 2 | 135 | 28.1 | 66 | yes |
| 4 | 158 | 29.4 | 129 | yes |
| 8 | 195 | 31.6 | 229 | yes |
| 16 | 216 | 36.4 | 428 | yes |
| 32 | 5,436 | 35.9 | 458 | **NO — queueing** |

## Key findings

1. **Decode is memory-bound, batching is near-free.** Throughput scales almost
   linearly from concurrency 1→16 (38 → 428 tok/s) while TPOT only degrades
   26 → 33 ms: continuous batching amortizes weight reads across requests.
   Single-stream 38 tok/s ≈ 86% of the bandwidth roofline
   (272 GB/s ÷ 6.2 GB weights ≈ 44 tok/s theoretical) — profiling target for week 4.
2. **Queueing wall at concurrency 32.** TTFT p95 explodes 216 ms → 5.4 s while
   throughput gains only 7% (428 → 458 tok/s). Root-cause chain:
   8 GB VRAM − 5.79 GB weights − overheads ⇒ ~0.5 GB KV cache ⇒
   `--max-num-seqs` capped at 16 ⇒ requests 17–32 wait in queue.
3. **FP16 SLA capacity on this machine: concurrency 16, ~430 tok/s.**
   Week 2 hypothesis: FP8/AWQ free ~3 GB of weights for KV cache, relaxing the
   seq cap and pushing the queueing wall right.

## Blockers & how resolved

1. vLLM 0.25 requires a newer driver than 560.94 (CUDA 12.6) → updated the
   **Windows** driver to 610.43 (CUDA 13.3); WSL2 needs no Linux driver.
2. Windows desktop reserves ~0.5–1 GB VRAM and free memory floats 6.9–7.4 GB →
   `--gpu-memory-utilization` capped at 0.86; check `nvidia-smi` before runs.
3. vLLM V2 Model Runner needs UVA, unsupported in WSL2 CUDA →
   `VLLM_USE_V2_MODEL_RUNNER=0`.
4. FlashInfer sampler JIT-compiles with nvcc, but no CUDA toolkit installed →
   `VLLM_USE_FLASHINFER_SAMPLER=0` (toolkit install deferred to week 4 profiling).
5. No memory left for KV cache after weights → `--enforce-eager` (frees the
   0.44 GB CUDA-graph reserve) + `--max-num-batched-tokens 1024` +
   `--max-num-seqs 16`. All configs share these flags → internally fair;
   absolute numbers slightly below CUDA-graph mode (noted in README).
6. Process lesson: edited scripts on both Mac and server → divergence, one
   benchmark ran a stale config. New rule: pull before editing, push right after.

## Next week

- FP8 (`--quantization fp8`) and AWQ sweeps on both workloads
- Record `GPU KV cache size` from vLLM startup log per config
- Plot pipeline: latency-vs-throughput curves per config with SLA line (money chart v1)
