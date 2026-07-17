# Execution Plan — 4 Weeks × ~20 h/week

Compressed from the original 6–8 week proposal. **Main cut: TensorRT-LLM moves to future work** — the project ships as a vLLM-only comparison (FP16 vs FP8 vs AWQ), which is still a complete result per the proposal's own fallback plan.

**Model:** `Qwen/Qwen2.5-3B-Instruct` (+ official `Qwen2.5-3B-Instruct-AWQ`)
**Hardware:** RTX 4060 8 GB (Ada, SM 8.9 → native FP8), WSL2 Ubuntu

> 8 GB VRAM note: 3B FP16 (~6.2 GB weights) leaves ~1 GB KV cache — FP16 will hit
> its concurrency ceiling early. That contrast (quantization → freed VRAM → bigger
> batches) is a headline finding, not a limitation. Stretch goal: 7B-AWQ vs 3B-FP16
> ("bigger-quantized vs smaller-full-precision") if time allows in week 3.

## Week 1 — Environment + FP16 baseline (~20 h)

| Block | Task | ~h |
|---|---|---|
| 1 | WSL2 GPU passthrough check, Python env, install vLLM + GenAI-Perf (`setup/`) | 4 |
| 2 | Serve Qwen2.5-7B FP16 with vLLM; smoke-test OpenAI API | 3 |
| 3 | Define 2 workloads + SLA targets in `configs/workloads.yaml` (interactive vs batch) | 3 |
| 4 | Drive load with GenAI-Perf against both workloads; collect TTFT/TPOT/throughput/p95 | 6 |
| 5 | Parse results into CSV (`scripts/parse_results.py`); commit baseline numbers + week1 log | 4 |

**Deliverable:** FP16 baseline table for both workloads, pushed to GitHub.

## Week 2 — Quant sweep + concurrency sweep + plots (~20 h)

| Block | Task | ~h |
|---|---|---|
| 1 | FP8 (vLLM online `--quantization fp8`) sweep | 4 |
| 2 | AWQ (official AWQ checkpoint) sweep | 4 |
| 3 | Concurrency sweep (1→64) per config per workload, automated via `scripts/run_sweep.sh` | 6 |
| 4 | Plot pipeline: latency-vs-throughput curves, one line per config, SLA line annotated | 6 |

**Deliverable:** the "money chart" (speed only, quality annotation comes in Week 3).

## Week 3 — Quality harness + cost model + recommendation (~20 h)

| Block | Task | ~h |
|---|---|---|
| 1 | Small eval set (~100 samples, summarization or QA) + quality harness (ROUGE / exact-match + optional LLM-judge) | 8 |
| 2 | Run quality eval per quant config → degradation table (**Contribution 1 done**) | 4 |
| 3 | $/1M-token cost model anchored to a published cloud GPU hourly price | 3 |
| 4 | SLA-constrained Pareto search → recommendation output (**Contribution 2 done**) | 5 |

**Deliverable:** `advisor recommend --workload interactive --sla ...` returns config + GPU count + $/1M tokens.

## Week 4 — Profiling deep-dive + write-up (~20 h)

| Block | Task | ~h |
|---|---|---|
| 1 | Nsight Systems + torch.profiler on ONE representative config pair (e.g., FP16 vs FP8 at high concurrency) | 8 |
| 2 | Case-study write-up: memory-bound vs compute-bound, KV-cache pressure (**Contribution 3 done**) | 4 |
| 3 | README polish, final money chart with quality annotations, repro instructions | 6 |
| 4 | Buffer | 2 |

**Deliverable:** portfolio-ready repo.

## Scope guards

- If FP8 on vLLM misbehaves on Ada → report it as a finding, continue with FP16 vs AWQ.
- If LLM-as-judge takes too long → ship ROUGE/exact-match only, judge as future work.
- Nsight in WSL2 can be finicky → torch.profiler alone is an acceptable fallback for Contribution 3.
- CI regression harness from the original plan → future work.

## GitHub progress convention

- One branch per week is unnecessary — work on `main`, tag `week1`…`week4` at each week's end.
- Each week ends with `docs/weekly-log/weekN.md` (template provided) committed with the numbers/plots produced that week.
