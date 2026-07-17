# LLM Serving Config Advisor

> Given a model, a workload profile, and your **own** evaluation set, this tool benchmarks serving configurations and recommends the deployment that is **cheapest within the SLA while keeping answer quality acceptable** — including required GPU count and $ / 1M tokens.

Existing benchmarking tools (GenAI-Perf/AIPerf, Triton Model Analyzer, vLLM benchmarks) measure **speed only**. They don't tell you what quantization or batching does to *task quality on your data*, and they don't turn measurements into an SLA-constrained, cost-aware recommendation. This project builds that missing layer — reusing GenAI-Perf as the load generator instead of reinventing it.

## Three contributions

1. **Quality-aware joint profiling** — latency/throughput/memory **plus task-quality degradation** per config, on your own eval set.
2. **Workload-aware Pareto recommendation** — `(QPS, length distribution) + SLA → config + GPU count + $/1M tokens`, via a benchmark-calibrated capacity model and Pareto-frontier search.
3. **Performance deep-dive** — Nsight Systems + torch.profiler case study explaining *why* the winning config wins.

## v1 scope

| Dimension | Choice |
|---|---|
| Model | Qwen2.5-3B-Instruct |
| Backend | vLLM (TensorRT-LLM: future work) |
| Load generator | NVIDIA GenAI-Perf |
| Quantization | FP16 vs FP8 vs INT4 (AWQ) + concurrency sweep |
| Workloads | interactive low-latency / batch high-throughput (see `configs/workloads.yaml`) |
| Quality | ROUGE / exact-match (+ optional LLM-as-judge) on a small custom eval set |
| Hardware | 1× RTX 4060 8 GB (Ada SM 8.9 → native FP8), WSL2 Ubuntu |

## Repo layout

```
setup/        environment install scripts (WSL2 + CUDA check, vLLM, GenAI-Perf)
configs/      workload + SLA definitions, sweep matrix
scripts/      serve / benchmark / parse / plot
advisor/      quality harness, cost model, Pareto recommendation (built weeks 3)
results/      raw benchmark output + parsed CSVs (committed per week)
docs/         PLAN.md + weekly progress logs
```

## Quickstart (Week 1)

```bash
bash setup/01_check_env.sh          # verify GPU visible in WSL2
bash setup/02_install.sh            # venv + vLLM + genai-perf
bash scripts/serve_vllm.sh fp16     # terminal 1: serve the model
bash scripts/run_benchmark.sh fp16 interactive 8   # terminal 2: config, workload, concurrency
python scripts/parse_results.py results/raw/ -o results/summary.csv
```

## Status

- [ ] Week 1 — FP16 baseline ([log](docs/weekly-log/week1.md))
- [ ] Week 2 — Quant + concurrency sweep, plots
- [ ] Week 3 — Quality harness, cost model, recommendation
- [ ] Week 4 — Profiling deep-dive, write-up

See [docs/PLAN.md](docs/PLAN.md) for the full 4-week plan.
