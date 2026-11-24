# loop

[![CI](https://github.com/shapens/loop/actions/workflows/ci.yml/badge.svg)](https://github.com/shapens/loop/actions/workflows/ci.yml)

Minimal, from-scratch **multi-turn coding RLVR engine** and empirical study under [`shapens`](https://github.com/shapens) (solo author).

> **Thesis:** Multi-turn coding RL breaks in three measurable places—**(1) sandbox tail latency** starving GPUs, **(2) live reward hacking / Goodharting** against naive verifiers, and **(3) train–inference mismatch (TIM) & vocab HBM pressure** during policy updates. `loop` builds the full stack from scratch (PyTorch GRPO + custom Triton fused logprob kernel + vLLM LoRA hot-reload + hardened/`shapens/tri` sandboxes) and proves all three empirically on <$10 of burst Modal H100 compute.

Full technical write-up (including what broke first & negative/nuanced results): **[docs/memo.md](docs/memo.md)**

---

## Empirical Highlights (All Reproduced on NVIDIA H100 80GB)

### 1. Headline 7B Multi-Turn RLVR Across 5 Coding Families (Live Triton Kernel)
`Qwen2.5-Coder-7B-Instruct` · 20 GRPO steps · 50 held-out tasks × 4 samples (200 episodes) · `pass@1` **`32.5% → 52.0%` (+19.5 pp)** · `pass@4` **`50.0% → 64.0%`** · `async_keyed_queue` (asyncio lock-handover race) **`7.5% → 57.5%` (+50 pp)**:

![Qwen2.5-Coder-7B 5-Family RLVR Results](docs/assets/headline_7b_5fam.svg)

### 2. Controlled Goodhart / Reward-Hacking Ablation (Hardened vs. Naive Verifier)
`Qwen2.5-Coder-3B-Instruct` · 20 GRPO steps · Hardened lifts true `pass@1` **5.75×** (`6.7% → 38.3%`, non-overlapping 95% CIs) with emergent multi-turn depth (`2.82 → 4.68` turns); Naive hits **100% fake train reward** while true `pass@1` stalls at `19.2%` and `ledger_invariants` collapses to `0%`:

![Hardened vs Naive Goodhart Divergence](docs/assets/ablation_goodhart.svg)

### 3. Systems Wins: Custom Triton Kernel (2.67× H100) & `shapens/tri` Sandbox Latency (7.0× p50)
* **Custom Triton online-LSE logprob kernel** ([triton_logprob.py](src/loop/trainer/triton_logprob.py)) on Qwen's `152,064` vocab `[2, 2048, 152064]` `bf16`: **`2.67×` faster** (`14.85 ms → 5.57 ms`), **`3.48 GB` peak HBM saved (-37.5%)**, `fwd err = 9.5e-7`, `bwd err = 4.7e-10`.
* **`shapens/tri` HTTP copy-on-write workspaces** ([sandbox.py](src/loop/envs/sandbox.py)) vs. ephemeral Modal Sandboxes: **`7.0×` lower p50** (`148 ms` vs `1,032 ms`), **`8.2×` lower p95**, and **`2.27×` higher conc=8 throughput** (`16.95/s` vs `7.48/s`).

![Systems Benchmarks: Sandbox Latency & Custom Triton Kernel](docs/assets/systems_bench.svg)

### 4. Async Off-Policy Staleness (`lag = 3`, `lr = 8e-5`): Vanilla GRPO vs. MIS
Under 3-step rollout staleness, max $|\Delta \log p|$ rises ~5× (`1.49–1.66`), MIS actively masks tail spikes (`r > 1.5`), and both policies climb from `~6% → 36.5–37.0%` `pass@1` across 50 held-out tasks while `sandbox_ms` reaches **32%–67% of step wall-clock time**:

![Async Off-Policy Staleness: Vanilla GRPO vs MIS](docs/assets/ablation_async_mis.svg)

---

## Architecture

```text
src/loop/
├── types.py                  # Strict data contracts (TaskSpec, Trajectory, VerifierReceipt, TIMStats)
├── bench_sandboxes.py        # p50/p95/p99 & throughput concurrency benchmark harness
├── envs/
│   ├── generator.py          # 5 parametric, contamination-free coding families (train / held-out eval)
│   ├── grader.py             # Hardened (python3 -I -S + pristine overlay + hidden tests) vs naive mode
│   └── sandbox.py            # SandboxBackend protocol: LocalSubprocess, ModalSandbox, TriSandboxBackend
├── rollout/
│   ├── protocol.py           # Escape-free XML + JSON coding tools + exact token-ID TrajectoryBuilder
│   ├── runner.py             # Lockstep batched multi-turn rollout engine (ChatML + binary loss_mask)
│   └── vllm_worker.py        # OpenAI-compatible vLLM client with exact token_id return + LoRA reload
├── trainer/
│   ├── grpo.py               # From-scratch PyTorch GRPO + observation masking + TIM stats + MIS/k3-KL
│   ├── triton_logprob.py     # Custom Triton online-LSE fused per-token logprob kernel (fwd + bwd)
│   ├── weight_sync.py        # Raw-byte SHA-256 parameter digest (zero silent staleness)
│   ├── engine.py             # Gradient-accumulated GRPO step + adapter export SHA-256 verification
│   └── modal_train.py        # Co-located H100 vLLM + PEFT LoRA loop (zero-VRAM ref KL + stale_steps)
├── eval/
│   ├── stats.py              # Unbiased pass@k + deterministic 95% percentile bootstrap CIs
│   └── baseline.py           # Per-family pass@k, tamper rate, turn depth, and latency telemetry
└── modal_app.py              # Modal GPU entrypoints (smoke test, Triton bench, baseline eval, GRPO train)
```

See [ARCH.md](ARCH.md) for the 5 enforced invariants and [docs/results-qwen3b-ablation.md](docs/results-qwen3b-ablation.md) for detailed ablation tables.

---

## Quickstart & Reproduction

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,ml,modal]"

# 1. Verify all 5 task families fail untouched, pass reference fix, and block visible-test tampering
loop self-check --count 20

# 2. Run strict lint, typecheck, and unit tests
ruff check src tests && mypy src tests && pytest -v

# 3. Benchmark the custom Triton kernel on Modal H100 (~$0.05)
modal run src/loop/modal_app.py::bench_triton --batch-size 2 --seq-len 2048 --vocab-size 152064

# 4. Reproduce the 7B 5-family H100 GRPO run (powered by the live Triton kernel)
modal run src/loop/modal_app.py::train_grpo \
  --run-name grpo_7b_5fam_hardened \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --steps 20 --tasks-per-step 5 --group-size 8 \
  --micro-batch-size 1 --eval-every 5 --eval-tasks 50 --eval-samples 4

# 5. Regenerate all SVG charts in docs/assets/ from results/*.json
python scripts/render_charts.py
```
