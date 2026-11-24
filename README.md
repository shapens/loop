# loop

Minimal, reproducible **multi-turn coding RLVR** engine and empirical study under the `shapens` org.

> **Thesis:** Multi-turn coding RL breaks in three measurable places—**(1) sandbox tail latency** starving GPUs, **(2) live reward hacking** against naive coding verifiers, and **(3) train–inference mismatch (TIM) & policy staleness** destabilizing GRPO. `loop` measures and solves all three end-to-end on <$300 of burst GPU compute.

## Architecture

```text
src/loop/
├── types.py          # Strict data contracts (TaskSpec, Trajectory, VerifierReceipt, TIMStats)
├── envs/
│   ├── generator.py  # Contamination-free parametric multi-turn coding task suites (train / held-out eval)
│   ├── grader.py     # Anti-cheat verifier (pristine test overlay + hidden tests + reward-hack flags)
│   └── sandbox.py    # Pluggable SandboxBackend: LocalSubprocess, ModalSandbox, and shapens/tri seam
├── rollout/
│   ├── protocol.py   # Multi-turn coding tools + exact token-ID trajectory builder with binary loss_mask
│   └── vllm_worker.py# Decoupled vLLM rollout client (token IDs + logprobs=1 + weight-sync hook)
├── trainer/
│   ├── grpo.py       # From-scratch PyTorch GRPO + observation masking + TIM logging + MIS/CIS ablation
│   └── weight_sync.py# SHA-256 parameter digest verification (zero silent staleness)
├── eval/
│   └── stats.py      # Unbiased pass@k + 95% bootstrap CIs + wall-clock GPU utilization breakdown
└── modal_app.py      # Ephemeral Modal Sandbox pool + GPU burst runner
```

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,ml,modal]"

# 1. Verify every generated task fails untouched, passes reference solution, and catches reward hacking
loop self-check --count 30

# 2. Run unit tests (grader anti-cheat, PyTorch GRPO numerics, TIM/MIS, batched rollout, pass@k CIs)
pytest

# 3. Cheap (~$0.05) Modal GPU smoke test
modal run src/loop/modal_app.py::gpu_smoke_test

# 4. Run batched multi-turn vLLM baseline eval on Modal L40S (~$0.15-$0.30 for 30 tasks x 4 samples)
modal run src/loop/modal_app.py::baseline_eval \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --num-tasks 30 --samples-per-task 4

# 5. Run co-located H100 GRPO + vLLM LoRA weight-sync training (~$0.80-$1.50 for 12 steps)
modal run src/loop/modal_app.py::train_grpo \
  --run-name grpo_qwen15b_mis --steps 12 --use-mis true --grader-mode hardened
```

## Compute Strategy

* **M1–M3 (All-Modal):** Ephemeral Modal Sandboxes + burst H100/A100 GPU functions sharing Modal Volumes (pay-per-second, zero idle burn, ~$2–$5 per 1.5B–3B ablation).
* **M4 (Optional RunPod/Linux bare-metal VM):** Native Linux host with `fuse3` + `gVisor`/`Kata` to benchmark **`shapens/tri`** copy-on-write workspaces head-to-head against standard container cold-starts.

See [ARCH.md](ARCH.md) for the exact invariants and milestone checklist.
