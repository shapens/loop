# Architecture & Non-Negotiable Invariants

`loop` is deliberately small and inspectable (no `TRL` trainer black box). Five invariants are enforced by automated tests:

1. **Exact Token-ID Handoff (No Re-Tokenization Drift)**
   The rollout engine (`vLLM`) returns exact generated token IDs and per-token rollout logprobs (`log pi_vllm`). The trainer never re-tokenizes assistant text strings back into IDs, eliminating BPE boundary mismatches.

2. **Strict Multi-Turn Observation Masking**
   Every trajectory stores aligned `token_ids` and binary `loss_mask`:
   * `1` on assistant-generated tokens (reasoning + tool calls)
   * `0` on system prompt, user prompt, and tool/sandbox observation tokens
   Unit tests verify strictly zero gradient flows through masked observation tokens.

3. **Zero Silent Weight Staleness**
   Every trainer → rollout weight sync (dynamic LoRA adapter reload or full in-place weight reload) computes and asserts a matching **SHA-256 parameter digest** before the next rollout batch begins.

4. **Train–Inference Mismatch (TIM) Telemetry + MIS/CIS Ablation**
   Every update logs:
   * `delta_t = log pi_train(a_t) - log pi_vllm(a_t)` (mean, p99, max abs)
   * Importance ratio tail (`r_t > 2.0` fraction) and Effective Sample Size (ESS)
   * Clean config switch between **Vanilla GRPO** and **Masked/Clipped Importance Sampling (MIS/CIS)** to measure divergence vs. stability empirically.

5. **Anti-Reward-Hacking Verification (With Naive Ablation Toggle)**
   Hardened grading runs isolated Python (`python3 -I -S`) against **withheld hidden tests + pristine copies of visible tests** (so editing or deleting visible tests cannot pass) and records a tamper flag in `VerifierReceipt`. A `naive=True` toggle runs the workspace's own visible tests directly to reproduce and measure live reward hacking under RL pressure.

## Milestones

* **M1 (Days 1–4):** Parametric multi-turn task suite (200 train / 50 held-out), hardened Modal Sandbox pool, `--self-check`, baseline `pass@1`/`pass@4` with 95% CIs.
* **M2 (Days 5–12):** End-to-end PyTorch GRPO + vLLM rollout + SHA-256 weight sync on 1–2x H100/A100 (Modal); first real reward & held-out eval lift curve.
* **M3 (Weeks 3–5):** TIM/MIS stability ablation + live reward-hacking study + wall-clock utilization breakdown (`rollout` / `sandbox` / `learner` / `sync`).
* **M4 (Weeks 6–8):** Plug in `shapens/tri` copy-on-write/gVisor backend and benchmark tail latency & GPU utilization vs. cold containers; publish the technical memo.
