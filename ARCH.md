# Architecture and Invariants

`loop` is deliberately small and inspectable (no `TRL` or `verl` trainer black box). Five invariants are enforced by automated tests:

1. **Exact Token-ID Handoff (No Re-Tokenization Drift)**
   The rollout engine (`vLLM`) returns exact generated token IDs and per-token rollout logprobs (`log pi_vllm`). The trainer never re-tokenizes assistant text strings back into IDs, eliminating BPE boundary mismatches.

2. **Strict Multi-Turn Observation Masking**
   Every trajectory stores aligned `token_ids` and binary `loss_mask`: `1` on assistant-generated tokens (reasoning and tool calls) and `0` on system prompt, user prompt, and tool/sandbox observation tokens. Unit tests verify strictly zero gradient flows through masked observation tokens.

3. **SHA-256 Weight-Sync Verification**
   Every trainer-to-rollout weight sync computes and asserts a matching raw-byte **SHA-256 parameter digest** before vLLM loads the updated adapter for the next rollout batch.

4. **Train-Inference Mismatch (TIM) Telemetry and MIS Switch**
   Every update logs `delta_t = log pi_train(a_t) - log pi_vllm(a_t)` (mean, p99, max abs), importance-ratio tail (`r_t > 2.0` fraction), and effective sample size (ESS), with a clean config switch between Vanilla GRPO and Masked Importance Sampling (MIS).

5. **Anti-Reward-Hacking Verification (With Naive Toggle)**
   Hardened grading runs isolated Python (`python3 -I -S`) against withheld hidden tests plus pristine copies of visible tests (so editing or deleting visible tests cannot pass) and records a tamper flag in `VerifierReceipt`. A `naive` mode runs the workspace's own visible tests directly to measure live reward hacking under RL pressure.
