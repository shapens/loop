# Engineering & Empirical Memo: Multi-Turn Coding RLVR on Burst H100 Compute

**Author:** Solo build (`shapens/loop`)  
**Stack:** Pure PyTorch 2.10 + Custom Triton 3.6 kernel + vLLM 0.19.1 + PEFT LoRA + Modal H100 / `shapens/tri`  
**Total GPU Spend Across All Runs & Ablations:** ~$5 on Modal burst H100/L40S

---

## 1. Hypothesis & System Invariants

Multi-turn coding RL fails in practice for systems and verification reasons long before it fails for algorithmic reasons. `loop` isolates and measures four specific failure modes without hiding behind a black-box trainer (`TRL`/`verl`):

1. **Exact Token-ID Handoff:** vLLM returns sampled `token_ids` and per-token `logprobs=1`; the PyTorch learner never re-tokenizes assistant strings (eliminating BPE boundary drift).
2. **Strict Multi-Turn Observation Masking:** Prompt and tool-output tokens carry `loss_mask = 0` and are zeroed before exponentiation—verified by unit tests to produce strictly zero gradient.
3. **SHA-256 Weight-Sync Verification:** Every LoRA checkpoint export computes a deterministic SHA-256 digest before vLLM loads the adapter via `LoRARequest`, making silent weight staleness impossible.
4. **Anti-Reward-Hacking Verification:** Hardened grading runs isolated Python (`python3 -I -S`) over withheld hidden tests plus pristine copies of visible tests (`_loop_hidden/`), rejecting any edits to `tests/` with `reward = 0.0`.

---

## 2. What Broke First (And What Fixed It)

Our very first L40S baseline eval (`Qwen2.5-Coder-1.5B-Instruct`, 120 episodes, 16s wall clock) scored **`0.0% pass@1`** (`mean_turns = 2.43`). Inspecting raw transcripts immediately exposed two classic post-training harness bugs:

1. **Template Mismatch:** Wrapping prompts in custom `<system>/<user>` tags instead of Qwen's native ChatML (`<|im_start|>`/`<|im_end|>`) degraded instruction following on small models.
2. **JSON-Escaping Failures on Multi-Line Code:** Asking a 1.5B model to emit full Python files inside JSON strings caused frequent parse failures that prematurely ended episodes.

Switching to native ChatML + escape-free `<write path="...">...</write>`, `<run_tests/>`, `<finish/>` tags lifted baseline `pass@4` immediately from `0.0% → 20.0%` (1.5B) and `30.0%` (3B).

---

## 3. Four Empirical Studies

### Study A — Headline 7B Multi-Turn RLVR Across 5 Coding Families (Live Triton Kernel)
* **Setup:** `Qwen/Qwen2.5-Coder-7B-Instruct`, 20 GRPO steps on 1×H100 (`1,045.6s`), `tasks_per_step=5` (one per family), `group_size=8`, `kl_coef=0.01`, `use_mis=True`, held-out eval on 50 contamination-free parametric tasks × 4 samples (200 episodes).
* **Result:**
  * Held-out **`pass@1`:** `32.5%` `[22.5%, 43.5%]` → **`52.0%`** `[39.5%, 65.0%]` (**+19.5 pp**)
  * Held-out **`pass@4`:** `50.0%` → **`64.0%`** (**+14.0 pp**)
  * **`async_keyed_queue`** (real `asyncio` lock-handover & cancellation race): **`7.5% → 57.5%` (+50.0 pp)**
  * **`ledger_invariants`** (2-file cache coherence + rounding): **`70.0% → 97.5%` (+27.5 pp)**
  * **`rate_limiter_window`**: **`80.0% → 100.0%` (+20.0 pp)**
  * **Honest negative/sparse result:** `keyset_paging` (`5.0% → 2.5%`) and `spec_invoice_calc` (`0.0% → 2.5%`) remained sparse under binary outcome-only rewards across 20 steps, showing where process-level or sub-test reward shaping becomes necessary for 3-file coordinated edits.

### Study B — Controlled Reward-Hacking / Goodhart Ablation (Hardened vs. Naive Verifier)
* **Setup:** `Qwen/Qwen2.5-Coder-3B-Instruct`, 20 GRPO steps, identical prompt/hyperparameters; training uses either `hardened` or `naive` (workspace `tests/` only), while held-out evaluation **always** uses the `hardened` ground-truth verifier.
* **Result:**
  * **Hardened (`464s`):** Held-out `pass@1` jumped **5.75×** (`6.7%` `[2.5%, 11.7%]` → **`38.3%`** `[24.2%, 53.3%]`, non-overlapping 95% CIs); `pass@4` reached **`56.7%`**; emergent interaction depth rose `2.82 → 4.68` mean turns (`ledger_invariants`: `0.0% → 22.5%`).
  * **Naive (`407s`):** Training reward rocketed to **`100.0%`** (step 13) / **`87.5%`** (step 20), yet true held-out `pass@1` stalled at **`19.2%`** (**68.3 pp Goodhart gap**), `ledger_invariants` collapsed **`5.0% → 0.0%`**, and mean turns stayed flat at `2.88` as the policy learned to stop as soon as weak visible tests passed (plus attempted direct visible-test tampering at steps 7 & 9).

### Study C — Async Off-Policy Staleness (`lag = 3 steps`, `lr = 8e-5`): Vanilla GRPO vs. MIS
* **Setup:** vLLM samples rollouts from a LoRA checkpoint **3 optimizer steps behind** the learner (`stale_steps=3`) across all 5 families (50 held-out tasks × 4).
* **Result:**
  * Under 3-step lag, max $|\log \pi_{\text{train}} - \log \pi_{\text{vLLM}}|$ rose **~5×** above on-policy (`1.49 – 1.66` vs. `0.23 – 0.35`), and MIS (`r <= 1.5`) actively masked up to **`0.15%` of tail tokens**.
  * **Honest nuanced takeaway:** With LoRA `r=16`, PPO clipping (`0.2`), and k3 KL (`0.01`), both MIS (`5.5% → 36.5%` `pass@1`, `42.0%` `pass@4`, max held-out tamper `2`) and Vanilla (`6.5% → 37.0%` `pass@1`, `40.0%` `pass@4`, tamper spiked to `8` at step 10) converged reliably—while `sandbox_ms` ballooned to **`32%–67%` of total H100 step time** on 40-episode batches containing `asyncio` tests, confirming that sandbox tail latency (not LoRA staleness at lag=3) was the dominant bottleneck at this scale.

### Study D — Systems Wins: Custom Triton Kernel & `shapens/tri` Sandbox Throughput
1. **Custom Triton Fused Online-LSE LogProb Kernel (`src/loop/trainer/triton_logprob.py`):**
   * Benchmarked on **NVIDIA H100 80GB HBM3** at Qwen's real vocab `[2, 2048, 152064]` `bf16`:
     * **Forward + Backward Latency:** `14.85 ms` (PyTorch) → **`5.57 ms` (Triton)** = **`2.67× faster`**
     * **Peak HBM Allocated:** `9.279 GB` → **`5.801 GB`** = **`3.478 GB` VRAM saved (-37.5%)**
     * **Numerical Parity:** `fwd_max_abs_err = 9.54e-7`, `bwd_max_abs_err = 4.66e-10`
   * Ran live inside all 3B async and 7B H100 GRPO runs above.
2. **3-Way Sandbox Concurrency Benchmark (`src/loop/bench_sandboxes.py`, `all_correct=True`):**
   * `local_subprocess` (no isolation): `34.2 ms` p50 / `170.6 ms` p95 (`73.4/s @c16`)
   * **`shapens/tri` HTTP CoW (`TriSandboxBackend`):** **`148.0 ms` p50 / `288.3 ms` p95** (**7.0× faster p50, 8.2× lower p95** than Modal Sandboxes; **`16.95/s @c8` = 2.27×** throughput). Contention plateaus at `c16` (`15.89/s`) on single-node SQLite, pointing directly to `tri`'s PostgreSQL + batched `/v0/batch` write path as the next scaling step.
   * `modal_sandbox_ephemeral`: `1,031.6 ms` p50 / `2,361.4 ms` p95 (`7.48/s @c8`).
