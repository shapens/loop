# Engineering & Empirical Memo: Multi-Turn Coding RLVR on Burst H100 Compute

**Author:** Solo build (`shapens/loop`)  
**Stack:** Pure PyTorch 2.10 + Custom Triton 3.6 kernel + vLLM 0.19.1 + PEFT LoRA + Modal H100 / `shapens/tri`  
**Total GPU Spend Across All Runs & Ablations:** ~$8 on Modal burst H100/L40S

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

## 3. Six Empirical Studies

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

### Study C — Async Off-Policy Staleness (`lag = 3` vs `lag = 8, kl = 0`): Vanilla GRPO vs. MIS
* **Setup:** vLLM samples rollouts from a LoRA checkpoint **3 steps** (`lr=8e-5, kl=0.01`) or **8 steps** (`lr=1.5e-4, kl=0.0`) behind the learner across all 5 families (50 held-out tasks × 4).
* **Result:**
  * At **`lag = 3`**: max $|\log \pi_{\text{train}} - \log \pi_{\text{vLLM}}|$ rose **~5×** above on-policy (`1.49 – 1.66` vs. `0.23 – 0.35`), and MIS (`r <= 1.5`) actively masked up to **`0.15%` of tail tokens**. With LoRA `r=16`, PPO clipping (`0.2`), and k3 KL (`0.01`), both MIS (`5.5% → 36.5%` `pass@1`, max held-out tamper `2`) and Vanilla (`6.5% → 37.0%` `pass@1`, tamper spike `8`) converged cleanly—while `sandbox_ms` ballooned to **`32%–67%` of step wall-clock time**.
  * At **`lag = 8, kl = 0.0, lr = 1.5e-4`**: max $|\Delta \log p|$ spiked to **`4.67 – 5.77` (`106×–322×` importance ratio)**. MIS accelerated early learning **`3×`** (`18.0%` vs `6.0%` at step 5) and kept step-20 tamper at `0.0%` (vs `7.5%` for Vanilla), while both plateaued at **`17.0%–17.5%`** because 8-step lag over 20 total steps leaves steps 1–9 rolling out from the frozen base model.

### Study D — Systems Wins: Custom Triton Kernel & `shapens/tri` Scaling (`c = 1 → 64`)
1. **Custom Triton Fused Online-LSE LogProb Kernel (`src/loop/trainer/triton_logprob.py`):**
   * Benchmarked on **NVIDIA H100 80GB HBM3** at Qwen's real vocab `[2, 2048, 152064]` `bf16`:
     * **Forward + Backward Latency:** `14.85 ms` (PyTorch) → **`5.57 ms` (Triton)** = **`2.67× faster`**
     * **Peak HBM Allocated:** `9.279 GB` → **`5.801 GB`** = **`3.478 GB` VRAM saved (-37.5%)**
     * **Numerical Parity:** `fwd_max_abs_err = 9.54e-7`, `bwd_max_abs_err = 4.66e-10`
   * Used live for both policy and k3 KL reference passes in all 7B H100 runs (extending Triton to `ref_lp` eliminated a real `4.64 GiB` fp32 softmax OOM on long 7B rollouts).
2. **`shapens/tri` High-Concurrency Substrate Upgrade (`src/loop/bench_sandboxes.py`, `all_correct=True`, 64 gradings/level):**
   * Our initial benchmark exposed a `c16` plateau (`15.89/s`, `835 ms` p50) caused by 12 HTTP round-trips per grading, SQLite `DELETE` journal mode, exclusive `admin_lock` + `PRAGMA wal_checkpoint(TRUNCATE)` per request, and a multi-worker `WAL ↔ DELETE` startup race in `tri-server`.
   * Shipped atomic `tri files write-batch`, SQLite `WAL` + `IMMEDIATE` + lock-free fast-open, and HTTP/1.1 keep-alive in [`shapens/tri@485a523`](https://github.com/shapens/tri/commit/485a523), plus single-HTTP `/v0/batch` execution in `TriSandboxBackend`:
     * `modal_sandbox_ephemeral` (remote cloud gVisor container, includes WAN RTT): `1,031.6 ms` p50 / `2,361.4 ms` p95 (`7.48/s @c8`)
     * `shapens_tri_http` **v1** (local loopback, 12-req DELETE): `148.0 ms` p50 / `288.3 ms` p95 (`15.89/s @c16`, `835.0 ms` p50 @c16)
     * `shapens_tri_http_v2_batch` (local loopback, `tri serve` WAL+batch): `133.3 ms` p50 (`27.48/s @c16`, `29.01/s @c32`, `26.17/s @c64`)
     * **`shapens_tri_native_seatbelt`** (local Go `tri-server` 16–32 workers + WAL/`IMMEDIATE` + `/v0/batch` + **`--run-sandbox required` macOS Seatbelt OS isolation**): **`66.5 ms` p50 @c1** (`49.6 ms` unsandboxed), **`47.75/s @c8`**, **`54.52/s @c16`** (**`3.43×` v1 throughput**, **`3.98×` lower `c16` p50 at `209.8 ms`**), **`45.95/s @c32`**, **`39.33/s @c64`**, `0` worker crashes.

### Study E & F — Breaking the Sparse Multi-File Wall on `7B`: Dense Sub-Test Shaping vs. Binary `G=16` Curriculum
* **Question:** Why did `keyset_paging` (`2.5%`) and `spec_invoice_calc` (`2.5%`) stall in Study A, and what actually unlocks them?
* **Study E — Dense Sub-Test Shaping (`grpo_7b_5fam_subtest`, strict 1-variable ablation vs Study A, tamper-gated `0.5 × passed/total` + `0.5` full bonus):**
  * **Negative result (`-9.5 pp` regression!):** Held-out binary `pass@1` rose only `30.0% → 42.5%` (vs `52.0%` binary); `async_keyed_queue` fell `-30 pp` (`27.5%` vs `57.5%`), `ledger_invariants` fell `-15 pp` (`82.5%` vs `97.5%`), and `keyset_paging` stayed **`0.0%`**.
  * **Mechanism:** GRPO group normalization `(r - μ) / σ` amplifies small partial-credit gaps (`0.25` vs `0.125`) into full unit-variance positive advantages, reinforcing partially buggy trajectories and diluting clean full-fix gradients.
* **Study F — Pure Binary RLVR + Larger Group Exploration (`G=16`) + Hard-Family Curriculum (`grpo_7b_hard_curriculum_g16`, 15 steps, `635s`, `lr=3e-5`, training only on `keyset_paging, spec_invoice_calc`, evaluated on all 5 families):**
  * **Breakthrough:** `keyset_paging` exploded from **`0.0% → 70.0%` held-out `pass@1` (+70.0 pp)**, lifting overall 5-family **`pass@4` to a new high of `68.0%`** `[56.0%, 80.0%]` (`pass@1` `32.0% → 47.0%`) with **zero catastrophic forgetting** on the 3 untrained families (`rate` `85% → 90%`, `ledger` `67.5% → 65%`, `async` `7.5% → 7.5%`).
  * **Remaining boundary:** `spec_invoice_calc` remained at `2.5%` even at `G=16`, cleanly marking the `pass@G ≈ 0` cold-start threshold where pure zero-shot RLVR needs SFT warm-start or `G ≥ 64` exploration.

---

## 4. Scope & Honest Caveats

1. **Micro-lab tasks, not SWE-bench:** All 5 families are synthetic 1–3 file stdlib Python repos (~50–100 LoC, ~40ms `unittest` runs). `loop` is designed as a contamination-free micro-lab to isolate RLVR mechanisms cheaply—not a claim of SWE-bench Verified SOTA.
2. **Sandbox comparison geometry:** `shapens/tri` benchmarks ran on local macOS loopback (`127.0.0.1`) with macOS Seatbelt OS isolation (`--run-sandbox required`), whereas Modal Sandboxes ran over the public internet with cold cloud gVisor container boot. The benchmark isolates the architectural delta between persistent CoW workspaces and ephemeral cloud containers, not same-network/same-hypervisor parity. On Modal H100 training runs, grading used `LocalSubprocessSandbox` inside the GPU container.
3. **Single-seed LoRA `r=16` runs & multi-variable Study F:** Each RL run is a single seed over 15–20 steps with LoRA `r=16` (which naturally dampens some off-policy instability relative to full-weight/MoE RL). Study E vs. Study A is a strict 1-variable ablation (`reward_mode` only); Study F intentionally combined `G=16`, hard-family curriculum, and `lr=3e-5` to test breaking the sparse wall.
