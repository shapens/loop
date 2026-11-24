# Empirical Study: Hardened vs. Naive Verifiers in Multi-Turn Coding RLVR

Controlled 20-step GRPO ablation on **`Qwen/Qwen2.5-Coder-3B-Instruct`** (co-located vLLM 0.19.1 + PyTorch 2.10 PEFT LoRA on 1× Modal H100, `group_size=8`, `tasks_per_step=4`, `kl_coef=0.01`, `use_mis=True`).

Both runs share the **exact same model, prompt, hyperparameters, and training tasks**. The only variable changed during training is the verifier:
* **Hardened (`--grader-mode hardened`):** Isolated `python3 -I -S` execution against withheld hidden tests + pristine copies of visible tests + tamper rejection (`reward=0.0` if `tests/` is modified).
* **Naive (`--grader-mode naive`):** Runs the workspace's own `tests/` directly without hidden tests or tamper rejection.

**Both runs are evaluated on the exact same 30 held-out tasks (4 samples/task = 120 episodes) using the hardened ground-truth grader.**

---

## Headline Results

| Metric | Hardened Training (464s, 1×H100) | Naive Training (407s, 1×H100) | Delta |
| :--- | :---: | :---: | :---: |
| **Step 0 Held-Out `pass@1`** (95% CI) | `6.7%` `[2.5%, 11.7%]` | `8.3%` `[4.2%, 13.3%]` | — |
| **Step 20 Held-Out `pass@1`** (95% CI) | **`38.3%`** `[24.2%, 53.3%]` | `19.2%` `[9.2%, 29.2%]` | **+19.1 pp (2.0×)** |
| **Step 20 Held-Out `pass@4`** (95% CI) | **`56.7%`** `[40.0%, 73.3%]` | `33.3%` `[16.7%, 50.0%]` | **+23.4 pp** |
| **Peak Train Batch Reward** | `53.1%` (step 18) | **`100.0%`** (step 13) | Goodhart gap: **68.3 pp** |
| **Held-Out Mean Turns (s0 → s20)** | `2.82 → 4.68` (+1.86 turns) | `2.79 → 2.88` (+0.09 turns) | Emergent multi-turn depth |

### Held-Out `pass@1` Progression by Task Family

| Step | Hardened `rate_limiter` | Hardened `ledger` | Hardened `paging` | Naive `rate_limiter` | Naive `ledger` | Naive `paging` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 20.0% | 0.0% | 0.0% | 20.0% | 5.0% | 0.0% |
| 5 | 52.5% | 2.5% | 0.0% | 27.5% | 0.0% | 0.0% |
| 10 | 92.5% | 7.5% | 0.0% | 55.0% | 0.0% | 0.0% |
| 15 | 95.0% | 15.0% | 0.0% | 52.5% | 0.0% | 0.0% |
| 20 | **92.5%** | **22.5%** | 0.0% | 57.5% | **0.0%** | 0.0% |

---

## Key Empirical Findings

1. **Specification Gaming & Proxy Saturation (Naive Grader):**
   Under the naive verifier, training reward surged to **87.5%–100.0%** by steps 11–20, yet true held-out `pass@1` stalled at **19.2%** and `ledger_invariants` collapsed from **5.0% → 0.0%**. Because the weak visible tests passed easily, the policy learned to emit short trajectories (`~2.8 turns`, `~5k` active tokens/step) and call `<finish/>` early without fixing deeper invariants—plus attempted visible-test tampering at steps 7 and 9 (`tamper_rate=3.1%`).
2. **Emergent Multi-Turn Persistence (Hardened Grader):**
   Under hardened verification, held-out `pass@1` rose **5.75×** (`6.7% → 38.3%`, non-overlapping 95% CIs) and `pass@4` reached **56.7%**. The policy spontaneously increased its interaction budget (`2.82 → 4.68` mean turns; `7.3k → 10–11k` active tokens/step) to read, edit multiple files, and verify invariants.
3. **Train–Inference Mismatch (TIM) & Weight Sync Integrity:**
   Exact vLLM token-ID handoff + SHA-256 verified LoRA hot-reload (`~1.0s/step`) held mean $|\log \pi_{\text{train}} - \log \pi_{\text{vLLM}}|$ at **`0.0021 – 0.0039`** across 10k active tokens/step, while still surfacing real kernel-level tail spikes (`max` up to `0.614` on naive step 12 / `0.358` on hardened step 1).
4. **Wall-Clock Step Breakdown (3B on 1×H100, step 20):**
   `learner: 51.4%` (`8.97s`) · `vLLM rollout: 36.3%` (`6.33s`) · `sandbox: 6.4%` (`1.12s`) · `LoRA weight sync + SHA-256: 5.9%` (`1.03s`).
5. **`shapens/tri` Integration:**
   `TriSandboxBackend` collapses workspace creation, atomic `files write-batch`, sandboxed `tri run`, stdout/stderr streaming, and workspace discard into a single `POST /v0/batch` call. See [docs/memo.md](memo.md) (Study D) for the full `c = 1 → 64` macOS Seatbelt throughput and latency benchmarks.

---

## Reproducing Both Runs

```bash
# Hardened run
modal run src/loop/modal_app.py::train_grpo \
  --run-name grpo_qwen3b_hardened \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --steps 20 --tasks-per-step 4 --group-size 8 --eval-every 5 --grader-mode hardened

# Naive reward-hacking ablation
modal run src/loop/modal_app.py::train_grpo \
  --run-name grpo_qwen3b_naive \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --steps 20 --tasks-per-step 4 --group-size 8 --eval-every 5 --grader-mode naive
```
