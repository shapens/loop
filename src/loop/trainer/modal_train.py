from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from loop.envs.generator import generate_split
from loop.envs.sandbox import LocalSubprocessSandbox
from loop.eval.baseline import summarize_eval
from loop.rollout.protocol import STOP_STRINGS
from loop.rollout.runner import BatchedGeneration, run_episodes_batched
from loop.trainer.engine import GRPOConfig, grpo_train_step


def run_colocated_grpo(
    cfg: GRPOConfig,
    *,
    output_dir: Path,
    hf_cache_dir: Path,
    vllm_gpu_mem_util: float = 0.40,
) -> dict[str, Any]:
    """Co-located single-GPU (H100/A100-80GB) vLLM rollout + PyTorch PEFT LoRA GRPO loop.

    Uses vLLM's dynamic LoRARequest hot-reload so each policy update syncs in ~200ms with
    SHA-256 export verification, and computes k3 KL reference logprobs with zero extra VRAM
    via `peft_model.disable_adapter()`.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    output_dir.mkdir(parents=True, exist_ok=True)
    adapters_root = output_dir / "adapters"
    adapters_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    # 1. Initialize vLLM rollout engine with dynamic LoRA support.
    llm = LLM(
        model=cfg.model_name,
        download_dir=str(hf_cache_dir),
        gpu_memory_utilization=vllm_gpu_mem_util,
        max_model_len=cfg.max_seq_len,
        enable_lora=True,
        max_lora_rank=cfg.lora_rank,
    )

    # 2. Initialize PyTorch PEFT LoRA learner in remaining HBM.
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=str(hf_cache_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.bfloat16,
        cache_dir=str(hf_cache_dir),
    )
    base_model.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg).to("cuda")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    train_pool = generate_split("train", max(200, cfg.steps * cfg.tasks_per_step))
    eval_pool = generate_split("eval", cfg.eval_tasks)
    sandbox = LocalSubprocessSandbox()

    def tokenize(text: str) -> list[int]:
        return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]

    def make_generator(*, temperature: float, lora_req: Any | None) -> Any:
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=cfg.max_new_tokens,
            logprobs=1,
            stop=list(STOP_STRINGS),
            include_stop_str_in_output=True,
        )

        def generate_batch(prompts: list[list[int]]) -> list[BatchedGeneration]:
            req_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]
            outputs = llm.generate(req_prompts, sampling, lora_request=lora_req, use_tqdm=False)
            results: list[BatchedGeneration] = []
            for out in outputs:
                seq = out.outputs[0]
                tok_ids = [int(t) for t in seq.token_ids]
                lps: list[float] = []
                lp_list = seq.logprobs or []
                for tid, step_dict in zip(tok_ids, lp_list, strict=True):
                    entry = step_dict.get(tid)
                    lps.append(float(entry.logprob) if entry is not None else 0.0)
                results.append(BatchedGeneration(text=seq.text, token_ids=tok_ids, logprobs=lps))
            return results

        return generate_batch

    def run_eval(step_idx: int, lora_req: Any | None) -> dict[str, Any]:
        # Always evaluate held-out tasks with the hardened ground-truth grader so naive-grader
        # reward-hacking runs reveal the gap between inflated train reward and true accuracy.
        eval_trajs = run_episodes_batched(
            eval_pool,
            samples_per_task=cfg.eval_samples,
            generate_batch=make_generator(temperature=0.7, lora_req=lora_req),
            tokenize=tokenize,
            sandbox=sandbox,
            max_turns=cfg.max_turns,
            policy_version=step_idx,
            grader_mode="hardened",
        )
        rep = summarize_eval(
            model=f"{cfg.model_name}@step{step_idx}",
            split="eval",
            grader_mode="hardened",
            tasks=eval_pool,
            trajectories=eval_trajs,
        )
        return asdict(rep)

    t_start = time.perf_counter()
    eval_history: list[dict[str, Any]] = []
    step_history: list[dict[str, Any]] = []

    lora_history: list[Any | None] = [None]
    eval_0 = run_eval(0, None)
    eval_history.append({"step": 0, "report": eval_0})

    for step in range(1, cfg.steps + 1):
        start_i = ((step - 1) * cfg.tasks_per_step) % len(train_pool)
        batch_tasks = [
            train_pool[(start_i + j) % len(train_pool)] for j in range(cfg.tasks_per_step)
        ]
        rollout_policy_idx = max(0, (step - 1) - cfg.stale_steps)
        rollout_lora = lora_history[rollout_policy_idx]

        t_rollout_wall = time.perf_counter()
        trajs = run_episodes_batched(
            batch_tasks,
            samples_per_task=cfg.group_size,
            generate_batch=make_generator(temperature=cfg.temperature, lora_req=rollout_lora),
            tokenize=tokenize,
            sandbox=sandbox,
            max_turns=cfg.max_turns,
            policy_version=rollout_policy_idx,
            grader_mode=cfg.grader_mode,
        )
        total_rollout_wall_ms = (time.perf_counter() - t_rollout_wall) * 1000.0
        gen_wall_ms = sum(tr.rollout_ms for tr in trajs)
        sb_wall_ms = max(0.0, total_rollout_wall_ms - gen_wall_ms)

        adapter_dir = adapters_root / f"step_{step:04d}"
        metrics = grpo_train_step(
            model,
            optimizer,
            trajs,
            pad_token_id=pad_token_id,
            cfg=cfg,
            step=step,
            device="cuda",
            adapter_export_dir=adapter_dir,
            rollout_wall_ms=gen_wall_ms,
            sandbox_wall_ms=sb_wall_ms,
        )
        latest_lora = LoRARequest(f"step_{step:04d}", step, str(adapter_dir))
        lora_history.append(latest_lora)

        rec = {
            **asdict(metrics),
            "rollout_policy_version": rollout_policy_idx,
            "timing_fractions": metrics.timing.fractions(),
        }
        step_history.append(rec)
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"[step {step}/{cfg.steps} lag={step - 1 - rollout_policy_idx}] "
            f"reward={metrics.mean_reward:.3f} "
            f"pass={metrics.pass_rate:.3f} tamper={metrics.tamper_rate:.3f} "
            f"loss={metrics.loss:.4f} gnorm={metrics.grad_norm:.3f} "
            f"tim_mean={metrics.tim.mean_abs_logprob_diff:.4f} "
            f"tim_max={metrics.tim.max_abs_logprob_diff:.4f} "
            f"hi_r={metrics.tim.high_ratio_fraction:.4f} "
            f"mis_masked={metrics.tim.masked_token_fraction:.4f} "
            f"sync_ms={metrics.timing.weight_sync_ms:.0f}",
            flush=True,
        )

        if step % cfg.eval_every == 0 or step == cfg.steps:
            ev = run_eval(step, latest_lora)
            eval_history.append({"step": step, "report": ev})
            p1 = ev["pass_at_k"][1]["mean"]
            p4 = ev["pass_at_k"].get(4, {}).get("mean", 0.0)
            print(f"[eval step {step}] pass@1={p1:.3f} pass@4={p4:.3f}", flush=True)

    summary = {
        "config": asdict(cfg),
        "wall_seconds": time.perf_counter() - t_start,
        "steps": step_history,
        "evals": eval_history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
