from __future__ import annotations

import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from loop.envs.generator import generate_split
from loop.envs.grader import grade_workspace
from loop.envs.sandbox import LocalSubprocessSandbox
from loop.eval.baseline import summarize_eval
from loop.rollout.protocol import STOP_STRINGS
from loop.rollout.runner import BatchedGeneration, run_episodes_batched
from loop.trainer.engine import GRPOConfig
from loop.trainer.modal_train import run_colocated_grpo
from loop.types import GraderMode, Split, TaskSpec, VerifierReceipt

_modal: Any
try:
    _modal = importlib.import_module("modal")
except ImportError:
    _modal = None


def grade_remote(
    task: TaskSpec, workspace_files: dict[str, str], mode: GraderMode = "hardened"
) -> VerifierReceipt:
    """Grade via Modal when deployed, otherwise run isolated local grader."""
    if _modal is None:
        return grade_workspace(task, workspace_files, mode=mode)
    raw = grade_task_modal.remote(asdict(task), workspace_files, mode)
    return VerifierReceipt(**raw)


if _modal is not None:
    app = _modal.App("shapens-loop")
    cpu_image = (
        _modal.Image.debian_slim(python_version="3.11")
        .pip_install("pydantic>=2.7", "httpx>=0.27")
        .add_local_python_source("loop")
    )
    gpu_image = (
        _modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.4",
            "transformers>=4.46",
            "peft>=0.13",
            "vllm>=0.6.4",
            "pydantic>=2.7",
            "httpx>=0.27",
        )
        .add_local_python_source("loop")
    )
    weights_vol = _modal.Volume.from_name("shapens-loop-weights", create_if_missing=True)

    @app.function(image=cpu_image, timeout=60, retries=0)
    def grade_task_modal(
        task_dict: dict[str, Any], workspace_files: dict[str, str], mode: GraderMode = "hardened"
    ) -> dict[str, Any]:
        task = TaskSpec(
            **{
                **task_dict,
                "allowed_edit_globs": tuple(task_dict["allowed_edit_globs"]),
            }
        )
        receipt = grade_workspace(task, workspace_files, mode=mode)
        return asdict(receipt)

    @app.function(
        image=gpu_image,
        gpu="L4",
        timeout=300,
        volumes={"/weights": weights_vol},
    )
    def gpu_smoke_test() -> dict[str, Any]:
        """Cheap (~$0.05) sanity check verifying CUDA, PyTorch GRPO, and weight digest on Modal."""
        import torch

        from loop.trainer.grpo import compute_group_advantages, grpo_loss
        from loop.trainer.weight_sync import state_dict_sha256

        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy_lp = torch.randn(4, 16, device=device, requires_grad=True)
        rollout_lp = policy_lp.detach() + 0.01 * torch.randn(4, 16, device=device)
        mask = torch.ones(4, 16, device=device)
        mask[:, :4] = 0.0
        adv = compute_group_advantages(torch.tensor([1.0, 0.0, 1.0, 0.0], device=device), 4)
        loss, tim = grpo_loss(policy_lp, rollout_lp, adv, mask)
        loss.backward()  # type: ignore[no-untyped-call]
        digest = state_dict_sha256({"probe": policy_lp.detach()})
        return {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "loss": float(loss.item()),
            "tim": asdict(tim),
            "digest_prefix": digest[:12],
        }

    @app.function(
        image=gpu_image,
        gpu="L40S",
        timeout=1800,
        volumes={"/weights": weights_vol},
    )
    def run_baseline_eval_modal(
        model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        split: Split = "eval",
        num_tasks: int = 30,
        samples_per_task: int = 4,
        max_turns: int = 6,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        grader_mode: GraderMode = "hardened",
    ) -> dict[str, Any]:
        """Run batched multi-turn vLLM baseline evaluation on Modal and save report to Volume."""
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        llm = LLM(
            model=model,
            download_dir="/weights/hf-cache",
            gpu_memory_utilization=0.9,
            max_model_len=8192,
        )
        tokenizer = llm.get_tokenizer()
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=1,
            stop=list(STOP_STRINGS),
            include_stop_str_in_output=True,
        )

        def tokenize(text: str) -> list[int]:
            return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]

        def generate_batch(prompts: list[list[int]]) -> list[BatchedGeneration]:
            req_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]
            outputs = llm.generate(req_prompts, sampling, use_tqdm=False)
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

        tasks = generate_split(split, num_tasks)
        sandbox = LocalSubprocessSandbox()
        trajs = run_episodes_batched(
            tasks,
            samples_per_task=samples_per_task,
            generate_batch=generate_batch,
            tokenize=tokenize,
            sandbox=sandbox,
            max_turns=max_turns,
            grader_mode=grader_mode,
        )
        report = summarize_eval(
            model=model,
            split=split,
            grader_mode=grader_mode,
            tasks=tasks,
            trajectories=trajs,
        )
        payload = asdict(report)
        payload["sample_transcripts"] = [
            {
                "prompt_id": tr.prompt_id,
                "passed": tr.receipt.passed,
                "tampered": tr.receipt.visible_tests_tampered,
                "turns": tr.turns,
                "stderr_tail": tr.receipt.stderr[-400:],
                "assistant_text": tokenizer.decode(
                    [tid for tid, m in zip(tr.token_ids, tr.loss_mask, strict=True) if m == 1]
                )[:1200],
            }
            for tr in trajs[:3]
        ]

        out_dir = Path("/weights/evals")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model.replace("/", "__")
        out_file = out_dir / f"baseline_{safe_model}_{split}_{grader_mode}.json"
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        weights_vol.commit()
        return payload

    @app.function(
        image=gpu_image,
        gpu="H100",
        timeout=300,
    )
    def bench_triton_kernel_modal(
        batch_size: int = 2,
        seq_len: int = 2048,
        vocab_size: int = 152_064,
    ) -> dict[str, Any]:
        """Benchmark fused Triton online-LSE logprobs vs eager PyTorch on H100."""
        from loop.trainer.triton_logprob import benchmark_triton_vs_pytorch

        return benchmark_triton_vs_pytorch(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
        )

    @app.function(
        image=gpu_image,
        gpu="H100",
        timeout=3600,
        volumes={"/weights": weights_vol},
    )
    def run_grpo_train_modal(
        run_name: str = "grpo_qwen15b_mis",
        model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        steps: int = 12,
        tasks_per_step: int = 4,
        group_size: int = 8,
        learning_rate: float = 2e-5,
        use_mis: bool = True,
        mis_max_ratio: float = 2.0,
        stale_steps: int = 0,
        micro_batch_size: int = 2,
        grader_mode: GraderMode = "hardened",
        eval_every: int = 4,
        eval_tasks: int = 30,
        eval_samples: int = 4,
    ) -> dict[str, Any]:
        """Run co-located H100 vLLM + PyTorch PEFT LoRA GRPO training and commit artifacts."""
        cfg = GRPOConfig(
            model_name=model,
            steps=steps,
            tasks_per_step=tasks_per_step,
            group_size=group_size,
            learning_rate=learning_rate,
            use_mis=use_mis,
            mis_max_ratio=mis_max_ratio,
            stale_steps=stale_steps,
            micro_batch_size=micro_batch_size,
            grader_mode=grader_mode,
            eval_every=eval_every,
            eval_tasks=eval_tasks,
            eval_samples=eval_samples,
        )
        out_dir = Path("/weights/runs") / run_name
        summary = run_colocated_grpo(
            cfg,
            output_dir=out_dir,
            hf_cache_dir=Path("/weights/hf-cache"),
        )
        weights_vol.commit()
        return summary

    @app.local_entrypoint()
    def bench_triton(
        batch_size: int = 2,
        seq_len: int = 2048,
        vocab_size: int = 152_064,
    ) -> None:
        res = bench_triton_kernel_modal.remote(
            batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size
        )
        out_local = Path("runs/triton_h100_bench.json")
        out_local.parent.mkdir(parents=True, exist_ok=True)
        out_local.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))

    @app.local_entrypoint()
    def baseline_eval(
        model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        num_tasks: int = 30,
        samples_per_task: int = 4,
        max_turns: int = 6,
        grader_mode: GraderMode = "hardened",
    ) -> None:
        result = run_baseline_eval_modal.remote(
            model=model,
            split="eval",
            num_tasks=num_tasks,
            samples_per_task=samples_per_task,
            max_turns=max_turns,
            grader_mode=grader_mode,
        )
        print(json.dumps(result, indent=2))

    @app.local_entrypoint()
    def train_grpo(
        run_name: str = "grpo_qwen15b_mis",
        model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        steps: int = 12,
        tasks_per_step: int = 4,
        group_size: int = 8,
        learning_rate: float = 2e-5,
        use_mis: bool = True,
        mis_max_ratio: float = 2.0,
        stale_steps: int = 0,
        micro_batch_size: int = 2,
        grader_mode: GraderMode = "hardened",
        eval_every: int = 4,
        eval_tasks: int = 30,
        eval_samples: int = 4,
    ) -> None:
        summary = run_grpo_train_modal.remote(
            run_name=run_name,
            model=model,
            steps=steps,
            tasks_per_step=tasks_per_step,
            group_size=group_size,
            learning_rate=learning_rate,
            use_mis=use_mis,
            mis_max_ratio=mis_max_ratio,
            stale_steps=stale_steps,
            micro_batch_size=micro_batch_size,
            grader_mode=grader_mode,
            eval_every=eval_every,
            eval_tasks=eval_tasks,
            eval_samples=eval_samples,
        )
        out_local = Path("runs") / f"{run_name}.json"
        out_local.parent.mkdir(parents=True, exist_ok=True)
        out_local.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        evals = summary["evals"]
        print(
            json.dumps(
                {
                    "run_name": run_name,
                    "saved_local": str(out_local),
                    "wall_seconds": round(summary["wall_seconds"], 1),
                    "eval_start": evals[0]["report"]["pass_at_k"],
                    "eval_end": evals[-1]["report"]["pass_at_k"],
                    "last_step": summary["steps"][-1] if summary["steps"] else None,
                },
                indent=2,
            )
        )
