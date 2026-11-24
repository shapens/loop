from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from loop.envs.sandbox import SandboxBackend
from loop.rollout.protocol import TrajectoryBuilder, apply_tool_call, parse_tool_call
from loop.types import GraderMode, TaskSpec, Trajectory

SYSTEM_PROMPT = (
    "You are a precise software engineering agent fixing bugs in a Python repository.\n"
    "On each turn, think briefly and emit exactly ONE tool action:\n"
    "- Overwrite a source file:\n"
    '  <write path="src/example.py">\n'
    "  # complete updated file content\n"
    "  </write>\n"
    "- Run the visible unit tests:\n"
    "  <run_tests/>\n"
    "- Stop once all fixes are written and verified:\n"
    "  <finish/>\n"
    "Rules:\n"
    "1. Only edit files under src/ — never modify tests/.\n"
    '2. Write the complete file inside <write path="...">...</write> (no markdown fences).\n'
    "3. Visible tests only check basic behavior — fix EVERY requirement and file named in the Task "
    "before calling <finish/>.\n"
    "4. After editing, run <run_tests/> to verify, then emit <finish/>."
)


@dataclass(frozen=True)
class BatchedGeneration:
    """One assistant turn output with exact sampled token IDs and rollout logprobs."""

    text: str
    token_ids: list[int]
    logprobs: list[float]


BatchGenerateFn = Callable[[list[list[int]]], list[BatchedGeneration]]
TokenizeFn = Callable[[str], list[int]]
FormatInitFn = Callable[[str, str], str]
FormatObsFn = Callable[[str], str]


def build_user_prompt(task: TaskSpec) -> str:
    sections: list[str] = [f"Task:\n{task.prompt}\n\nCurrent repository files:"]
    for rel in sorted(task.repo_files.keys()):
        content = task.repo_files[rel]
        if not content:
            continue
        sections.append(f"--- {rel} ---\n{content}")
    return "\n\n".join(sections)


def format_chatml_initial(system_prompt: str, user_prompt: str) -> str:
    """Native Qwen/ChatML prompt format (keeps instruct models in-distribution)."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def format_chatml_observation(observation: str) -> str:
    return (
        "<|im_end|>\n"
        f"<|im_start|>user\nTool result:\n{observation}\n"
        "Continue with your next action.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def format_initial_prompt(task: TaskSpec) -> str:
    return format_chatml_initial(SYSTEM_PROMPT, build_user_prompt(task))


def format_observation_turn(observation: str) -> str:
    return format_chatml_observation(observation)


@dataclass
class _EpisodeState:
    task: TaskSpec
    prompt_id: str
    workspace: dict[str, str]
    builder: TrajectoryBuilder
    turns: int = 0
    done: bool = False
    rollout_ms: float = 0.0
    sandbox_ms: float = 0.0


def run_episodes_batched(
    tasks: Sequence[TaskSpec],
    *,
    samples_per_task: int,
    generate_batch: BatchGenerateFn,
    tokenize: TokenizeFn,
    sandbox: SandboxBackend,
    max_turns: int = 6,
    policy_version: int = 0,
    grader_mode: GraderMode = "hardened",
    max_sandbox_workers: int = 16,
    format_initial: FormatInitFn = format_chatml_initial,
    format_observation: FormatObsFn = format_chatml_observation,
) -> list[Trajectory]:
    """Run N * samples_per_task multi-turn coding episodes using lockstep batched generation.

    Batching every active episode's next turn into one `generate_batch` call keeps vLLM's
    continuous batching saturated instead of starving the GPU on serial tool calls.
    """
    if samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive")
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")

    states: list[_EpisodeState] = []
    for task in tasks:
        init_text = format_initial(SYSTEM_PROMPT, build_user_prompt(task))
        init_tokens = tokenize(init_text)
        for sample_idx in range(samples_per_task):
            builder = TrajectoryBuilder()
            builder.append_observation(init_tokens)
            states.append(
                _EpisodeState(
                    task=task,
                    prompt_id=f"{task.task_id}#s{sample_idx}",
                    workspace=dict(task.repo_files),
                    builder=builder,
                )
            )

    with ThreadPoolExecutor(max_workers=max(1, max_sandbox_workers)) as pool:
        for _turn in range(max_turns):
            active = [s for s in states if not s.done]
            if not active:
                break

            prompts = [list(s.builder.token_ids) for s in active]
            t_gen = time.perf_counter()
            gens = generate_batch(prompts)
            batch_gen_ms = (time.perf_counter() - t_gen) * 1000.0
            if len(gens) != len(active):
                raise RuntimeError(
                    f"generate_batch returned {len(gens)} outputs for {len(active)} prompts"
                )
            per_ep_gen_ms = batch_gen_ms / len(active)

            # Dispatch tool calls across active episodes concurrently.
            pending_tools: list[tuple[_EpisodeState, Future[tuple[str, bool, float]]]] = []
            for state, gen in zip(active, gens, strict=True):
                state.turns += 1
                state.rollout_ms += per_ep_gen_ms
                state.builder.append_assistant(gen.token_ids, gen.logprobs)
                call = parse_tool_call(gen.text)
                if call is None:
                    state.done = True
                    continue
                fut = pool.submit(
                    apply_tool_call,
                    call,
                    state.workspace,
                    sandbox,
                    state.task.timeout_seconds,
                )
                pending_tools.append((state, fut))

            for state, fut in pending_tools:
                obs, finished, sb_ms = fut.result()
                state.sandbox_ms += float(sb_ms)
                if finished:
                    state.done = True
                else:
                    obs_tokens = tokenize(format_observation(obs))
                    state.builder.append_observation(obs_tokens)

        # Grade all finished workspaces concurrently.
        grade_futs = [
            (state, pool.submit(sandbox.grade, state.task, state.workspace, grader_mode))
            for state in states
        ]
        trajectories: list[Trajectory] = []
        for state, gfut in grade_futs:
            receipt = gfut.result()
            traj = Trajectory(
                task_id=state.task.task_id,
                prompt_id=state.prompt_id,
                token_ids=state.builder.token_ids,
                loss_mask=state.builder.loss_mask,
                rollout_logprobs=state.builder.rollout_logprobs,
                policy_version=policy_version,
                turns=state.turns,
                receipt=receipt,
                rollout_ms=state.rollout_ms,
                sandbox_ms=state.sandbox_ms + receipt.duration_ms,
            )
            traj.validate()
            trajectories.append(traj)

    return trajectories
