from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from loop.envs.generator import generate_split
from loop.envs.grader import run_self_check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loop", description="shapens/loop RLVR CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser(
        "self-check",
        help="Verify generated tasks fail untouched, pass solution, and block tampering",
    )
    sc.add_argument("--count", type=int, default=30, help="Number of tasks per split to verify")

    gen = sub.add_parser(
        "gen-tasks", help="Export deterministic train/eval task manifests to JSONL"
    )
    gen.add_argument("--train", type=int, default=200)
    gen.add_argument("--eval", type=int, default=50)
    gen.add_argument("--out-dir", type=Path, default=Path("runs/tasks"))

    args = parser.parse_args(argv)

    if args.cmd == "self-check":
        train_tasks = generate_split("train", args.count)
        eval_tasks = generate_split("eval", args.count)
        res_train = run_self_check(train_tasks)
        res_eval = run_self_check(eval_tasks)
        print(json.dumps({"train": res_train, "eval": res_eval}, indent=2))
        return 0

    if args.cmd == "gen-tasks":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for split, n in (("train", args.train), ("eval", args.eval)):
            tasks = generate_split(split, n)  # type: ignore[arg-type]
            out_path = args.out_dir / f"{split}.jsonl"
            with out_path.open("w", encoding="utf-8") as fh:
                for task in tasks:
                    fh.write(json.dumps(asdict(task)) + "\n")
            print(f"wrote {len(tasks)} tasks -> {out_path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
