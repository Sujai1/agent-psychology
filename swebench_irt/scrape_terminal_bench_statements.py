#!/usr/bin/env python3
"""
Extract Terminal-Bench 2.0 task statements from the local `terminal-bench-2/`
repository and write a SWE-bench-like JSONL.

Output JSONL schema per line:
  {"task_id": "...", "problem_statement": "...", "patch": "...", "tests": "..."}

Gold patches:
  - We store the *entire* contents of `solution/solve.sh` (when present) in the
    `patch` field. Many tasks don't use a `diff --git` patch; they instead
    generate files, run commands, etc. Keeping the whole script captures the
    intended reference solution behavior.

terminal-bench-2 directory layout per task:
  terminal-bench-2/{task_id}/
    ├── task.toml          # metadata (category, difficulty, tags, etc.)
    ├── instruction.md     # task instruction (problem statement)
    ├── solution/          # reference solution
    │   └── solve.sh
    └── tests/             # test scripts
        ├── test.sh
        └── test_outputs.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def normalize_newlines_preserve_whitespace(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def extract_instruction_from_markdown(instruction_md_path: Path) -> Optional[str]:
    """Read instruction from an instruction.md file."""
    if not instruction_md_path.exists():
        return None
    text = instruction_md_path.read_text(encoding="utf-8")
    text = normalize_text(text)
    return text if text else None


def extract_patch_from_solution_dir(task_id: str, solution_dir: Path) -> str:
    """
    Return the contents of the solution script from the solution/ directory.

    Looks for solve.sh first, then falls back to any .sh file.
    Raises an error if no solution script is found.
    """
    if not solution_dir.exists() or not solution_dir.is_dir():
        raise FileNotFoundError(
            f"Solution directory not found for task '{task_id}': {solution_dir}"
        )

    # Prefer solve.sh
    solve_sh = solution_dir / "solve.sh"
    if solve_sh.exists():
        s = normalize_newlines_preserve_whitespace(
            solve_sh.read_text(encoding="utf-8")
        )
        if not s.endswith("\n"):
            s += "\n"
        return s

    # Fall back to first .sh file
    sh_files = sorted(solution_dir.glob("*.sh"))
    if sh_files:
        s = normalize_newlines_preserve_whitespace(
            sh_files[0].read_text(encoding="utf-8")
        )
        if not s.endswith("\n"):
            s += "\n"
        return s

    raise FileNotFoundError(
        f"No solution script (.sh) found for task '{task_id}' in {solution_dir}"
    )


def _read_text_truncated(path: Path, *, max_chars: int = 50_000) -> str:
    """
    Read a text file with UTF-8 and truncate extremely large files to keep JSONL
    and downstream prompts reasonable.
    """
    try:
        s = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        s = path.read_text(encoding="utf-8", errors="replace")
    s = normalize_newlines_preserve_whitespace(s)
    if len(s) > int(max_chars):
        return s[: int(max_chars)].rstrip() + "\n\n# [truncated]\n"
    return s


def extract_tests_from_task_dir(task_dir: Path) -> str:
    """
    Extract Terminal-Bench tests in a prompt-friendly form.

    Includes all files under `tests/` (test scripts and expected outputs).
    """
    chunks: List[str] = []

    tests_dir = task_dir / "tests"
    if tests_dir.exists() and tests_dir.is_dir():
        for p in sorted(tests_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(task_dir).as_posix()
            chunks.append(f"### {rel}\n{_read_text_truncated(p)}")

    s = "\n\n".join(chunks)
    return normalize_text(s) if s.strip() else ""


def _fetch_utf8_text(url: str) -> str:
    try:
        with urlopen(url, timeout=30) as resp:
            raw = resp.read()
    except URLError as e:
        raise RuntimeError(f"Failed fetching remote test file: {url}") from e
    return normalize_newlines_preserve_whitespace(raw.decode("utf-8"))


def extract_tests_with_remote_fallback(task_id: str, task_path: Path) -> str:
    tests = extract_tests_from_task_dir(task_path)
    if tests:
        return tests

    remote_files = _REMOTE_TEST_SOURCES.get(task_id)
    if not remote_files:
        return ""

    chunks: List[str] = []
    for rel, url in sorted(remote_files.items()):
        body = _fetch_utf8_text(url)
        if not body.strip():
            raise RuntimeError(f"Remote test file was empty for task '{task_id}': {url}")
        chunks.append(f"### {rel}\n{body}")

    return normalize_text("\n\n".join(chunks))


def load_task_list(meta_json_path: Path) -> List[str]:
    meta = json.loads(meta_json_path.read_text(encoding="utf-8"))
    task_list = meta.get("task_list")
    if not isinstance(task_list, list) or not all(isinstance(t, str) for t in task_list):
        raise ValueError(f"Invalid task_list in meta json: {meta_json_path}")
    return list(task_list)


# Project root: swebench_irt/ is one level below model_irt/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Path to write JSONL, e.g. terminal_bench_tasks.jsonl")
    ap.add_argument(
        "--tasks-dir",
        default=str(_PROJECT_ROOT / "terminal-bench-2"),
        help="Path to the terminal-bench-2 repository root (tasks are top-level dirs)",
    )
    ap.add_argument(
        "--meta",
        default=str(_PROJECT_ROOT / "data" / "terminal_bench" / "terminal_bench_2.0.meta.json"),
        help="Path to a meta JSON containing a `task_list` to filter tasks (recommended)",
    )
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process this many tasks (debugging)")
    args = ap.parse_args()

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        raise FileNotFoundError(f"tasks dir not found: {tasks_dir}")

    meta_path = Path(args.meta)
    if meta_path.exists():
        task_ids = load_task_list(meta_path)
    else:
        # Discover tasks: directories containing task.toml
        task_ids = sorted(
            p.name for p in tasks_dir.iterdir()
            if p.is_dir() and (p / "task.toml").exists()
        )
    if args.limit and args.limit > 0:
        task_ids = task_ids[: int(args.limit)]

    records = []
    missing_instruction: List[str] = []
    missing_task_dir: List[str] = []
    patches_found = 0
    tests_found = 0

    for tid in task_ids:
        task_path = tasks_dir / tid
        if not task_path.exists() or not (task_path / "task.toml").exists():
            missing_task_dir.append(tid)
            continue

        instruction_md = task_path / "instruction.md"
        instr = extract_instruction_from_markdown(instruction_md)
        if not instr:
            missing_instruction.append(tid)
            continue

        # Extract metadata from task.toml
        with open(task_path / "task.toml", "rb") as f:
            task_toml = tomllib.load(f)
        metadata_section = task_toml.get("metadata", {})

        patch = extract_patch_from_solution_dir(tid, task_path / "solution")
        patches_found += 1

        tests = extract_tests_with_remote_fallback(tid, task_path)
        if tests:
            tests_found += 1

        records.append({
            "task_id": tid,
            "problem_statement": instr,
            "patch": patch,
            "tests": tests,
            "category": metadata_section.get("category", ""),
            "tags": metadata_section.get("tags", []),
            "difficulty": metadata_section.get("difficulty", ""),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} tasks to {args.out}")
    print(f"Found {patches_found} tasks with non-empty patches")
    print(f"Found {tests_found} tasks with non-empty tests")
    if missing_task_dir:
        print(f"WARNING: missing task dir/toml for {len(missing_task_dir)} tasks, e.g. {missing_task_dir[:10]}")
    if missing_instruction:
        print(f"WARNING: failed to extract instruction for {len(missing_instruction)} tasks, e.g. {missing_instruction[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
