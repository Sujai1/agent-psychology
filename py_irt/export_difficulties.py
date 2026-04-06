"""
Stdlib-only utility to export per-question difficulty values from a py_irt
parameter file (e.g., parameters.json / best_parameters.json).

This intentionally avoids third-party dependencies (typer/rich/pyro/torch/etc.)
so it can be used in minimal environments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_params(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r))
            f.write("\n")


def export_question_difficulties(
    *,
    parameter_path: Path,
    output_path: Path,
    include_other_params: bool = True,
) -> None:
    params = _load_params(parameter_path)
    if "item_ids" not in params:
        raise ValueError("Missing `item_ids` in parameter file; is this a py_irt export?")
    if "diff" not in params:
        raise ValueError("Missing `diff` in parameter file; cannot export difficulty.")

    item_ids = params["item_ids"]
    id_pairs = sorted(((int(k), v) for k, v in item_ids.items()), key=lambda x: x[0])

    diffs = params["diff"]
    discs = params.get("disc")
    lambdas = params.get("lambdas")

    rows: List[Dict[str, Any]] = []
    for ix, item_id in id_pairs:
        row: Dict[str, Any] = {"item_ix": ix, "item_id": item_id, "diff": diffs[ix]}
        if include_other_params:
            if discs is not None:
                row["disc"] = discs[ix]
            if lambdas is not None:
                row["lambdas"] = lambdas[ix]
        rows.append(row)

    suffix = output_path.suffix.lower()
    if suffix in [".jsonl", ".jsonlines"]:
        _write_jsonl(output_path, rows)
        return
    if suffix == ".json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(rows, f)
        return
    if suffix == ".csv":
        fieldnames = ["item_ix", "item_id", "diff"]
        if include_other_params and discs is not None:
            fieldnames.append("disc")
        if include_other_params and lambdas is not None:
            fieldnames.append("lambdas")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return

    raise ValueError("output_path must end with one of: .csv, .json, .jsonl/.jsonlines")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export per-question difficulty from py_irt parameters.json/best_parameters.json"
    )
    parser.add_argument("parameter_path", type=Path, help="Path to parameters.json or best_parameters.json")
    parser.add_argument("output_path", type=Path, help="Output path ending with .csv/.json/.jsonl")
    parser.add_argument(
        "--include-other-params",
        action="store_true",
        default=False,
        help="Also export disc/lambdas if present (2PL/3PL/4PL).",
    )

    args = parser.parse_args(argv)
    export_question_difficulties(
        parameter_path=args.parameter_path,
        output_path=args.output_path,
        include_other_params=args.include_other_params,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

