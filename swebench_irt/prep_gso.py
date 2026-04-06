from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Tuple

from prep_utils import build_records, print_matrix_stats, resolve_path, write_jsonl_records

logger = logging.getLogger(__name__)

def _list_str(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]


def _collect_task_ids(instance_sets: object) -> set[str]:
    if not isinstance(instance_sets, dict):
        return set()

    items: set[str] = set()

    items.update(_list_str(instance_sets.get("completed_ids")))
    items.update(_list_str(instance_sets.get("passed_ids")))
    items.update(_list_str(instance_sets.get("test_failed_ids")))
    items.update(_list_str(instance_sets.get("base_failed_ids")))
    items.update(_list_str(instance_sets.get("patch_failed_ids")))
    items.update(_list_str(instance_sets.get("error_ids")))
    items.update(_list_str(instance_sets.get("empty_patch_ids")))

    for k, v in instance_sets.items():
        if isinstance(k, str) and k.endswith("_ids"):
            items.update(_list_str(v))

    return items


def load_report(report_path: Path, *, success_ids_key: str) -> Tuple[set[str], set[str]]:
    with report_path.open() as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected report JSON format at {report_path}")

    instance_sets = obj.get("instance_sets")
    items = _collect_task_ids(instance_sets)
    successes = set()
    if isinstance(instance_sets, dict):
        successes = set(_list_str(instance_sets.get(success_ids_key)))
    return items, successes


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare GSO report JSONs into IRT response JSONL")
    p.add_argument(
        "--reports_dir",
        type=str,
        default="gso-experiments/results/reports",
        help="Directory containing per-model report JSONs",
    )
    p.add_argument(
        "--model_regex",
        type=str,
        default=None,
        help="Only include reports whose model_name (filename stem) matches this regex.",
    )
    p.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Limit number of subjects after sorting by subject_id",
    )
    p.add_argument(
        "--no_complete_matrix",
        action="store_true",
        help="Don't fill missing tasks (sparse matrix). Default is to fill with 0.",
    )
    p.add_argument(
        "--success_ids_key",
        type=str,
        default="opt_commit_ids",
        help='Which `instance_sets` key to treat as "success". Default: opt_commit_ids',
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="data/gso/responses.jsonl",
        help="Output JSONL path",
    )
    args = p.parse_args()

    reports_dir = resolve_path(args.reports_dir)
    output_path = resolve_path(args.output_path)

    if not reports_dir.exists():
        raise FileNotFoundError(f"reports_dir not found: {reports_dir}")

    rx = re.compile(args.model_regex) if args.model_regex else None

    per_subject: dict[str, Tuple[set[str], set[str]]] = {}
    all_items: set[str] = set()

    report_paths = sorted(reports_dir.glob("*.json"))
    if not report_paths:
        raise ValueError(f"No report JSONs found under {reports_dir}")

    for report_path in report_paths:
        subject_id = report_path.stem
        if rx is not None and not rx.search(subject_id):
            continue

        items, successes = load_report(report_path, success_ids_key=args.success_ids_key)
        if not items:
            logger.warning(f"Skipping {report_path} (no task ids found)")
            continue

        per_subject[subject_id] = (items, successes)
        all_items.update(items)

    if not per_subject:
        raise ValueError("No subjects selected (after filtering / empty reports).")

    selected_subjects = sorted(per_subject.keys())
    if args.max_subjects is not None:
        selected_subjects = selected_subjects[: args.max_subjects]

    subject_responses: Dict[str, Dict[str, int]] = {}
    for subject_id in selected_subjects:
        items, successes = per_subject[subject_id]
        subject_responses[subject_id] = {item_id: (1 if item_id in successes else 0) for item_id in items}
    records, summary = build_records(
        subject_responses=subject_responses,
        selected_subjects=selected_subjects,
        all_items=all_items,
        no_complete_matrix=args.no_complete_matrix,
        skip_empty_sparse=False,
    )
    write_jsonl_records(output_path, records)
    print_matrix_stats(
        records=records,
        all_items=all_items,
        no_complete_matrix=args.no_complete_matrix,
        subject_label="subjects",
        output_path=output_path,
        summary=summary,
    )
    if summary:
        best = max(summary, key=lambda t: t[2])
        worst = min(summary, key=lambda t: t[2])
        print(f"Most successes: {best[0]} ({best[2]}/{best[1]})")
        print(f"Fewest successes: {worst[0]} ({worst[2]}/{worst[1]})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

