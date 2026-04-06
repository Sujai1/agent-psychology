from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

from prep_utils import build_records, print_matrix_stats, resolve_path, write_jsonl_records

logger = logging.getLogger(__name__)


def _list_items(value: object) -> Iterable[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]


def load_results_json(results_path: Path) -> Tuple[Dict[str, int], set[str]]:
    with results_path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected results.json format at {results_path}")

    resolved = set(_list_items(data.get("resolved", [])))
    all_items = set()
    for value in data.values():
        all_items.update(_list_items(value))

    responses = {iid: (1 if iid in resolved else 0) for iid in all_items}
    return responses, all_items


def load_logs(logs_dir: Path) -> Tuple[Dict[str, int], set[str]]:
    responses: Dict[str, int] = {}
    json_errors = []
    missing_resolved = []

    for report_path in logs_dir.glob("*/report.json"):
        try:
            with report_path.open() as f:
                record = json.load(f)
        except json.JSONDecodeError as e:
            json_errors.append((report_path, str(e)))
            continue
        resolved = record.get("resolved")
        if resolved is None:
            missing_resolved.append(report_path)
            continue
        instance_id = report_path.parent.name
        responses[instance_id] = 1 if bool(resolved) else 0

    if json_errors:
        logger.warning(
            f"Skipped {len(json_errors)} report.json files with JSON parse errors. "
            f"First error: {json_errors[0]}"
        )
    if missing_resolved:
        logger.warning(
            f"Skipped {len(missing_resolved)} report.json files without 'resolved' field. "
            f"First: {missing_resolved[0]}"
        )

    return responses, set(responses.keys())


def collect_agent(agent_dir: Path) -> Tuple[Dict[str, int], set[str], str]:
    results_path = agent_dir / "results" / "results.json"
    logs_dir = agent_dir / "logs"
    if results_path.exists():
        responses, items = load_results_json(results_path)
        return responses, items, "results.json"
    if logs_dir.exists():
        responses, items = load_logs(logs_dir)
        return responses, items, "logs"
    return {}, set(), "missing"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SWE-bench Verified response JSONL")
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments/evaluation/verified",
        help="Path to experiments/evaluation/verified",
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        default=None,
        help="Include agents with date prefix <= cutoff (YYYYMMDD)",
    )
    parser.add_argument(
        "--max_agents",
        type=int,
        default=None,
        help="Limit number of agents after sorting by name",
    )
    parser.add_argument(
        "--no_complete_matrix",
        action="store_true",
        help="Don't fill missing tasks (sparse matrix). Default is to fill with 0.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/swebench_verified/responses.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    experiments_dir = resolve_path(args.experiments_dir)
    output_path = resolve_path(args.output_path)

    all_items: set[str] = set()
    agents_payload = []

    if not experiments_dir.exists():
        raise FileNotFoundError(f"experiments_dir not found: {experiments_dir}")

    agent_dirs = [d for d in sorted(experiments_dir.iterdir()) if d.is_dir()]
    if not agent_dirs:
        raise ValueError(f"No agent directories found under {experiments_dir}")

    for agent_dir in agent_dirs:
        responses, items, source = collect_agent(agent_dir)
        if responses:
            all_items.update(items)
        agents_payload.append((agent_dir, responses, source))

    selected = []
    for agent_dir, responses, source in agents_payload:
        if args.cutoff_date is not None:
            date_prefix = agent_dir.name.split("_", 1)[0]
            if date_prefix > args.cutoff_date:
                continue
        selected.append((agent_dir, responses, source))

    if args.max_agents is not None:
        selected = selected[: args.max_agents]

    subject_responses = {agent_dir.name: responses for agent_dir, responses, _source in selected}
    selected_subjects = [agent_dir.name for agent_dir, _responses, _source in selected]
    records, summary = build_records(
        subject_responses=subject_responses,
        selected_subjects=selected_subjects,
        all_items=all_items,
        no_complete_matrix=args.no_complete_matrix,
        skip_empty_sparse=True,
    )
    write_jsonl_records(output_path, records)
    print_matrix_stats(
        records=records,
        all_items=all_items,
        no_complete_matrix=args.no_complete_matrix,
        subject_label="agents",
        output_path=output_path,
        summary=summary,
    )


if __name__ == "__main__":
    main()
