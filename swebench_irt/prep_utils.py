from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def write_jsonl_records(output_path: Path, records: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def matrix_responses(
    responses: Mapping[str, int], all_items_sorted: list[str], no_complete_matrix: bool
) -> dict[str, int]:
    if no_complete_matrix:
        return dict(responses)
    return {item_id: int(responses.get(item_id, 0)) for item_id in all_items_sorted}


def build_records(
    subject_responses: Mapping[str, Mapping[str, int]],
    selected_subjects: list[str],
    all_items: set[str],
    no_complete_matrix: bool,
    skip_empty_sparse: bool = True,
) -> tuple[list[dict], list[tuple[str, int, int]]]:
    items_sorted = sorted(all_items)
    records: list[dict] = []
    summary: list[tuple[str, int, int]] = []
    for subject_id in selected_subjects:
        raw = subject_responses.get(subject_id, {})
        responses = matrix_responses(raw, items_sorted, no_complete_matrix)
        if no_complete_matrix and skip_empty_sparse and not responses:
            summary.append((subject_id, 0, 0))
            continue
        resolved_ct = sum(responses.values())
        summary.append((subject_id, len(responses), resolved_ct))
        records.append({"subject_id": subject_id, "responses": responses})
    return records, summary


def print_matrix_stats(
    records: list[dict],
    all_items: set[str],
    no_complete_matrix: bool,
    subject_label: str,
    output_path: Path,
    summary: list[tuple[str, int, int]] | None = None,
) -> None:
    print(f"Wrote {len(records)} {subject_label} to {output_path}")
    print(f"Unique tasks observed: {len(all_items)}")
    obs_total = sum(len(record["responses"]) for record in records)
    if records:
        counts = [len(record["responses"]) for record in records]
        print(f"Total observations: {obs_total}")
        print(
            f"Tasks per {subject_label.rstrip('s')}: "
            f"min={min(counts)} max={max(counts)} mean={obs_total / len(counts):.2f}"
        )
        print(f"Complete matrix: {'no (sparse)' if no_complete_matrix else 'yes'}")
    if summary:
        empty = sum(1 for _subject, count, _resolved in summary if count == 0)
        print(f"{subject_label.capitalize()} with no responses: {empty}")
