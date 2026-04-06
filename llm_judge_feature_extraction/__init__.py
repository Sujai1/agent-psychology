"""LLM Judge feature extraction module.

Provides batched feature extraction from tasks using LLMs, with per-feature
info level isolation and prefix caching.

Example usage:
    from llm_judge_feature_extraction import (
        BatchedFeatureExtractor,
        get_task_context,
        load_tasks,
    )

    ctx = get_task_context("swebench_verified")
    extractor = BatchedFeatureExtractor(
        feature_names=["solution_hint", "problem_clarity", "solution_complexity"],
        task_context=ctx,
    )

    tasks = load_tasks("swebench_verified")
    csv_path = extractor.run(tasks, output_dir=Path("output/"))

CLI usage:
    python -m llm_judge_feature_extraction extract --all --dataset swebench_verified --dry-run
    python -m llm_judge_feature_extraction extract --all --dataset terminalbench
"""

from llm_judge_feature_extraction.api_client import LLMApiClient
from llm_judge_feature_extraction.batched_extractor import BatchedFeatureExtractor
from llm_judge_feature_extraction.feature_registry import (
    ALL_FEATURES,
    get_all_feature_names,
    get_features,
    get_features_by_level,
)
from llm_judge_feature_extraction.prompt_config import FeatureDefinition, InfoLevel
from llm_judge_feature_extraction.response_parser import (
    parse_llm_response,
    validate_features,
)
from llm_judge_feature_extraction.task_context import get_task_context
from llm_judge_feature_extraction.task_loaders import load_tasks

__all__ = [
    "BatchedFeatureExtractor",
    "LLMApiClient",
    "FeatureDefinition",
    "InfoLevel",
    "ALL_FEATURES",
    "get_all_feature_names",
    "get_features",
    "get_features_by_level",
    "get_task_context",
    "load_tasks",
    "parse_llm_response",
    "validate_features",
]
