"""
Predictive Clinical Benchmark — 评测工具包

用法:
    from eval import run_benchmark, construct_inference_prompt, parse_model_output
    from eval.metrics import compute_binary_benefit, compute_weighted_kappa
"""

from .metrics import (
    compute_binary_benefit,
    compute_weighted_kappa,
    compute_recist_metrics,
    compute_direction_accuracy,
    compute_toxicity_metrics,
    compute_calibration,
    compute_composite_score,
    CSF_MAPPING,
    SYMPTOM_MAPPING,
)
from .parser import parse_model_output, validate_output
from .prompts import (
    construct_inference_prompt,
    format_judge_prompt_A1,
    format_judge_prompt_A2,
    format_judge_prompt_A3,
    format_judge_prompt_A4,
)
from .runner import run_benchmark
from .bootstrap import bootstrap_ci
from .na_filter import get_na_skip_dimensions
from .compliance import compute_temporal_compliance, compute_global_compliance
from .sanity import check_random_baseline, run_sanity_checks

__all__ = [
    # Metrics
    "compute_binary_benefit",
    "compute_weighted_kappa",
    "compute_recist_metrics",
    "compute_direction_accuracy",
    "compute_toxicity_metrics",
    "compute_calibration",
    "compute_composite_score",
    "CSF_MAPPING",
    "SYMPTOM_MAPPING",
    # Parser
    "parse_model_output",
    "validate_output",
    # Prompts
    "construct_inference_prompt",
    "format_judge_prompt_A1",
    "format_judge_prompt_A2",
    "format_judge_prompt_A3",
    "format_judge_prompt_A4",
    # Runner
    "run_benchmark",
    # Bootstrap
    "bootstrap_ci",
    # NA filter
    "get_na_skip_dimensions",
    # Compliance
    "compute_temporal_compliance",
    "compute_global_compliance",
    # Sanity
    "check_random_baseline",
    "run_sanity_checks",
]
