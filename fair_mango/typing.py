from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import pandas as pd


SensitiveGroupT: TypeAlias = list[str] | list[int] | list[bool] | list[str | int | bool]
SensitiveGroupOptionalT: TypeAlias = (
    SensitiveGroupT | None
)  # when sensitive group identification is not possible (one group or all groups have identical scores).

SensitiveGroupTupleT: TypeAlias = tuple[str | int | bool, ...]

MetricsDict: TypeAlias = dict[str, list[float]]
"""Mapping of metric name to list of metric values for different groups or samples."""


@dataclass
class DatasetTargetResult:
    """Result of target data for a sensitive group from Dataset."""

    sensitive_group: SensitiveGroupT
    data: pd.Series


@dataclass
class GroupRankingResult:
    """Result of group ranking."""

    sensitive_group: SensitiveGroupT
    score: float


@dataclass
class DatasetGroupResult:
    """Result of group data from Dataset."""

    sensitive_group: SensitiveGroupT
    data: pd.DataFrame


@dataclass
class DisparityResult:
    """Result of disparity calculation between two groups."""

    group_1: SensitiveGroupT
    group_2: SensitiveGroupT
    disparity: float


class DisparityResultDict(TypedDict):
    """TypedDict for disparity calculation result between two groups."""

    group_1: SensitiveGroupT
    group_2: SensitiveGroupT
    disparity: float


@dataclass
class FairnessSummaryDifferenceResult:
    """Summary of a difference-based fairness metric when unfair."""

    difference: float
    privileged_sensitive_group: SensitiveGroupT
    unprivileged_sensitive_group: SensitiveGroupT


@dataclass
class FairnessSummaryDifferenceFairResult:
    """Summary of a difference-based fairness metric when fair."""

    difference: Literal[0]
    privileged_sensitive_group: None
    unprivileged_sensitive_group: None


@dataclass
class FairnessSummaryRatioResult:
    """Summary of a ratio-based fairness metric when unfair."""

    ratio: float
    privileged_sensitive_group: SensitiveGroupT
    unprivileged_sensitive_group: SensitiveGroupT


@dataclass
class FairnessSummaryRatioFairResult:
    """Summary of a ratio-based fairness metric when fair."""

    ratio: Literal[1]
    privileged_sensitive_group: None
    unprivileged_sensitive_group: None


@dataclass
class RankResult:
    """Individual rank result for a group."""

    sensitive_group: SensitiveGroupT
    score: float


@dataclass
class BaseMetricResult:
    """Base class for metric result types that can be used in disparity calculations."""

    sensitive_group: SensitiveGroupT
    data: float | MetricsDict


@dataclass
class SelectionRateResult(BaseMetricResult):
    """Result of selection rate for a single sensitive group."""

    data: float
    selection_rate_in_data: float
    selection_rate_in_predictions: float


@dataclass
class CombinedPerformanceResult(BaseMetricResult):
    """Unified result containing all performance metrics for a sensitive group."""

    selection_rate_in_data: float
    selection_rate_in_predictions: float

    accuracy: float | None = None
    balanced_accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None

    false_negative_rate: float | None = None
    false_positive_rate: float | None = None
    true_negative_rate: float | None = None
    true_positive_rate: float | None = None

    specificity: float | None = None
    negative_predictive_value: float | None = None

    def __post_init__(self):
        """Maintaining compatibility with BaseMetricResult by providing the specific
        selection rate value for the combined performance result.
        """
        self.data = self.selection_rate_in_predictions


@dataclass
class PerformanceMetricResult(BaseMetricResult):
    """Result of performance metrics for a single sensitive group with dynamic metrics."""

    data: MetricsDict


@dataclass
class ConfusionMatrixResult(BaseMetricResult):
    """Result of confusion matrix metrics for a single sensitive group with dynamic metrics."""

    data: MetricsDict


@dataclass
class DetailedPerformanceMetricsResult:
    """A collection of all performance metrics for a single sensitive group."""

    sensitive_group: SensitiveGroupT
    selection_rate_in_data: float
    selection_rate_in_predictions: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_negative_rate: float
    false_positive_rate: float
    true_negative_rate: float
    true_positive_rate: float


@dataclass
class SupersetFairnessRank:
    """Result of a superset fairness ranking."""

    sensitive_attributes: list[str]
    rank: list[RankResult]


@dataclass
class SupersetPerformanceEvaluation:
    """Result of a superset performance evaluation."""

    sensitive_attributes: list[str]
    results: list[DetailedPerformanceMetricsResult]


@dataclass
class FairnessRankingResult:
    """Result container for fairness metric rankings."""

    rankings: list[RankResult]


@dataclass
class SupersetFairnessRankingResult:
    """Result of fairness metric rankings for a sensitive group combination."""

    sensitive_group: SensitiveGroupT
    rankings: dict[str, list[RankResult]]


@dataclass
class SupersetFairnessSummaryResult:
    """Result of fairness summary evaluation for a superset."""

    summaries: dict[
        str,
        FairnessSummaryDifferenceResult
        | FairnessSummaryDifferenceFairResult
        | FairnessSummaryRatioResult
        | FairnessSummaryRatioFairResult,
    ]


@dataclass
class SupersetBiasResult:
    """Result of bias determination for a sensitive group combination."""

    sensitive_group: SensitiveGroupT
    bias_results: dict[str, bool]


@dataclass
class SupersetPerformanceMetricsResult:
    """Result container for superset performance metrics."""

    sensitive_group: tuple[str, ...]
    data: list[CombinedPerformanceResult]


@dataclass
class GroupData:
    """
    Data structure produced by helper functions such as
    `Dataset.get_*_for_all_groups`.

    Attributes
    ----------
    sensitive_group : list[str] | np.ndarray
        Labels identifying the sensitive group.
    data : pd.Series | np.ndarray | list[float]
        Raw per-record metric values (present for some helpers).
    result : float | pd.Series | np.ndarray | list[float]
        Pre-computed score for the group (present for others).
    """

    sensitive_group: list[str] | np.ndarray
    data: pd.Series | np.ndarray | list[float]
    result: float | pd.Series | np.ndarray | list[float]
