from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import pandas as pd


SensitiveGroupT: TypeAlias = list[str] | list[int] | list[bool] | list[str | int | bool] 

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
class MetricResult:
    """Result of a metric for a single sensitive group."""

    sensitive_group: SensitiveGroupT
    result: float


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
class FairnessSummaryResult:
    """Summary of a difference-based fairness metric."""

    disparity: float
    privileged_sensitive_group: SensitiveGroupT
    unprivileged_sensitive_group: SensitiveGroupT


@dataclass
class FairnessRatioSummaryResult:
    """Summary of a ratio-based fairness metric."""

    ratio: float
    privileged_sensitive_group: SensitiveGroupT
    unprivileged_sensitive_group: SensitiveGroupT


@dataclass
class RankResult:
    """Individual rank result for a group."""

    sensitive_group: SensitiveGroupT
    score: float


@dataclass
class BaseMetricResult:
    """Base class for metric result types that can be used in disparity calculations."""

    sensitive_group: SensitiveGroupT
    data: float | pd.Series | np.ndarray | list[float] | MetricsDict


@dataclass
class SelectionRateResult(BaseMetricResult):
    """Result of selection rate for a single sensitive group."""

    data: float
    selection_rate_in_data: float
    selection_rate_in_predictions: float


@dataclass
class PerformanceMetricResult(BaseMetricResult):
    """Result of performance metrics for a single sensitive group with dynamic metrics."""

    metrics: MetricsDict | None = None

    def __post_init__(self):
        """Ensure metrics and data are synchronized."""
        if self.metrics is not None and self.data is None:
            self.data = self.metrics
        elif self.data is not None and self.metrics is None:
            self.metrics = self.data

    def __getattr__(self, name: str):
        """Allow attribute access to metric values."""
        if self.metrics is not None and name in self.metrics:
            return self.metrics[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


@dataclass
class ConfusionMatrixResult(BaseMetricResult):
    """Result of confusion matrix metrics for a single sensitive group with dynamic metrics."""

    metrics: MetricsDict | None = None

    def __post_init__(self):
        """Ensure metrics and data are synchronized."""
        if self.metrics is not None and self.data is None:
            self.data = self.metrics
        elif self.data is not None and self.metrics is None:
            self.metrics = self.data

    def __getattr__(self, name: str):
        """Allow attribute access to metric values."""
        if self.metrics is not None and name in self.metrics:
            return self.metrics[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


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
class DemographicParitySummaryResult:
    """Summary result for Demographic Parity metrics."""

    privileged_sensitive_group: SensitiveGroupT
    unprivileged_sensitive_group: SensitiveGroupT
    demographic_parity_difference: float
    demographic_parity_ratio: float
    label: Literal["demographic_parity_difference", "demographic_parity_ratio"] = (
        "demographic_parity_difference"
    )


@dataclass
class DisparateImpactSummaryResult:
    """Summary result for Disparate Impact metrics."""

    privileged_sensitive_group: SensitiveGroupT
    unprivileged_sensitive_group: SensitiveGroupT
    label: Literal["disparate_impact_difference", "disparate_impact_ratio"]
    disparate_impact_difference: float | None = None
    disparate_impact_ratio: float | None = None


@dataclass
class EqualOpportunitySummaryResult:
    """Summary result for Equal Opportunity metrics."""

    privileged_sensitive_group: SensitiveGroupT
    unprivileged_sensitive_group: SensitiveGroupT
    equal_opportunity_difference: float | None = None
    equal_opportunity_ratio: float | None = None
    label: Literal["equal_opportunity_difference", "equal_opportunity_ratio"] = (
        "equal_opportunity_difference"
    )


@dataclass
class FalsePositiveRateSummaryResult:
    """Summary result for False Positive Rate metrics."""

    privileged_sensitive_group: SensitiveGroupT
    unprivileged_sensitive_group: SensitiveGroupT
    false_positive_rate_difference: float | None = None
    false_positive_rate_ratio: float | None = None
    label: Literal["false_positive_rate_difference", "false_positive_rate_ratio"] = (
        "false_positive_rate_difference"
    )


@dataclass
class EqualisedOddsSummaryResult:
    """Summary result for Equalised Odds metrics."""

    privileged_sensitive_group: SensitiveGroupT
    unprivileged_sensitive_group: SensitiveGroupT
    equalised_odds_difference: float | None = None
    equalised_odds_ratio: float | None = None


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
        DemographicParitySummaryResult
        | DisparateImpactSummaryResult
        | EqualOpportunitySummaryResult
        | EqualisedOddsSummaryResult
        | FalsePositiveRateSummaryResult,
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
    data: list[SelectionRateResult]


@dataclass
class GroupData:
    """
    Data structure produced by helper functions such as
    `Dataset.get_*_for_all_groups`.

    Attributes
    ----------
    sensitive_group : list[str] | np.ndarray
        Labels identifying the sensitive group.
    data : pd.Series | np.ndarray | list[float] | None
        Raw per-record metric values (present for some helpers).
    result : float | pd.Series | np.ndarray | list[float] | None
        Pre-computed score for the group (present for others).
    """

    sensitive_group: list[str] | np.ndarray
    data: pd.Series | np.ndarray | list[float] | None = None
    result: float | pd.Series | np.ndarray | list[float] | None = None
