from dataclasses import asdict, dataclass
from typing import TypedDict

import numpy as np
import pandas as pd


@dataclass
class DatasetTargetResult:
    """Result of target data for a sensitive group from Dataset."""

    sensitive_group: list[str]
    data: pd.Series

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return asdict(self)


@dataclass
class GroupRankingResult:
    """Result of group ranking."""
    
    sensitive_group: list[str]
    score: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return asdict(self)


@dataclass
class DatasetGroupResult:
    """Result of group data from Dataset."""

    sensitive_group: list[str]
    data: pd.DataFrame

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return asdict(self)


@dataclass
class MetricResult:
    """Result of a metric for a single sensitive group."""

    sensitive_group: list[str]
    result: float


@dataclass
class DisparityResult:
    """Result of disparity calculation between two groups."""

    group_1: list[str]
    group_2: list[str]
    disparity: float


@dataclass
class FairnessSummaryResult:
    """Summary of a difference-based fairness metric."""

    disparity: float
    privileged_sensitive_group: list[str] | None
    unprivileged_sensitive_group: list[str] | None


@dataclass
class FairnessRatioSummaryResult:
    """Summary of a ratio-based fairness metric."""

    ratio: float
    privileged_sensitive_group: list[str] | None
    unprivileged_sensitive_group: list[str] | None


@dataclass
class RankResult:
    """Individual rank result for a group."""

    sensitive_group: list[str]
    score: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return asdict(self)


@dataclass
class DetailedPerformanceMetricsResult:
    """A collection of all performance metrics for a single sensitive group."""

    sensitive_group: list[str]
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

    def to_dict(self) -> list[dict[str, object]]:
        """Convert to dictionary format for backward compatibility."""
        return [rank.to_dict() for rank in self.rankings]


@dataclass
class DemographicParitySummaryResult:
    """Summary result for Demographic Parity metrics."""

    demographic_parity_difference: float | None = None
    demographic_parity_ratio: float | None = None
    privileged_sensitive_group: list[str] | None = None
    unprivileged_sensitive_group: list[str] | None = None


@dataclass
class DisparateImpactSummaryResult:
    """Summary result for Disparate Impact metrics."""

    disparate_impact_difference: float | None = None
    disparate_impact_ratio: float | None = None
    privileged_sensitive_group: list[str] | None = None
    unprivileged_sensitive_group: list[str] | None = None


@dataclass
class EqualOpportunitySummaryResult:
    """Summary result for Equal Opportunity metrics."""

    equal_opportunity_difference: float | None = None
    equal_opportunity_ratio: float | None = None
    privileged_sensitive_group: list[str] | None = None
    unprivileged_sensitive_group: list[str] | None = None


@dataclass
class FalsePositiveRateSummaryResult:
    """Summary result for False Positive Rate metrics."""

    false_positive_rate_difference: float | None = None
    false_positive_rate_ratio: float | None = None
    privileged_sensitive_group: list[str] | None = None
    unprivileged_sensitive_group: list[str] | None = None


@dataclass
class EqualisedOddsSummaryResult:
    """Summary result for Equalised Odds metrics."""

    equalised_odds_difference: float | None = None
    equalised_odds_ratio: float | None = None
    privileged_sensitive_group: list[str] | None = None
    unprivileged_sensitive_group: list[str] | None = None


@dataclass
class SupersetFairnessRankingResult:
    """Result of fairness metric rankings for a sensitive group combination."""
    
    sensitive_group: list[str]
    rankings: dict[str, list[GroupRankingResult]]
    
    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return {
            "sensitive_group": self.sensitive_group,
            "rankings": {k: [r.to_dict() for r in v] for k, v in self.rankings.items()}
        }


@dataclass
class SupersetFairnessSummaryResult:
    """Result of fairness summary evaluation for a superset."""

    summaries: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary format for backward compatibility."""
        result: dict[str, object] = {"summaries": {}}
        summaries_dict: dict[str, object] = {}
        for key, value in self.summaries.items():
            if hasattr(value, "to_dict"):
                summaries_dict[key] = value.to_dict()
            else:
                summaries_dict[key] = value
        result["summaries"] = summaries_dict
        return result


@dataclass
class SupersetBiasResult:
    """Result of bias determination for a sensitive group combination."""
    
    sensitive_group: list[str]
    bias_results: dict[str, bool]
    
    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return {
            "sensitive_group": self.sensitive_group,
            "bias_results": self.bias_results
        }


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

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return asdict(self)
