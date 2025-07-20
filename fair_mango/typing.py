from dataclasses import dataclass, asdict
from typing import TypedDict

import pandas as pd
import numpy as np

class DatasetTargetResult(TypedDict):
    """Result of target data for a sensitive group from Dataset."""
    sensitive: list[str]
    result: pd.Series

class DatasetGroupResult(TypedDict):
    """Result of group data from Dataset."""
    sensitive: list[str]
    result: pd.DataFrame

@dataclass
class MetricResult:
    """Result of a metric for a single sensitive group."""
    sensitive: list[str]  
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
    privileged_group: list[str] | None  
    unprivileged_group: list[str] | None  

@dataclass
class FairnessRatioSummaryResult:
    """Summary of a ratio-based fairness metric."""
    ratio: float
    privileged_group: list[str] | None   
    unprivileged_group: list[str] | None   

@dataclass
class RankResult:
    """Individual rank result for a group."""
    sensitive: list[str]  # Changed from tuple to list[str]
    score: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return asdict(self)

@dataclass
class DetailedPerformanceMetricsResult:
    """A collection of all performance metrics for a single sensitive group."""
    sensitive: list[str]  
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
    privileged_group: list[str] | None = None
    unprivileged_group: list[str] | None = None

@dataclass
class DisparateImpactSummaryResult:
    """Summary result for Disparate Impact metrics."""
    disparate_impact_difference: float | None = None
    disparate_impact_ratio: float | None = None
    privileged_group: list[str] | None = None
    unprivileged_group: list[str] | None = None

@dataclass
class EqualOpportunitySummaryResult:
    """Summary result for Equal Opportunity metrics."""
    equal_opportunity_difference: float | None = None
    equal_opportunity_ratio: float | None = None
    privileged_group: list[str] | None = None
    unprivileged_group: list[str] | None = None

@dataclass
class FalsePositiveRateSummaryResult:
    """Summary result for False Positive Rate metrics."""
    false_positive_rate_difference: float | None = None
    false_positive_rate_ratio: float | None = None
    privileged_group: list[str] | None = None
    unprivileged_group: list[str] | None = None

@dataclass
class EqualisedOddsSummaryResult:
    """Summary result for Equalised Odds metrics."""
    equalised_odds_difference: float | None = None
    equalised_odds_ratio: float | None = None
    privileged_group: list[str] | None = None
    unprivileged_group: list[str] | None = None 

@dataclass
class GroupData(TypedDict, total=False):
    """
    Dictionary produced by helper functions such as
    `Dataset.get_*_for_all_groups`.

    Keys
    ----
    sensitive : list[str] | np.ndarray
        Labels identifying the sensitive group.
    data      : pd.Series | np.ndarray | list[float]
        Raw per-record metric values (present for some helpers).
    result    : float | pd.Series | np.ndarray | list[float]
        Pre-computed score for the group (present for others).
    """
    sensitive: list[str] | np.ndarray
    data: pd.Series | np.ndarray | list[float]
    result: float | pd.Series | np.ndarray | list[float]