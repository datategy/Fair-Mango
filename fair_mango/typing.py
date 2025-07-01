from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SensitiveGroup = tuple[str, ...]

RankResult = dict[SensitiveGroup, float]

class MetricResult(TypedDict):
    """Result of a metric for a single sensitive group."""
    sensitive: NDArray[np.str_]  
    result: pd.Series 

class FairnessSummaryResult(TypedDict):
    """Summary of a difference-based fairness metric."""
    disparity: float
    privileged_group: SensitiveGroup | None
    unprivileged_group: SensitiveGroup | None

class FairnessRatioSummaryResult(TypedDict):
    """Summary of a ratio-based fairness metric."""
    ratio: float
    privileged_group: SensitiveGroup | None
    unprivileged_group: SensitiveGroup | None

class DetailedPerformanceMetricsResult(TypedDict):
    """A collection of all performance metrics for a single sensitive group."""
    sensitive: SensitiveGroup
    selection_rate_in_data: NDArray[np.float64]
    selection_rate_in_predictions: NDArray[np.float64]
    accuracy: list[float]
    balanced_accuracy: list[float]
    precision: list[float]
    recall: list[float]
    f1_score: list[float]
    false_negative_rate: list[float]
    false_positive_rate: list[float]
    true_negative_rate: list[float]
    true_positive_rate: list[float]

class SupersetFairnessRank(TypedDict):
    """Result of a superset fairness ranking."""
    sensitive: SensitiveGroup
    result: RankResult

class SupersetPerformanceEvaluation(TypedDict):
    """Result of a superset performance evaluation."""
    sensitive: SensitiveGroup
    result: list[DetailedPerformanceMetricsResult] 