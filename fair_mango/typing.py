from typing import TypedDict
import pandas as pd


class DatasetTargetResult(TypedDict):
    """Result of target data for a sensitive group from Dataset."""
    sensitive: list[str]
    result: pd.Series  

class MetricResult(TypedDict):
    """Result of a metric for a single sensitive group."""
    sensitive: list[str] 
    result: float 

class DisparityResult(TypedDict):
    """Result of disparity calculation between two groups."""
    group_1: list[str]
    group_2: list[str]
    disparity: float

class FairnessSummaryResult(TypedDict):
    """Summary of a difference-based fairness metric."""
    disparity: float
    privileged_group: list[str] | None  
    unprivileged_group: list[str] | None 

class FairnessRatioSummaryResult(TypedDict):
    """Summary of a ratio-based fairness metric."""
    ratio: float
    privileged_group: list[str] | None 
    unprivileged_group: list[str] | None 

class RankResult(TypedDict):
    """Individual rank result for a group."""
    sensitive: list[str]  # Changed from tuple to list[str]
    score: float

class DetailedPerformanceMetricsResult(TypedDict):
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

class SupersetFairnessRank(TypedDict):
    """Result of a superset fairness ranking."""
    sensitive_attributes: list[str]
    rank: list[RankResult] 

class SupersetPerformanceEvaluation(TypedDict):
    """Result of a superset performance evaluation."""
    sensitive_attributes: list[str]  
    results: list[DetailedPerformanceMetricsResult]  