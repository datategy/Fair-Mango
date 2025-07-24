from dataclasses import asdict, dataclass
from typing import Protocol, TypeAlias, TypedDict, Union, cast

import numpy as np
import pandas as pd


SensitiveGroupT: TypeAlias = Union[list[str], list[int], list[bool], list[str | int | bool]]


@dataclass
class DatasetTargetResult:
    """Result of target data for a sensitive group from Dataset."""

    sensitive_group: SensitiveGroupT
    data: pd.Series

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return {
            "sensitive_group": self.sensitive_group,
            "data": self.data.tolist() if hasattr(self.data, "tolist") else self.data,
        }


@dataclass
class GroupRankingResult:
    """Result of group ranking."""

    sensitive_group: SensitiveGroupT
    score: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return {"sensitive_group": self.sensitive_group, "score": self.score}


@dataclass
class DatasetGroupResult:
    """Result of group data from Dataset."""

    sensitive_group: SensitiveGroupT
    data: pd.DataFrame

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return {
            "sensitive_group": self.sensitive_group,
            "data": self.data.to_dict() if hasattr(self.data, "to_dict") else self.data,
        }


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
    privileged_sensitive_group: SensitiveGroupT | None
    unprivileged_sensitive_group: SensitiveGroupT | None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        result: dict[str, object] = {"disparity": self.disparity}
        if self.privileged_sensitive_group is not None:
            result["privileged_sensitive_group"] = self.privileged_sensitive_group
        if self.unprivileged_sensitive_group is not None:
            result["unprivileged_sensitive_group"] = self.unprivileged_sensitive_group
        return result


@dataclass
class FairnessRatioSummaryResult:
    """Summary of a ratio-based fairness metric."""

    ratio: float
    privileged_sensitive_group: SensitiveGroupT | None
    unprivileged_sensitive_group: SensitiveGroupT | None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        result: dict[str, object] = {"ratio": self.ratio}
        if self.privileged_sensitive_group is not None:
            result["privileged_sensitive_group"] = self.privileged_sensitive_group
        if self.unprivileged_sensitive_group is not None:
            result["unprivileged_sensitive_group"] = self.unprivileged_sensitive_group
        return result


@dataclass
class RankResult:
    """Individual rank result for a group."""

    sensitive_group: SensitiveGroupT
    score: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return asdict(self)


class MetricResultProtocol(Protocol):
    """Protocol for metric result types that can be used in disparity calculations."""

    sensitive_group: SensitiveGroupT
    data: float | pd.Series | np.ndarray | list[float]


@dataclass
class SelectionRateResult:
    """Result of selection rate for a single sensitive group."""

    sensitive_group: SensitiveGroupT
    data: float

    def __getitem__(self, key):
        """Allow dictionary-style access for backward compatibility."""
        if key == "sensitive_group":
            return np.array(self.sensitive_group)
        elif key == "data":
            return self.data
        elif isinstance(key, str) and hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found")

    def __setitem__(self, key: str, value):
        """Allow dictionary-style assignment for backward compatibility."""
        if key == "data":
            self.data = value
        elif hasattr(self, key):
            setattr(self, key, value)
        else:
            setattr(self, key, value)

    def pop(self, key: str, default=None):
        """Remove and return a value like dict.pop()."""
        if key == "data":
            value = self.data
            self.data = default
            return value
        elif hasattr(self, key):
            value = getattr(self, key)
            delattr(self, key)
            return value
        return default

    def __contains__(self, key) -> bool:
        """Check if key exists for backward compatibility."""
        if key in ["sensitive_group", "data"]:
            return True
        if isinstance(key, str):
            return hasattr(self, key)
        return False

    def update(self, other):
        """Update this object with values from another object or dict."""
        if hasattr(other, "metrics"):
            # Handle PerformanceMetricResult
            for key, value in other.metrics.items():
                setattr(self, key, value)
        elif isinstance(other, dict):
            for key, value in other.items():
                if key != "sensitive_group":
                    setattr(self, key, value)
        else:
            # Handle other dataclass objects
            for key, value in other.__dict__.items():
                if key != "sensitive_group":
                    setattr(self, key, value)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return {"sensitive_group": np.array(self.sensitive_group), "data": self.data}


@dataclass
class PerformanceMetricResult:
    """Result of performance metrics for a single sensitive group with dynamic metrics."""

    sensitive_group: SensitiveGroupT
    metrics: dict[str, list[float]]

    def __getattr__(self, name: str):
        """Allow attribute access to metric values."""
        if name in self.metrics:
            return self.metrics[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __getitem__(self, key: str):
        """Dictionary-style access for backward compatibility."""
        if key == "sensitive_group":
            return np.array(self.sensitive_group)
        elif key in self.metrics:
            return self.metrics[key]
        else:
            raise KeyError(f"Key '{key}' not found")

    def __contains__(self, key: str) -> bool:
        """Check if key exists for backward compatibility."""
        return key == "sensitive_group" or key in self.metrics

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        result: dict[str, object] = {"sensitive_group": np.array(self.sensitive_group)}
        result.update(cast(dict[str, object], self.metrics))
        return result


@dataclass
class ConfusionMatrixResult:
    """Result of confusion matrix metrics for a single sensitive group with dynamic metrics."""

    sensitive_group: SensitiveGroupT
    metrics: dict[str, list[float]]

    def __getattr__(self, name: str):
        """Allow attribute access to metric values."""
        if name in self.metrics:
            return self.metrics[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __getitem__(self, key: str):
        """Dictionary-style access for backward compatibility."""
        if key == "sensitive_group":
            return np.array(self.sensitive_group)
        elif key in self.metrics:
            return self.metrics[key]
        else:
            raise KeyError(f"Key '{key}' not found")

    def __contains__(self, key: str) -> bool:
        """Check if key exists for backward compatibility."""
        return key == "sensitive_group" or key in self.metrics

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        result: dict[str, object] = {"sensitive_group": np.array(self.sensitive_group)}
        result.update(cast(dict[str, object], self.metrics))
        return result


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

    def to_dict(self) -> list[dict[str, object]]:
        """Convert to dictionary format for backward compatibility."""
        return [rank.to_dict() for rank in self.rankings]


@dataclass
class DemographicParitySummaryResult:
    """Summary result for Demographic Parity metrics."""

    demographic_parity_difference: float | None = None
    demographic_parity_ratio: float | None = None
    privileged_sensitive_group: SensitiveGroupT | None = None
    unprivileged_sensitive_group: SensitiveGroupT | None = None
    label: str = "demographic_parity_difference"

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        result: dict[str, object] = {}
        if self.demographic_parity_difference is not None:
            result[self.label] = self.demographic_parity_difference
        elif self.demographic_parity_ratio is not None:
            result[self.label] = self.demographic_parity_ratio
        if self.privileged_sensitive_group is not None:
            result["privileged_sensitive_group"] = self.privileged_sensitive_group
        if self.unprivileged_sensitive_group is not None:
            result["unprivileged_sensitive_group"] = self.unprivileged_sensitive_group
        return result


@dataclass
class DisparateImpactSummaryResult:
    """Summary result for Disparate Impact metrics."""

    disparate_impact_difference: float | None = None
    disparate_impact_ratio: float | None = None
    privileged_sensitive_group: SensitiveGroupT | None = None
    unprivileged_sensitive_group: SensitiveGroupT | None = None
    label: str = "disparate_impact_difference"

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        result: dict[str, object] = {}
        if self.disparate_impact_difference is not None:
            result[self.label] = self.disparate_impact_difference
        elif self.disparate_impact_ratio is not None:
            result[self.label] = self.disparate_impact_ratio
        if self.privileged_sensitive_group is not None:
            result["privileged_sensitive_group"] = self.privileged_sensitive_group
        if self.unprivileged_sensitive_group is not None:
            result["unprivileged_sensitive_group"] = self.unprivileged_sensitive_group
        return result


@dataclass
class EqualOpportunitySummaryResult:
    """Summary result for Equal Opportunity metrics."""

    equal_opportunity_difference: float | None = None
    equal_opportunity_ratio: float | None = None
    privileged_sensitive_group: SensitiveGroupT | None = None
    unprivileged_sensitive_group: SensitiveGroupT | None = None
    label: str = "equal_opportunity_difference"

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        result: dict[str, object] = {}
        if self.equal_opportunity_difference is not None:
            result[self.label] = self.equal_opportunity_difference
        elif self.equal_opportunity_ratio is not None:
            result[self.label] = self.equal_opportunity_ratio
        if self.privileged_sensitive_group is not None:
            result["privileged_sensitive_group"] = self.privileged_sensitive_group
        if self.unprivileged_sensitive_group is not None:
            result["unprivileged_sensitive_group"] = self.unprivileged_sensitive_group
        return result


@dataclass
class FalsePositiveRateSummaryResult:
    """Summary result for False Positive Rate metrics."""

    false_positive_rate_difference: float | None = None
    false_positive_rate_ratio: float | None = None
    privileged_sensitive_group: SensitiveGroupT | None = None
    unprivileged_sensitive_group: SensitiveGroupT | None = None
    label: str = "false_positive_rate_difference"

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        result: dict[str, object] = {}
        if self.false_positive_rate_difference is not None:
            result[self.label] = self.false_positive_rate_difference
        elif self.false_positive_rate_ratio is not None:
            result[self.label] = self.false_positive_rate_ratio
        if self.privileged_sensitive_group is not None:
            result["privileged_sensitive_group"] = self.privileged_sensitive_group
        if self.unprivileged_sensitive_group is not None:
            result["unprivileged_sensitive_group"] = self.unprivileged_sensitive_group
        return result


@dataclass
class EqualisedOddsSummaryResult:
    """Summary result for Equalised Odds metrics."""

    equalised_odds_difference: float | None = None
    equalised_odds_ratio: float | None = None
    privileged_sensitive_group: SensitiveGroupT | None = None
    unprivileged_sensitive_group: SensitiveGroupT | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        result: dict[str, object] = {}
        if self.equalised_odds_difference is not None:
            result["equalised_odds_difference"] = self.equalised_odds_difference
        if self.equalised_odds_ratio is not None:
            result["equalised_odds_ratio"] = self.equalised_odds_ratio
        if self.privileged_sensitive_group is not None:
            result["privileged_sensitive_group"] = self.privileged_sensitive_group
        if self.unprivileged_sensitive_group is not None:
            result["unprivileged_sensitive_group"] = self.unprivileged_sensitive_group
        return result


@dataclass
class SupersetFairnessRankingResult:
    """Result of fairness metric rankings for a sensitive group combination."""

    sensitive_group: SensitiveGroupT
    rankings: dict[str, list[RankResult]]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return {
            "sensitive_group": self.sensitive_group,
            "rankings": {k: [r.to_dict() for r in v] for k, v in self.rankings.items()},
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

    sensitive_group: SensitiveGroupT
    bias_results: dict[str, bool]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for backward compatibility."""
        return {
            "sensitive_group": self.sensitive_group,
            "bias_results": self.bias_results,
        }


@dataclass
class SupersetPerformanceMetricsResult:
    """Result container for superset performance metrics."""

    sensitive_group: tuple[str, ...]
    data: list[SelectionRateResult]

    def __getitem__(self, key: str):
        """Dictionary-style access for backward compatibility."""
        if key == "sensitive_group":
            return self.sensitive_group
        elif key == "data":
            return self.data
        else:
            raise KeyError(f"Key '{key}' not found")

    def __contains__(self, key) -> bool:
        """Check if key exists for backward compatibility."""
        if key in ["sensitive_group", "data"]:
            return True
        if isinstance(key, str):
            return hasattr(self, key)
        return False

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary format for backward compatibility."""
        return {"sensitive_group": self.sensitive_group, "data": self.data}


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
