from abc import ABC, abstractmethod
from collections.abc import Hashable
from itertools import combinations
from typing import Generic, Literal, TypeVar

import numpy as np
import pandas as pd

from fair_mango.dataset.dataset import Dataset
from fair_mango.typing import (
    DisparityResultDict,
    FairnessSummaryDifferenceResult,
    FairnessSummaryDifferenceFairResult,
    FairnessSummaryRatioResult,
    FairnessSummaryRatioFairResult,
    BaseMetricResult,
    CombinedPerformanceResult,
    RankResult,
    SensitiveGroupTupleT,
    SensitiveGroupOptionalT,
)

LabelT = TypeVar("LabelT", bound=str)


def is_binary(y: pd.Series) -> bool:
    """Check if a data contains binary values (0/1) or can be treated as binary.

    Parameters
    ----------
    y : pd.Series
        Input data.

    Returns
    -------
    bool
        True if data contains binary values or can be treated as binary, False otherwise.
    """
    return y.nunique() <= 2


def encode_target(data: Dataset, col: str | Hashable) -> None:
    """Encode target as [0,1]

    Parameters
    ----------
    data : Dataset
        Dataset object.
    col : str | Hashable
        Column name.

    Raises
    ------
    ValueError
        If the positive target parameter was not provided when creating the
        dataset.
    KeyError
        If the positive target value does not exist in the column.
    """
    if data.positive_target is None:
        raise ValueError(
            f"Calculations failed because target '{col}' has values different "
            "than [0,1]. Provide the positive_target parameter when creating "
            "the dataset to solve this issue."
        )

    unique_values = data.df[col].unique()
    if data.positive_target in unique_values:
        mapping = {data.positive_target: 1}
        data.df[col] = data.df[col].map(mapping).fillna(0).astype(int)
    else:
        raise KeyError(
            "Positive target value provided does not exist in the column. "
            f"{data.positive_target} does not exist in column {col}: "
            f"{unique_values}"
        )


def false_negative_rate(fn: int, tp: int, **_) -> float:
    """Calculate false negative rate.

    Parameters
    ----------
    fn : int
        Number of false negatives from the confusion matrix.
    tp : int
        Number of true positives from the confusion matrix.

    Returns
    -------
    float
        False negative rate value. Returns NaN if denominator is zero.
    """
    denominator = fn + tp
    if denominator == 0:
        return float("nan")
    return fn / denominator


def false_positive_rate(tn: int, fp: int, **_) -> float:
    """Calculate false positive rate.

    Parameters
    ----------
    tn : int
        Number of true negatives from the confusion matrix.
    fp : int
        Number of false positives from the confusion matrix.

    Returns
    -------
    float
        False positive rate value. Returns NaN if denominator is zero.
    """
    denominator = fp + tn
    if denominator == 0:
        return float("nan")
    return fp / denominator


def true_negative_rate(tn: int, fp: int, **_) -> float:
    """Calculate true negative rate.

    Parameters
    ----------
    tn : int
        Number of true negatives from the confusion matrix.
    fp : int
        Number of false positives from the confusion matrix.

    Returns
    -------
    float
        True negative rate value. Returns NaN if denominator is zero.
    """
    denominator = tn + fp
    if denominator == 0:
        return float("nan")
    return tn / denominator


def true_positive_rate(fn: int, tp: int, **_) -> float:
    """Calculate true positive rate.

    Parameters
    ----------
    fn : int
        Number of false negatives from the confusion matrix.
    tp : int
        Number of true positives from the confusion matrix.

    Returns
    -------
    float
        True positive rate value. Returns NaN if denominator is zero.
    """
    denominator = fn + tp
    if denominator == 0:
        return float("nan")
    return tp / denominator


class Metric(ABC):
    """An abstract class that is inherited by every class that measures some
    metric for the different sensitive groups present in the sensitive
    feature.

    Parameters
    ----------
    data : Dataset
        Input data.

    Raises
    ------
    ValueError
        - If data is a DataFrame and the parameters 'sensitive_group' and 'real_target'
          are not provided.
        - If the target variable is not binary (has two unique values).
    """

    def __init__(self, data: Dataset) -> None:
        self.data = data
        self.predicted_target_by_group = []
        y: pd.Series = self.data.df[self.data.real_target]

        if is_binary(y):
            if (np.unique(y) != [0, 1]).all():
                encode_target(self.data, y.name)
            self.real_target_by_group = self.data.get_real_target_for_all_groups()

            if self.data.predicted_target is not None:
                y = self.data.df[self.data.predicted_target]
                if is_binary(y) and (np.unique(y) != [0, 1]).all():
                    encode_target(self.data, y.name)
                self.predicted_target_by_group = (
                    self.data.get_predicted_target_for_all_groups()
                )
        else:
            raise (
                ValueError(
                    f"target variable needs to be binary. Found {y.nunique()}"
                    " unique values"
                )
            )

    @abstractmethod
    def __call__(self):
        pass


def calculate_disparity(
    result_per_groups: list[BaseMetricResult],
    method: Literal["difference", "ratio"],
) -> list[DisparityResultDict]:
    """Calculate the disparity in the scores between every possible pair in
    the provided groups using two available methods:
    - difference (Example: for three groups a, b, c:
      `[score_a - score_b], [score_a - score_c], [score_b - score_c]`).
    - ratio (Example: for three groups a, b, c:
      `[score_a / score_b], [score_a / score_c], [score_b / score_c]`).

    Parameters
    ----------
    result_per_groups : list[BaseMetricResult],
        List of metric result objects with the sensitive group and the corresponding
        score data.
    method : Literal["difference", "ratio"]
        Method used to calculate the disparity. Either 'difference' or 'ratio'.

    Returns
    -------
    list[dict[str, list[str] | float]]
        A list of dictionaries, each containing:
        - "group_1": list of sensitive group labels for first group
        - "group_2": list of sensitive group labels for second group
        - "disparity": float value of the calculated disparity

    Raises
    ------
    AttributeError
        If method is not 'difference' or 'ratio'.
    """

    disparities: list[DisparityResultDict] = []
    for i, j in combinations(range(len(result_per_groups)), 2):
        rec_i, rec_j = result_per_groups[i], result_per_groups[j]

        grp_i = rec_i.sensitive_group
        grp_j = rec_j.sensitive_group

        def _to_float(result) -> float:
            """Convert metric result to a single float value."""
            if isinstance(result, CombinedPerformanceResult):
                return float(result.selection_rate_in_predictions)

            if isinstance(result, BaseMetricResult):
                data = result.data
            else:
                raise ValueError(
                    f"Unsupported result type: {type(result)}. Expected BaseMetricResult or CombinedPerformanceResult."
                )

            if isinstance(data, dict):
                first_key = next(iter(data))
                values = data[first_key]
                return float(np.mean(values) if isinstance(values, list) else values)
            elif isinstance(data, (pd.Series, list, np.ndarray)):
                return float(data[0] if len(data) == 1 else np.mean(data))
            else:
                return float(data)

        a, b = _to_float(rec_i), _to_float(rec_j)
        if method == "difference":
            disp = a - b
        else:
            if b == 0:
                disp = 1.0 if a == 0 else float("inf")
            else:
                disp = a / b

        disparities.append(
            {
                "group_1": grp_i,
                "group_2": grp_j,
                "disparity": disp,
            }
        )

    return disparities


class FairnessMetricDifference(ABC, Generic[LabelT]):
    """An abstract class that is inherited by every fairness metric that is
    based on the 'difference' to calculate disparity between the sensitive
    groups present in the sensitive feature.

    Parameters
    ----------
    data : Dataset
        Input data.
    metric : type[Metric]
        A sequence of metrics or a dictionary with keys being custom labels
        and values a callable that calculates the score.
    label : str
        The key to give to the result in the different returned dictionaries.
    metric_type : str, optional
        Whether the metric measures performance or error. Either 'performance'
        or 'error', by default 'performance'.

    Raises
    ------
    ValueError
        If data is a DataFrame and the parameters 'sensitive_group' and 'real_target'
        are not provided.
    AttributeError
        If metric_type is not 'performance' or 'error'.
    """

    def __init__(
        self,
        data: Dataset,
        metric: type[Metric],
        label: LabelT,
        metric_type: str = "performance",
        **metric_kwargs,
    ) -> None:
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.label = label
        self.metric_results: list = []
        self.metric_type = metric_type
        self.data = data
        if metric_type == "performance":
            self.label1 = "privileged"
            self.label2 = "unprivileged"
        elif metric_type == "error":
            self.label1 = "unprivileged"
            self.label2 = "privileged"
        else:
            raise AttributeError(
                "Metric type not recognized. accepted values 'performance' or 'error'"
            )

        self.result: dict
        self.ranking: dict | None = None
        self.results: list[DisparityResultDict] | None = None

    def _compute(self) -> list[DisparityResultDict]:
        """Calculate the disparity in the scores between every possible pair in
        the provided groups.

        Returns
        -------
        list[DisparityResultDict]
            A list of DisparityResultDict with:
            - group_1: First group as list[str]
            - group_2: Second group as list[str]
            - disparity: The difference value between groups
        """

        metric = self.metric(self.data, **self.metric_kwargs)
        metric_result = metric()

        self.metric_results = metric_result

        results = calculate_disparity(self.metric_results, "difference")

        return results

    def summary(
        self,
    ) -> FairnessSummaryDifferenceResult | FairnessSummaryDifferenceFairResult:
        """Return the fairness metric value, in other words the biggest
        disparity found with specifying the privileged and discriminated
        groups.

        Returns
        -------
        FairnessSummaryDifferenceResult | FairnessSummaryDifferenceFairResult
            A single summary result with:
            - difference: The maximum absolute difference value found.
            - privileged_sensitive_group: List of strings identifying the privileged group.
                None if no group could be determined (e.g., if there is only one group
                or all groups have identical scores).
            - unprivileged_sensitive_group: List of strings identifying the unprivileged group.
                None if no group could be determined (e.g., if there is only one group
                or all groups have identical scores).
        """
        if self.results is None:
            self.results = self._compute()

        max_disparity = 0.0
        privileged_sensitive_group: SensitiveGroupOptionalT = None
        unprivileged_sensitive_group: SensitiveGroupOptionalT = None

        for disparity_result in self.results:
            abs_disparity = abs(disparity_result["disparity"])

            if abs_disparity > max_disparity:
                max_disparity = abs_disparity
                if disparity_result["disparity"] > 0:
                    privileged_sensitive_group = disparity_result["group_1"]
                    unprivileged_sensitive_group = disparity_result["group_2"]
                else:
                    privileged_sensitive_group = disparity_result["group_2"]
                    unprivileged_sensitive_group = disparity_result["group_1"]

        if max_disparity == 0:
            return FairnessSummaryDifferenceFairResult(
                difference=0,
                privileged_sensitive_group=None,
                unprivileged_sensitive_group=None,
            )
        else:
            # At this point, both groups must be non-None since max_disparity > 0
            assert privileged_sensitive_group is not None
            assert unprivileged_sensitive_group is not None
            return FairnessSummaryDifferenceResult(
                difference=max_disparity,
                privileged_sensitive_group=privileged_sensitive_group,
                unprivileged_sensitive_group=unprivileged_sensitive_group,
            )

    def rank(self) -> list[RankResult]:
        """Assign a score to every sensitive group present in the sensitive
        features and rank them from most privileged to most discriminated.
        The score can be interpreted like:
        - ['Male': 0.0314]: Males have on average a score higher by 3.14% than
          the Females.
        - ['White': -0.0628]: Whites have on average a score lower by 6.28% than
          other groups (Black, Asian...).

        Returns
        -------
        list[RankResult]:
            List of RankResult objects with sensitive_group and score attributes.
        """
        if self.results is None:
            self.results = self._compute()

        group_disparities: dict[tuple, list[float]] = {}

        for disparity_result in self.results:
            group_1 = disparity_result["group_1"]
            group_2 = disparity_result["group_2"]
            difference = disparity_result["disparity"]

            group_1_tuple: SensitiveGroupTupleT = tuple(group_1)
            group_2_tuple: SensitiveGroupTupleT = tuple(group_2)

            if group_1_tuple not in group_disparities:
                group_disparities[group_1_tuple] = []
            group_disparities[group_1_tuple].append(difference)

            if group_2_tuple not in group_disparities:
                group_disparities[group_2_tuple] = []
            group_disparities[group_2_tuple].append(-difference)

        ranking_list = []
        for group_tuple, disparities in group_disparities.items():
            average_score = sum(disparities) / len(disparities)
            ranking_list.append(
                RankResult(sensitive_group=list(group_tuple), score=average_score)
            )

        ranking_list.sort(
            key=lambda x: float(x.score)
            if isinstance(x.score, (int, float, str))
            else 0.0,
            reverse=True,
        )

        return ranking_list

    def is_biased(self, threshold: float = 0.1) -> bool:
        """Return a decision of whether there is bias or not for each target
        depending on the provided threshold.

        Parameters
        ----------
        threshold : float, optional
            The threshold to make the decision of whether there is bias or not,
            by default 0.1.

        Returns
        -------
        bool
            Boolean bias indicator.

        Raises
        ------
        ValueError
            If threshold parameter is not in the range of [0, 1].
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative for difference metrics.")

        if self.results is None:
            self.results = self._compute()

        is_biased_result = any(
            abs(disparity_result["disparity"]) > threshold
            for disparity_result in self.results
        )

        return is_biased_result


class FairnessMetricRatio(ABC, Generic[LabelT]):
    """An abstract class that is inherited by every fairness metric that is
    based on the 'ratio' to calculate disparity between the sensitive groups
    present in the sensitive feature.

    Parameters
    ----------
    data : Dataset
        Input data.
    metric : type[Metric]
        A sequence of metrics or a dictionary with keys being custom labels
        and values a callable that calculates the score.
    label : str
        The key to give to the result in the different returned dictionaries.
    metric_type : str, optional
        Whether the metric measures performance or error. Either 'performance'
        or 'error', by default 'performance'.

    Raises
    ------
    ValueError
        If data is a DataFrame and the parameters 'sensitive_group' and 'real_target'
        are not provided.
    AttributeError
        If metric_type is not 'performance' or 'error'.
    """

    def __init__(
        self,
        data: Dataset,
        metric: type[Metric],
        label: LabelT,
        metric_type: str = "performance",
        **metric_kwargs,
    ) -> None:
        self.data = data
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.label = label

        if metric_type == "performance":
            self.label1, self.label2 = "privileged", "unprivileged"
        elif metric_type == "error":
            self.label1, self.label2 = "unprivileged", "privileged"
        else:
            raise AttributeError(
                "metric_type must be 'performance' or 'error', got %s" % metric_type
            )

        self.metric_type = metric_type
        self.metric_results: list = []
        self.result: dict | None = None
        self.ranking: dict | None = None
        self.results: list[DisparityResultDict] | None = None

    def _compute(self) -> list[DisparityResultDict]:
        """Calculate the disparity in the scores between every possible pair in
        the provided groups.

        Returns
        -------
        list[DisparityResultDict]
            A list of DisparityResultDict with:
            - group_1: First group as list[str]
            - group_2: Second group as list[str]
            - disparity: The ratio value between groups
        """

        metric = self.metric(self.data, **self.metric_kwargs)
        metric_result = metric()

        self.metric_results = metric_result

        results = calculate_disparity(self.metric_results, "ratio")

        return results

    def summary(
        self,
    ) -> FairnessSummaryRatioResult | FairnessSummaryRatioFairResult:
        """Return the fairness metric value, in other words the biggest
            disparity found with specifying the privileged and discriminated
            groups.

            Returns
            -------
            FairnessSummaryRatioResult | FairnessSummaryRatioFairResult
                A single summary result with:
        ratio : float
            The minimum ratio value found (closest to 0).
        privileged_sensitive_group : SensitiveGroupT | None
            List of values identifying the privileged group. None if no group could be determined
            (e.g., if there is only one group or all groups have identical scores).
        unprivileged_sensitive_group : SensitiveGroupT | None
            List of values identifying the unprivileged group. None if no group could be determined
            (e.g., if there is only one group or all groups have identical scores).
        """
        if self.results is None:
            self.results = self._compute()

        min_ratio = 1.0
        privileged_sensitive_group: SensitiveGroupOptionalT = None
        unprivileged_sensitive_group: SensitiveGroupOptionalT = None

        for disparity_result in self.results:
            ratio_value = disparity_result["disparity"]

            if ratio_value > 1:
                adjusted_ratio = 1 / ratio_value
                temp_privileged = disparity_result["group_1"]
                temp_unprivileged = disparity_result["group_2"]
            else:
                adjusted_ratio = ratio_value
                temp_privileged = disparity_result["group_2"]
                temp_unprivileged = disparity_result["group_1"]

            if adjusted_ratio < min_ratio:
                min_ratio = adjusted_ratio
                privileged_sensitive_group = temp_privileged
                unprivileged_sensitive_group = temp_unprivileged

        if min_ratio == 1:
            return FairnessSummaryRatioFairResult(
                ratio=1,
                privileged_sensitive_group=None,
                unprivileged_sensitive_group=None,
            )
        else:
            # At this point, both groups must be non-None since min_ratio < 1
            assert privileged_sensitive_group is not None
            assert unprivileged_sensitive_group is not None
            return FairnessSummaryRatioResult(
                ratio=min_ratio,
                privileged_sensitive_group=privileged_sensitive_group,
                unprivileged_sensitive_group=unprivileged_sensitive_group,
            )

    def rank(self) -> list[RankResult]:
        """Assign a score to every sensitive group present in the sensitive
        features and rank them from most privileged to most discriminated.
        The score can be interpreted like:
        - {"sensitive_group": ["Male"], "score": 0.814}: Males have on average 81.4% the score of the
          Females.
        - {"sensitive_group": ["White"], "score": 1.20}: Whites have on average 120% the score of the
          other groups (Black, Asian...).

        Returns
        -------
        list[RankResult]:
            List of RankResult objects with sensitive_group and score attributes.
        """
        if self.results is None:
            self.results = self._compute()

        group_ratios: dict[tuple, list[float]] = {}

        for disparity_result in self.results:
            group_1 = disparity_result["group_1"]
            group_2 = disparity_result["group_2"]
            ratio = disparity_result["disparity"]

            group_1_tuple: SensitiveGroupTupleT = tuple(group_1)
            group_2_tuple: SensitiveGroupTupleT = tuple(group_2)

            if group_1_tuple not in group_ratios:
                group_ratios[group_1_tuple] = []
            group_ratios[group_1_tuple].append(ratio)

            if group_2_tuple not in group_ratios:
                group_ratios[group_2_tuple] = []
            inverse_ratio = 1.0 / ratio if ratio != 0 else float("inf")
            group_ratios[group_2_tuple].append(inverse_ratio)

        ranking_list = []
        for group_tuple, ratios in group_ratios.items():
            finite_ratios = [r for r in ratios if r != float("inf")]
            if finite_ratios:
                average_score = sum(finite_ratios) / len(finite_ratios)
            else:
                average_score = float("inf")
            ranking_list.append(
                RankResult(
                    sensitive_group=list(group_tuple), score=float(average_score)
                )
            )

        ranking_list.sort(
            key=lambda x: float(x.score)
            if isinstance(x.score, (int, float, str)) and x.score != float("inf")
            else 0.0
        )

        return ranking_list

    def is_biased(self, threshold: float = 0.8) -> bool:
        """Return a decision of whether there is bias or not for each target
        depending on the provided threshold.

        Parameters
        ----------
        threshold : float, optional
            The threshold to make the decision of whether there is bias or not,
            by default 0.8.

        Returns
        -------
        bool
            Boolean bias indicator.

        Raises
        ------
        ValueError
            If threshold parameter is not in the range of [0, 1].
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in range [0, 1]")

        if self.results is None:
            self.results = self._compute()

        is_biased_result = any(
            disparity_result["disparity"] < threshold
            for disparity_result in self.results
        )

        return is_biased_result
