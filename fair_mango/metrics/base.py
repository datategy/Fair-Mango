from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fair_mango.dataset.dataset import Dataset


def is_binary(y: pd.Series) -> bool:
    """Check if a data contains two unique values.

    Parameters
    ----------
    y : pd.Series
        Input data.

    Returns
    -------
    bool
        True if data contains two unique values else False.
    """
    if y.nunique() == 2:
        return True
    else:
        return False


def encode_target(data: Dataset, col: str | Hashable) -> None:
    """encode target as [0,1]

    Parameters
    ----------
    data : Dataset
        Dataset object.
    ind : int
        Index of the positive target.
    col : str
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
    else:
        if data.positive_target in data.df[col].unique():
            mapping = {data.positive_target: 1}
            data.df[col] = data.df[col].map(mapping).fillna(0).astype(int)
        else:
            raise KeyError(
                "Positive target value provided does not exist in the column. "
                f"{data.positive_target} does not exist in column {col}: "
                f"{data.df[col].unique()}"
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
        False negative rate value.
    """
    return fn / (fn + tp)


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
        False positive rate value.
    """
    return fp / (fp + tn)


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
        True negative rate value.
    """
    return tn / (tn + fp)


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
        True positive rate value.
    """
    return tp / (fn + tp)


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
        - If data is a DataFrame and the parameters 'sensitive' and 'real_target'
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
    result_per_groups: list[dict], method: Literal["difference", "ratio"]
) -> dict[tuple, NDArray[np.float64]]:
    """Calculate the disparity in the scores between every possible pair in
    the provided groups using two available methods:
    - difference (Example: for three groups a, b, c:
      `[score_a - score_b], [score_a - score_c], [score_b - score_c]`).
    - ratio (Example: for three groups a, b, c:
      `[score_a / score_b], [score_a / score_c], [score_b / score_c]`).

    Parameters
    ----------
    result_per_groups : list[dict]
        List of dictionaries with the sensitive group and the corresponding
        score.
    method : Literal['difference', 'ratio']
        Method used to calculate the disparity. Either 'difference' or 'ratio'.

    Returns
    -------
    dict[tuple, np.ndarray[float]]
        A dictionary with:
        - keys: tuple with the pair of the sensitive groups labels.
        - values: a numpy array with the corresponding disparity.

    Raises
    ------
    AttributeError
        If method is not 'difference' or 'ratio'.
    """
    result = {}
    pairs = combinations(range(len(result_per_groups)), 2)

    for i, j in pairs:
        group_i = tuple(result_per_groups[i]["sensitive"])
        group_j = tuple(result_per_groups[j]["sensitive"])

        
        key_i = 'data' if 'data' in result_per_groups[i] else 'result'
        key_j = 'data' if 'data' in result_per_groups[j] else 'result'
        
        data_i = result_per_groups[i][key_i]
        data_j = result_per_groups[j][key_j]
        
        if isinstance(data_i, list) or (
            isinstance(data_i, np.ndarray) and data_i.ndim == 1
        ):
            result_i = np.array(data_i)
            result_j = np.array(data_j)
        else:
            result_i = np.array([data_i])
            result_j = np.array([data_j])

        key = (group_i, group_j)

        if method == "difference":
            result[key] = result_i - result_j
        elif method == "ratio":
            result[key] = result_i / result_j
        else:
            raise AttributeError(
                f"method {method} not recognised. Use 'difference' or 'ratio' instead."
            )

    return result


class FairnessMetricDifference(ABC):
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
        If data is a DataFrame and the parameters 'sensitive' and 'real_target'
        are not provided.
    AttributeError
        If metric_type is not 'performance' or 'error'.
    """

    def __init__(
        self,
        data: Dataset,
        metric: type[Metric],
        metric_type: str = "performance",
        **kwargs,
    ) -> None:
        self.metric = metric
        self.kwargs = kwargs
        self.target: str
        self.metric_results: list
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

        self.result: dict | None = None
        self.ranking: dict | None = None
        self.results: dict | None = None

    def _compute(self) -> dict[tuple, NDArray[np.float64]]:
        """Calculate the disparity in the scores between every possible pair in
        the provided groups.

        Returns
        -------
        dict[tuple, np.ndarray[float]]
            A dictionary with:
            - keys: tuple with the pair of the sensitive groups labels.
            - values: a numpy array with the corresponding disparity.
        """
       
        filtered_kwargs = {k: v for k, v in self.kwargs.items() if k != 'label'}
        metric = self.metric(self.data, **filtered_kwargs)
        self.target, self.metric_results = metric()
        results = calculate_disparity(self.metric_results, "difference")
        
       
        self.differences = (self.target, results)
        
        return results

    def summary(self) -> dict[str, dict[str, float | tuple | None]]:
        """Return the fairness metric value, in other words the biggest
        disparity found with specifying the privileged and discriminated
        groups.

        Returns
        -------
        dict[str, dict[str, float | tuple | None]]
            A dictionary with:
            - keys: name of the target variable.
            - values: a dictionary corresponding to the results for that target
              variable with:
                1. keys: labels for the biggest disparity, the privileged group
                   and the discriminated group.
                2. values: values for the biggest disparity, the privileged
                   group and the discriminated group.
        """
        if self.results is None:
            self.results = self._compute()
        
        disparity = 0.0
        privileged_group = None
        unprivileged_group = None

        for key, value in self.results.items():
            if np.abs(value[0]) > disparity:
                disparity = float(np.abs(value[0]))  # Convert to scalar float
                if value[0] > 0:
                    privileged_group = key[0]
                    unprivileged_group = key[1]
                else:
                    privileged_group = key[1]
                    unprivileged_group = key[0]

       
        target_name = self.target
        label = self.kwargs.get('label', getattr(self, 'label', 'disparity'))
        
        
        if privileged_group is not None:
            privileged_group = tuple(str(x) for x in privileged_group)
        if unprivileged_group is not None:
            unprivileged_group = tuple(str(x) for x in unprivileged_group)
        
       
        return {
            target_name: {
                label: disparity,
                "privileged": privileged_group,
                "unprivileged": unprivileged_group,
            }
        }

    def rank(self) -> dict[str, dict[tuple, float]]:
        """Assign a score to every sensitive group present in the sensitive
        features and rank them from most privileged to most discriminated.
        The score can be interpreted like:
        - [('Male',): 0.0314]: Males have on average a score higher by 3.14% than
          the Females.
        - [('White',): -0.0628]: Whites have on average a score lower by 6.28% than
          other groups (Black, Asian...).

        Returns
        -------
        dict[str, dict[tuple, float]]
            A dictionary with:
            - keys: name of the target variable.
            - values: a dictionary corresponding to the ranking for that target
              variable with:
                1. keys: a tuple with the sensitive group.
                2. values: the corresponding score.
        """
        result: dict = {}
        ranking: dict[tuple, float] = {}

        if self.results is None:
            self.results = self._compute()

        for key, values in self.results.items():
            for value in values:
                if self.metric_type == "performance":
                    result.setdefault(key[0], []).append(value)
                    result.setdefault(key[1], []).append(-value)
                elif self.metric_type == "error":
                    result.setdefault(key[0], []).append(-value)
                    result.setdefault(key[1], []).append(value)
        
        for group, differences in result.items():
            difference = np.mean(np.array(differences))
            ranking[group] = difference
        
        ranking = dict(
            sorted(
                ranking.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

       
        ranking_str = {}
        for group, score in ranking.items():
            group_str = tuple(str(x) for x in group)
            ranking_str[group_str] = score

       
        return {self.data.real_target: ranking_str}

    def is_biased(self, threshold: float = 0.1) -> dict[str, bool]:
        """Return a decision of whether there is bias or not
        depending on the provided threshold.

        Parameters
        ----------
        threshold : float, optional
            The threshold to make the decision of whether there is bias or not,
            by default 0.1.

        Returns
        -------
        dict[str, bool]
            Dictionary with target name as key and bias decision as value.

        Raises
        ------
        ValueError
            If threshold parameter is not in the range of [0, 1].
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in range [0, 1]")

        ranking_nested = self.rank()
        
        if not ranking_nested:
            return {self.data.real_target: False}

        
        ranking = list(ranking_nested.values())[0]
        
        if not ranking:
            return {self.target: False}

        max_diff, min_diff = list(ranking.values())[0], list(ranking.values())[-1]
        is_biased = max_diff > threshold or min_diff < -threshold
        
        
        return {self.data.real_target: is_biased}


class FairnessMetricRatio(ABC):
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
        If data is a DataFrame and the parameters 'sensitive' and 'real_target'
        are not provided.
    AttributeError
        If metric_type is not 'performance' or 'error'.
    """

    def __init__(
        self,
        data: Dataset,
        metric: type[Metric],
        metric_type: str = "performance",
        **kwargs,
    ) -> None:
        self.data = data
        self.metric = metric
        self.label = kwargs.pop('label', 'difference') 

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

        self.kwargs = kwargs
        self.metric_type = metric_type
        self.target: Sequence
        self.metric_results: list
        self.result: dict | None = None
        self.ranking: dict | None = None
        self.results: dict | None = None
        self.ratios: tuple | None = None  
        self.differences: tuple | None = None  

    def _compute(self) -> dict[tuple, NDArray[np.float64]]:
        """Calculate the disparity in the scores between every possible pair in
        the provided groups.

        Returns
        -------
        dict[tuple, np.ndarray[float]]
            A dictionary with:
            - keys: tuple with the pair of the sensitive groups labels.
            - values: a numpy array with the corresponding disparity.
        """
        
        filtered_kwargs = {k: v for k, v in self.kwargs.items() if k != 'label'}
        metric = self.metric(self.data, **filtered_kwargs)
        self.target, self.metric_results = metric()
        results = calculate_disparity(self.metric_results, "ratio")
        
       
        self.ratios = (self.target, results)
        
        return results

    def summary(self) -> dict[str, dict[str, float | tuple | None]]:
        """Return the fairness metric value, in other words the biggest
        disparity found with specifying the privileged and discriminated
        groups.

        Returns
        -------
        dict[str, dict[str, float | tuple | None]]
            A dictionary with:
            - keys: name of the target variable.
            - values: a dictionary corresponding to the results for that target
              variable with:
                1. keys: labels for the biggest disparity, the privileged group
                   and the discriminated group.
                2. values: values for the biggest disparity, the privileged
                   group and the discriminated group.
        """
        if self.results is None:
            self.results = self._compute()

        ratio = 1.0
        privileged_group = None
        unprivileged_group = None

        for key, value in self.results.items():
            if value[0] > 1:
                temp = 1 / value[0]
                key = list(key)
                key[0], key[1] = key[1], key[0]
            else:
                temp = value[0]

            if temp < ratio:
                ratio = float(temp)  # Convert to scalar float
                privileged_group = key[1]
                unprivileged_group = key[0]

       
        target_name = self.target
        label = self.kwargs.get('label', getattr(self, 'label', 'ratio'))
        
        
        if privileged_group is not None:
            privileged_group = tuple(str(x) for x in privileged_group)
        if unprivileged_group is not None:
            unprivileged_group = tuple(str(x) for x in unprivileged_group)
        
        
        return {
            target_name: {
                label: ratio,
                "privileged": privileged_group,
                "unprivileged": unprivileged_group,
            }
        }

    def rank(self) -> dict[str, dict[tuple, float]]:
        """Assign a score to every sensitive group present in the sensitive
        features and rank them from most privileged to most discriminated.
        The score can be interpreted like:
        - [('Male',): 0.814]: Males have on average 81.4% the score of the
          Females.
        - [('White',): 1.20]: Whites have on average 120% the score of the
          other groups (Black, Asian...).

        Returns
        -------
        dict[str, dict[tuple, float]]
            A dictionary with target name as key and group rankings as values.
        """
        result: dict = {}
        ranking: dict[tuple, float] = {}

        if self.results is None:
            self.results = self._compute()

        for key, values in self.results.items():
            for value in values:
                if self.metric_type == "performance":
                    result.setdefault(key[0], []).append(1 / value)
                    result.setdefault(key[1], []).append(value)
                elif self.metric_type == "error":
                    result.setdefault(key[0], []).append(value)
                    result.setdefault(key[1], []).append(1 / value)
        
        for group, ratios in result.items():
            ratio = np.mean(np.array(ratios))
            ranking[group] = ratio

        ranking = dict(
            sorted(
                ranking.items(),
                key=lambda item: item[1],
                reverse=False,
            )
        )

       
        ranking_str = {}
        for group, score in ranking.items():
            group_str = tuple(str(x) for x in group)
            ranking_str[group_str] = score

        
        return {self.data.real_target: ranking_str}

    def is_biased(self, threshold: float = 0.8) -> dict[str, bool]:
        """Return a decision of whether there is bias or not for each target
        depending on the provided threshold.

        Parameters
        ----------
        threshold : float, optional
            The threshold to make the decision of whether there is bias or not,
            by default 0.8.

        Returns
        -------
        dict[str, bool]
            A dictionary with:
            - keys: a string with the target column name.
            - values: True if there is bias else False.

        Raises
        ------
        ValueError
            If threshold parameter is not in the range of [0, 1].
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in range [0, 1]")

        ranking_nested = self.rank()

        if not ranking_nested:
            return {self.data.real_target: False}

        
        ranking = list(ranking_nested.values())[0]
        
        if not ranking:
            return {self.target: False}

        min_ratio, max_ratio = list(ranking.values())[0], list(ranking.values())[-1]
        is_biased = max_ratio > (1 / threshold) or min_ratio < threshold
        
        
        return {self.data.real_target: is_biased}
