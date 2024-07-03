from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from fair_mango.dataset.dataset import Dataset


def is_binary(y: pd.Series | pd.DataFrame) -> bool:
    """Check if a data contains two unique values

    Parameters
    ----------
    y : pd.Series | pd.DataFrame
        data

    Returns
    -------
    bool
        true if data contains two unique values else false
    """
    try:
        if y.nunique() == 2:
            return True
        else:
            return False
    except ValueError:
        if (y.nunique() == 2).all():
            return True
        else:
            return False


def encode_target(data: Dataset, ind: int, col: str):
    """encode targets to [0,1]

    Parameters
    ----------
    data : Dataset
        dataset
    ind : int
        index of the positive target
    col : str
        column name

    Raises
    ------
    ValueError
        if the positive target parameter was not provided when creating the
        dataset
    """
    if data.positive_target is None:
        raise ValueError(
            f"Calculations failed because target '{col}' has values different "
            "than [0,1]. Provide the positive_target parameter when creating "
            "the dataset to solve this issue."
        )
    else:
        if data.positive_target[ind] in data.df[col].unique():
            mapping = {data.positive_target[ind]: 1}
            data.df[col] = data.df[col].map(mapping).fillna(0).astype(int)
        else:
            raise KeyError(
                "Positive target value provided does not exist in the column. "
                f"{data.positive_target[ind]} does not exist in column {col}: "
                f"{data.df[col].unique()}"
            )


def false_negative_rate(fn: int, tp: int, **kwargs) -> float:
    """calculate false negative rate

    Parameters
    ----------
    fn : int
        number of false negatives from the confusion matrix
    tp : int
        number of true positives from the confusion matrix

    Returns
    -------
    float
        result
    """
    return fn / (fn + tp)


def false_positive_rate(tn: int, fp: int, **kwargs) -> float:
    """calculate false positive rate

    Parameters
    ----------
    tn : int
        number of true negatives from the confusion matrix
    fp : int
        number of false positives from the confusion matrix

    Returns
    -------
    float
        result
    """
    return fp / (fp + tn)


def true_negative_rate(tn: int, fp: int, **kwargs) -> float:
    """calculate true negative rate

    Parameters
    ----------
    tn : int
        number of true negatives from the confusion matrix
    fp : int
        number of false positives from the confusion matrix

    Returns
    -------
    float
        result
    """
    return tn / (tn + fp)


def true_positive_rate(fn: int, tp: int, **kwargs) -> float:
    """calculate true positive rate

    Parameters
    ----------
    fn : int
        number of false negatives from the confusion matrix
    tp : int
        number of true positives from the confusion matrix

    Returns
    -------
    float
        result
    """
    return tp / (fn + tp)


class Metric(ABC):
    """A class for the different fairness metrics and performance evaluation
    in different groups"""

    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        data : Dataset | pd.DataFrame
            data to evaluate
        sensitive : Sequence[str]
            list of sensitive attributes (Ex: gender, race...), by default None
        real_target : Sequence[str]
            list of column names of actual labels for target variables, by
            default None
        predicted_target : Sequence[str], optional
            list of column names of predicted labels for target variables, by
            default None
        positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
            list of the positive labels corresponding to the provided targets,
            by default None

        Raises
        ------
        ValueError
            if sensitive and real_target are not provided
        """
        if isinstance(data, Dataset):
            self.data = data
        else:
            if sensitive is None or real_target is None:
                raise ValueError(
                    "When providing a DataFrame, 'sensitive' and 'real_target'"
                    " must be specified."
                )
            self.data = Dataset(
                data, sensitive, real_target, predicted_target, positive_target
            )

        self.predicted_targets_by_group = []
        y = self.data.df[self.data.real_target]
        if len(self.data.real_target) > 1:
            y = y.squeeze()
        if is_binary(y):
            for ind, col in enumerate(y):
                if (np.unique(y[col]) != [0, 1]).all():
                    encode_target(self.data, ind, col)
            self.real_targets_by_group = self.data.get_real_target_for_all_groups()
            if self.data.predicted_target != []:
                self.predicted_targets_by_group = (
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
    def __call__(self): ...


def difference(result_per_groups: np.ndarray) -> dict[tuple, np.ndarray[float]]:
    result = {}
    pairs = combinations(range(len(result_per_groups)), 2)

    for i, j in pairs:
        group_i = tuple(result_per_groups[i]["sensitive"])
        group_j = tuple(result_per_groups[j]["sensitive"])
        result_i = np.array(result_per_groups[i]["result"])
        result_j = np.array(result_per_groups[j]["result"])

        key = (group_i, group_j)
        result[key] = result_i - result_j

    return result


def ratio(result_per_groups: np.ndarray) -> dict[tuple, np.ndarray[float]]:
    result = {}
    pairs = list(combinations(range(len(result_per_groups)), 2))

    for i, j in pairs:
        result_i = np.array(result_per_groups[i]["result"])
        result_j = np.array(result_per_groups[j]["result"])
        group_i = tuple(result_per_groups[i]["sensitive"])
        group_j = tuple(result_per_groups[j]["sensitive"])
        key = (group_i, group_j)

        result[key] = result_i / result_j

    return result


class FairnessMetricDifference(ABC):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        metric: type[Metric],
        label: str,
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
        positive_target: Sequence[int | float | str | bool] | None = None,
        metric_type: str = "performance",
        **kwargs,
    ) -> None:
        if isinstance(data, Dataset):
            self.data = data
        else:
            if sensitive is None or real_target is None:
                raise ValueError(
                    "When providing a DataFrame, 'sensitive' and 'real_target' must be specified."
                )
            self.data = Dataset(
                data, sensitive, real_target, predicted_target, positive_target
            )

        self.metric = metric
        self.label = label
        self.kwargs = kwargs
        self.targets: Sequence
        self.metric_results: list
        self.metric_type = metric_type

        if metric_type == "performance":
            self.label1 = "privileged"
            self.label2 = "unprivileged"
        elif metric_type == "error":
            self.label1 = "unprivileged"
            self.label2 = "privileged"
        else:
            raise ValueError(
                "Metric type not recognized. accepted values 'performance' or 'error'"
            )

        self.result: dict | None = None
        self.ranking: dict | None = None
        self.results: dict | None = None

    def _compute(self) -> dict:
        metric = self.metric(self.data, **self.kwargs)
        self.targets, self.metric_results = metric()
        results = difference(self.metric_results)
        return results

    def summary(self) -> dict:
        if self.results is None:
            self.results = self._compute()
        self.differences = self.targets, self.results
        self.result = {}

        for target in self.targets:
            self.result[target] = {
                self.label: 0.0,
                "privileged": None,
                "unprivileged": None,
            }

        for key, value in self.results.items():
            if isinstance(value, (float, int)):
                value = [value]
            for ind, target in enumerate(self.targets):
                if np.abs(value[ind]) > self.result[target][self.label]:
                    self.result[target][self.label] = np.abs(value[ind])
                    if value[ind] > 0:
                        self.result[target][self.label1] = key[0]
                        self.result[target][self.label2] = key[1]
                    else:
                        self.result[target][self.label1] = key[1]
                        self.result[target][self.label2] = key[0]

        return self.result

    def rank(self) -> dict:
        result: dict = {}
        self.ranking = {}

        for target in self.data.real_target:
            result.setdefault(target, {})
            self.ranking.setdefault(target, {})

        if self.results is None:
            self.results = self._compute()

        for key, values in self.results.items():
            if isinstance(values, float):
                values = [values]
            for target, value in zip(self.data.real_target, values):
                if self.metric_type == "performance":
                    result[target].setdefault(key[0], []).append(value)
                    result[target].setdefault(key[1], []).append(-value)
                elif self.metric_type == "error":
                    result[target].setdefault(key[0], []).append(-value)
                    result[target].setdefault(key[1], []).append(value)

        for target, target_result in result.items():
            for group, differences in target_result.items():
                difference = np.mean(np.array(differences))
                self.ranking[target].setdefault(group, difference)
            self.ranking[target] = dict(
                sorted(
                    self.ranking[target].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

        return self.ranking

    def is_biased(self, threshold: float = 0.1) -> dict:
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in range [0, 1]")

        if self.ranking is None:
            self.ranking = self.rank()

        bias: dict = {}

        for target, dicts in self.ranking.items():
            max_diff, min_diff = list(dicts.values())[0], list(dicts.values())[-1]
            if max_diff > threshold or min_diff < -threshold:
                bias[target] = True
            else:
                bias[target] = False

        return bias


class FairnessMetricRatio(ABC):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        metric: type[Metric],
        label: str,
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
        positive_target: Sequence[int | float | str | bool] | None = None,
        metric_type: str = "performance",
        **kwargs,
    ) -> None:
        if isinstance(data, Dataset):
            self.data = data
        else:
            if sensitive is None or real_target is None:
                raise ValueError(
                    "When providing a DataFrame, 'sensitive' and 'real_target' must be specified."
                )
            self.data = Dataset(
                data, sensitive, real_target, predicted_target, positive_target
            )
        self.metric = metric

        if metric_type == "performance":
            self.label1 = "privileged"
            self.label2 = "unprivileged"
        elif metric_type == "error":
            self.label1 = "unprivileged"
            self.label2 = "privileged"

        self.kwargs = kwargs
        self.label = label
        self.metric_type = metric_type
        self.targets: Sequence
        self.metric_results: list
        self.result: dict | None = None
        self.ranking: dict | None = None
        self.results: dict | None = None

    def _compute(self) -> dict:
        metric = self.metric(self.data, **self.kwargs)
        self.targets, self.metric_results = metric()
        results = ratio(self.metric_results)
        return results

    def summary(self) -> dict:
        if self.results is None:
            self.results = self._compute()

        self.ratios = self.targets, self.results
        self.result = {}

        for target in self.targets:
            self.result[target] = {
                self.label: 1.0,
                "privileged": None,
                "unprivileged": None,
            }

        for key, value in self.results.items():
            if isinstance(value, (float, int)):
                value = [value]
            for ind, target in enumerate(self.targets):
                if value[ind] > 1:
                    temp = 1 / value[ind]
                    key = list(key)
                    key[0], key[1] = key[1], key[0]
                else:
                    temp = value[ind]

                if temp < self.result[target][self.label]:
                    self.result[target][self.label] = temp
                    self.result[target][self.label1] = key[1]
                    self.result[target][self.label2] = key[0]

        return self.result

    def rank(self) -> dict:
        result: dict = {}
        self.ranking = {}

        for target in self.data.real_target:
            result.setdefault(target, {})
            self.ranking.setdefault(target, {})

        if self.results is None:
            self.results = self._compute()

        for key, values in self.results.items():
            if isinstance(values, float):
                values = [values]
            for target, value in zip(self.data.real_target, values):
                if self.metric_type == "performance":
                    result[target].setdefault(key[0], []).append(1 / value)
                    result[target].setdefault(key[1], []).append(value)
                elif self.metric_type == "error":
                    result[target].setdefault(key[0], []).append(value)
                    result[target].setdefault(key[1], []).append(1 / value)

        for target, target_result in result.items():
            for group, ratios in target_result.items():
                ratio = np.mean(np.array(ratios))
                self.ranking[target].setdefault(group, ratio)

            self.ranking[target] = dict(
                sorted(
                    self.ranking[target].items(),
                    key=lambda item: item[1],
                    reverse=False,
                )
            )

        return self.ranking

    def is_biased(self, threshold: float = 0.8) -> dict:
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in range [0, 1]")

        if self.ranking is None:
            self.ranking = self.rank()

        bias: dict = {}

        for target, dicts in self.ranking.items():
            min_ratio, max_ratio = list(dicts.values())[0], list(dicts.values())[-1]
            if max_ratio > (1 / threshold) or min_ratio < threshold:
                bias[target] = True
            else:
                bias[target] = False

        return bias
