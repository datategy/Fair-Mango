from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd

from fair_mango.dataset.dataset import Dataset


def is_binary(y: pd.Series | pd.DataFrame) -> bool:
    """Check if a data contains two unique values.

    Parameters
    ----------
    y : pd.Series | pd.DataFrame
        Input data.

    Returns
    -------
    bool
        True if data contains two unique values else False.
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
    """Encode targets as [0,1].

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


def false_positive_rate(tn: int, fp: int, **kwargs) -> float:
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


def true_negative_rate(tn: int, fp: int, **kwargs) -> float:
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


def true_positive_rate(fn: int, tp: int, **kwargs) -> float:
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
    data : Dataset | pd.DataFrame
        Input data.
    sensitive : Sequence[str] | None, optional if data is a Dataset object
        Sequence of column names corresponding to sensitive features
        (Ex: gender, race...), by default None.
    real_target : Sequence[str] | None, optional if data is a Dataset object
        Sequence of column names corresponding to the real targets
        (true labels), by default None.
    predicted_target : Sequence[str] | None, optional
        Sequence of column names corresponding to the predicted targets,
        by default None.
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        Sequence of the positive labels corresponding to the provided targets,
        by default None.

    Raises
    ------
    ValueError
        - If data is a DataFrame and the parameters 'sensitive' and 'real_target'
        are not provided.
        - If the target variable is not binary (has two unique values).
    """

    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
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


def calculate_disparity(
    result_per_groups: np.ndarray, method: Literal["difference", "ratio"]
) -> dict[tuple, np.ndarray[float]]:
    """Calculate the disparity in the scores between every possible pair in
    the provided groups using two available methods:
    - difference (Example: for three groups a, b, c:
    [score_a - score_b], [score_a - score_c], [score_b - score_c]).
    - ratio (Example: for three groups a, b, c:
    [score_a / score_b], [score_a / score_c], [score_b / score_c]).

    Parameters
    ----------
    result_per_groups : np.ndarray
        Array of dictionaries with the sensitive group and the corresponding
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

        if isinstance(result_per_groups[i]["result"], list) or (
            isinstance(result_per_groups[i]["result"], np.ndarray)
            and result_per_groups[i]["result"].ndim == 1
        ):
            result_i = np.array(result_per_groups[i]["result"])
            result_j = np.array(result_per_groups[j]["result"])
        else:
            result_i = np.array([result_per_groups[i]["result"]])
            result_j = np.array([result_per_groups[j]["result"]])

        key = (group_i, group_j)

        if method == "difference":
            result[key] = result_i - result_j
        elif method == "ratio":
            result[key] = result_i / result_j
        else:
            raise AttributeError(
                f"method {method} not recognised. Use 'difference' or "
                "'ratio' instead."
            )

    return result


class FairnessMetricDifference(ABC):
    """An abstract class that is inherited by every fairness metric that is
    based on the 'difference' to calculate disparity between the sensitive
    groups present in the sensitive feature.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    metric : type[Metric]
        A sequence of metrics or a dictionary with keys being custom labels
        and values a callable that calculates the score.
    label : str
        The key to give to the result in the different returned dictionaries.
    sensitive : Sequence[str] | None, optional if data is a Dataset object
        Sequence of column names corresponding to sensitive features
        (Ex: gender, race...), by default None.
    real_target : Sequence[str] | None, optional if data is a Dataset object
        Sequence of column names corresponding to the real targets
        (true labels), by default None.
    predicted_target : Sequence[str] | None, optional
        Sequence of column names corresponding to the predicted targets,
        by default None.
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        Sequence of the positive labels corresponding to the provided targets,
        by default None.
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
                    "When providing a DataFrame, 'sensitive' and 'real_target'"
                    " must be specified."
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
            raise AttributeError(
                "Metric type not recognized. accepted values 'performance' or "
                "'error'"
            )

        self.result: dict | None = None
        self.ranking: dict | None = None
        self.results: dict | None = None

    def _compute(self) -> dict[tuple, np.ndarray[float]]:
        """Calculate the disparity in the scores between every possible pair in
        the provided groups.

        Returns
        -------
        dict[tuple, np.ndarray[float]]
            A dictionary with:
            - keys: tuple with the pair of the sensitive groups labels.
            - values: a numpy array with the corresponding disparity.
        """
        metric = self.metric(self.data, **self.kwargs)
        self.targets, self.metric_results = metric()
        results = calculate_disparity(self.metric_results, "difference")
        return results

    def summary(self) -> dict[str, dict[str, float | tuple | None]]:
        """Return the fairness metric value, in other words the biggest
        disparity found with specifying the priviliged and discriminated
        groups.

        Returns
        -------
        dict[str, dict[str, float | tuple | None]]
        A dictionary with:
        - keys: name of the target variable.
        - values: a dictionary corresponding to the results for that target
        variable with:
            - keys: labels for the biggest disparity, the privileged group
            and the discriminated group.
            - values: values for the biggest disparity, the privileged
            group and the discriminated group.
        """
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

    def rank(self) -> dict[str, dict[tuple[str], float]]:
        """Assign a score to every sensitive group present in the sensitive
        features and rank them from most privileged to most discriminated.
        The score can be interpreted like:
        - ['Male': 0.0314]: Males have on average a score higher by 3.14% than
        the Females.
        - ['White': -0.0628]: Whites have on average a score lower by 6.28% than
        other groups (Black, Asian...).

        Returns
        -------
        dict[str, dict[tuple[str], float]]
        A dictionary with:
        - keys: name of the target variable.
        - values: a dictionary corresponding to the ranking for that target
        variable with:
            - keys: a tuple with the sensitive group.
            - values: the corresponding score.
        """
        result: dict = {}
        self.ranking = {}

        for target in self.data.real_target:
            result.setdefault(target, {})
            self.ranking.setdefault(target, {})

        if self.results is None:
            self.results = self._compute()

        for key, values in self.results.items():
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

    def is_biased(self, threshold: float = 0.1) -> dict[str, bool]:
        """Return a decision of whether there is bias or not for each target
        depending on the provided threshold.

        Parameters
        ----------
        threshold : float, optional
            The threshold to make the decision of whether there is bias or not,
            by default 0.1.

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
    """An abstract class that is inherited by every fairness metric that is
    based on the 'ratio' to calculate disparity between the sensitive groups
    present in the sensitive feature.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    metric : type[Metric]
        A sequence of metrics or a dictionary with keys being custom labels
        and values a callable that calculates the score.
    label : str
        The key to give to the result in the different returned dictionaries.
    sensitive : Sequence[str] | None, optional if data is a Dataset object
        Sequence of column names corresponding to sensitive features
        (Ex: gender, race...), by default None.
    real_target : Sequence[str] | None, optional if data is a Dataset object
        Sequence of column names corresponding to the real targets
        (true labels), by default None.
    predicted_target : Sequence[str] | None, optional
        Sequence of column names corresponding to the predicted targets,
        by default None.
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        Sequence of the positive labels corresponding to the provided targets,
        by default None.
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
        else:
            raise AttributeError(
                "Metric type not recognized. accepted values 'performance' or "
                "'error'"
            )

        self.kwargs = kwargs
        self.label = label
        self.metric_type = metric_type
        self.targets: Sequence
        self.metric_results: list
        self.result: dict | None = None
        self.ranking: dict | None = None
        self.results: dict | None = None

    def _compute(self) -> dict[tuple, np.ndarray[float]]:
        """Calculate the disparity in the scores between every possible pair in
        the provided groups.

        Returns
        -------
        dict[tuple, np.ndarray[float]]
            A dictionary with:
            - keys: tuple with the pair of the sensitive groups labels.
            - values: a numpy array with the corresponding disparity.
        """
        metric = self.metric(self.data, **self.kwargs)
        self.targets, self.metric_results = metric()
        results = calculate_disparity(self.metric_results, "ratio")
        return results

    def summary(self) -> dict[str, dict[str, float | tuple | None]]:
        """Return the fairness metric value, in other words the biggest
        disparity found with specifying the priviliged and discriminated
        groups.

        Returns
        -------
        dict[str, dict[str, float | tuple | None]]
        A dictionary with:
        - keys: name of the target variable.
        - values: a dictionary corresponding to the results for that target
        variable with:
            - keys: labels for the biggest disparity, the privileged group
            and the discriminated group.
            - values: values for the biggest disparity, the privileged
            group and the discriminated group.
        """
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

    def rank(self) -> dict[str, dict[tuple[str], float]]:
        """Assign a score to every sensitive group present in the sensitive
        features and rank them from most privileged to most discriminated.
        The score can be interpreted like:
        - ['Male': 0.814]: Males have on average 81.4% the score of the
        Females.
        - ['White': 1.20]: Whites have on average 120% the score of the
        other groups (Black, Asian...).

        Returns
        -------
        dict[str, dict[tuple[str], float]]
        A dictionary with:
        - keys: name of the target variable.
        - values: a dictionary corresponding to the ranking for that target
        variable with:
            - keys: a tuple with the sensitive group.
            - values: the corresponding score.
        """
        result: dict = {}
        self.ranking = {}

        for target in self.data.real_target:
            result.setdefault(target, {})
            self.ranking.setdefault(target, {})

        if self.results is None:
            self.results = self._compute()

        for key, values in self.results.items():
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
