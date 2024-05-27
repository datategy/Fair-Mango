from collections.abc import Collection, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

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
        if the positive target parameter was not provided when creating the dataset
    """
    if data.positive_target is None:
        raise ValueError(
            f"Calculations failed because target '{col}' has values \
different than [0,1]. Provide the positive_target parameter when creating the dataset to solve this issue."
        )
    else:
        if data.positive_target[ind] in data.df[col].unique():
            mapping = {data.positive_target[ind]: 1}
            data.df[col] = data.df[col].map(mapping).fillna(0).astype(int)
        else:
            raise KeyError(
                f"Positive target value provided does not exist in the column. {data.positive_target[ind]} does not exist in column {col}: {data.df[col].unique()}"
            )


def false_negative_rate(
    tn: int, fp: int, fn: int, tp: int, zero_division: float | str | None
) -> float | str | None:
    """calculate false negative rate

    Parameters
    ----------
    fn : int
        number of false negatives from the confusion matrix
    tp : int
        number of true positives from the confusion matrix
    zero_division : float | str | None
        default value in case of zero division

    Returns
    -------
    float | str | None
        result
    """
    try:
        return fn / (fn + tp)
    except ZeroDivisionError:
        return zero_division


def false_positive_rate(
    tn: int, fp: int, fn: int, tp: int, zero_division: float | str | None
) -> float | str | None:
    """calculate false positive rate

    Parameters
    ----------
    tn : int
        number of true negatives from the confusion matrix
    fp : int
        number of false positives from the confusion matrix
    zero_division : float | str | None
        default value in case of zero division

    Returns
    -------
    float | str | None
        result
    """
    try:
        return fp / (fp + tn)
    except ZeroDivisionError:
        return zero_division


def true_negative_rate(
    tn: int, fp: int, fn: int, tp: int, zero_division: float | str | None
) -> float | str | None:
    """calculate true negative rate

    Parameters
    ----------
    tn : int
        number of true negatives from the confusion matrix
    fp : int
        number of false positives from the confusion matrix
    zero_division : float | str | None
        default value in case of zero division

    Returns
    -------
    float | str | None
        result
    """
    try:
        return tn / (tn + fp)
    except ZeroDivisionError:
        return zero_division


def true_positive_rate(
    tn: int, fp: int, fn: int, tp: int, zero_division: float | str | None
) -> float | str | None:
    """calculate true positive rate

    Parameters
    ----------
    fn : int
        number of false negatives from the confusion matrix
    tp : int
        number of true positives from the confusion matrix
    zero_division : float | str | None
        default value in case of zero division

    Returns
    -------
    float | str | None
        result
    """
    try:
        return tp / (fn + tp)
    except ZeroDivisionError:
        return zero_division


class PerformanceMetric:
    """A class for the different fairness metrics and performance evaluation in different groups"""

    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
    ):
        if isinstance(data, Dataset):
            self.data = data
        else:
            if sensitive == [] or real_target == []:
                raise ValueError(
                    "When providing a DataFrame, 'sensitive' and 'real_target' must be specified."
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
            self.results: list = []
        else:
            raise (
                ValueError(
                    f"target variable needs to be binary. Found {y.nunique()} unique values"
                )
            )

    def __call__(self): ...


class SelectionRate(PerformanceMetric):
    """Calculate selection rate for different sensitive groups"""

    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        use_y_true: bool = False,
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
    ):
        super().__init__(
            data, sensitive, real_target, predicted_target, positive_target
        )
        self.use_y_true = use_y_true

    def __call__(self):
        self.results = []
        if self.use_y_true:
            targets = self.data.real_target
            targets_by_group = self.real_targets_by_group
        else:
            if self.predicted_targets_by_group == []:
                raise ValueError(
                    "No predictions found, provide predicted_target parameter when creating the dataset or set use_y_true to True to use the real labels"
                )
            targets = self.data.predicted_target
            targets_by_group = self.predicted_targets_by_group
        for group in targets_by_group:
            group_ = group["sensitive"]
            y_group = group["data"]
            self.results.append(
                {"sensitive": group_, "result": np.array(y_group.mean())}
            )
        return targets, self.results

    def all_data(self):
        if self.use_y_true:
            return self.data.df[self.data.real_target].mean()
        else:
            return self.data.df[self.data.predicted_target].mean()


class ConfusionMatrix(PerformanceMetric):
    """Calculate
    - false positive rate
    - false negative rate
    - true positive rate
    - true negative rate
    """

    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        metrics: Collection | Sequence | None = None,
        zero_division: float | str | None = None,
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        super().__init__(
            data, sensitive, real_target, predicted_target, positive_target
        )
        if self.predicted_targets_by_group == []:
            raise ValueError(
                "No predictions found, provide predicted_target parameter when creating the dataset"
            )
        self.zero_division = zero_division
        if metrics is None:
            self.metrics = {
                "false_negative_rate": false_negative_rate,
                "false_positive_rate": false_positive_rate,
                "true_negative_rate": true_negative_rate,
                "true_positive_rate": true_positive_rate,
            }
        else:
            if isinstance(metrics, dict):
                self.metrics = metrics
            else:
                metrics = set(metrics)
                self.metrics = {}
                for metric in metrics:
                    self.metrics[metric.__name__] = metric

    def __call__(self) -> tuple[Sequence, list]:
        for real_group, predicted_group in zip(
            self.real_targets_by_group, self.predicted_targets_by_group
        ):
            group_ = real_group["sensitive"]
            real_y_group = real_group["data"]
            predicted_y_group = predicted_group["data"]
            result_for_group = {"sensitive": group_}
            for real_col, predicted_col in zip(
                self.data.real_target, self.data.predicted_target
            ):
                if len(self.data.real_target) != 1:
                    real_values = real_y_group[real_col]
                    predicted_values = predicted_y_group[predicted_col]
                else:
                    real_values = real_y_group
                    predicted_values = predicted_y_group
                if (np.unique(real_values) == 1).all() and (
                    np.unique(predicted_values) == 1
                ).all():
                    tp = len(real_values)
                    tn = fp = fn = 0
                elif (np.unique(real_values) == 0).all() and (
                    np.unique(predicted_values) == 0
                ).all():
                    tn = len(real_values)
                    tp = fp = fn = 0
                elif (np.unique(real_values) == 1).all() and (
                    np.unique(predicted_values) == 0
                ).all():
                    fn = len(real_values)
                    fp = tn = tp = 0
                elif (np.unique(real_values) == 0).all() and (
                    np.unique(predicted_values) == 1
                ).all():
                    fp = len(real_values)
                    fn = tn = tp = 0
                else:
                    conf_matrix = confusion_matrix(real_values, predicted_values)
                    tn = conf_matrix[0, 0]
                    tp = conf_matrix[1, 1]
                    fn = conf_matrix[1, 0]
                    fp = conf_matrix[0, 1]
                for metric_name, metric in self.metrics.items():
                    result_for_group.setdefault(metric_name, []).append(
                        metric(
                            tn=tn, fp=fp, fn=fn, tp=tp, zero_division=self.zero_division
                        )
                    )
            self.results.append(result_for_group)
        return self.data.real_target, self.results


def difference(result_per_groups: np.array) -> dict:
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


def ratio(
    result_per_groups: np.array, zero_division: float | str | None = None
) -> dict:
    result = {}
    pairs = list(combinations(range(len(result_per_groups)), 2))
    for i, j in pairs:
        result_i = result_per_groups[i]["result"]
        result_j = result_per_groups[j]["result"]
        group_i = tuple(result_per_groups[i]["sensitive"])
        group_j = tuple(result_per_groups[j]["sensitive"])
        key = (group_i, group_j)

        try:
            result[key] = result_i / result_j
        except ZeroDivisionError:
            result[key] = zero_division
    return result


class FairnessMetricDifference:
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        metric: type[SelectionRate] | type[ConfusionMatrix],
        label: str,
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
        **kwargs,
    ) -> None:
        if isinstance(data, Dataset):
            self.data = data
        else:
            if sensitive == [] or real_target == []:
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

    def call(self) -> None:
        metric = self.metric(self.data, **self.kwargs)
        self.targets, self.metric_results = metric()
        self.result = difference(self.metric_results)
        self.differences = self.targets, self.result
        self.summary = {}
        for target in self.targets:
            self.summary[target] = {
                self.label: 0.0,
                "privileged": None,
                "unprivileged": None,
            }
        for key, value in self.result.items():
            if isinstance(value, (float, int)):
                value = [value]
            for ind, target in enumerate(self.targets):
                if np.abs(value[ind]) > self.summary[target][self.label]:
                    self.summary[target][self.label] = np.abs(value[ind])
                    if value[ind] > 0:
                        self.summary[target]["privileged"] = key[0]
                        self.summary[target]["unprivileged"] = key[1]
                    else:
                        self.summary[target]["privileged"] = key[1]
                        self.summary[target]["unprivileged"] = key[0]

    def mean_differences(self):
        result = {}
        for key, value in self.result.items():
            result.setdefault(key[0], []).append(value)
            result.setdefault(key[1], []).append(-value)
        for key, value in result.items():
            stacked_array = np.stack(value)
            result[key] = np.mean(stacked_array, axis=0)
        results = {}
        for target in self.targets:
            results[target] = {
                f"pr_{self.label}": 0.0,
                "most_privileged": None,
                f"unp_{self.label}": 0.0,
                "most_unprivileged": None,
            }
        for key, value in result.items():
            if isinstance(value, (float, int)):
                value = [value]
            for ind, target in enumerate(self.targets):
                if value[ind] > results[target][f"pr_{self.label}"]:
                    results[target][f"pr_{self.label}"] = value[ind]
                    results[target]["most_privileged"] = key
                if value[ind] < results[target][f"unp_{self.label}"]:
                    results[target][f"unp_{self.label}"] = value[ind]
                    results[target]["most_unprivileged"] = key

        return results


class DemographicParityDifference(FairnessMetricDifference):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "demographic_parity_difference",
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        super().__init__(
            data,
            SelectionRate,
            label,
            sensitive,
            real_target,
            predicted_target,
            positive_target,
            **{"use_y_true": True},
        )
        super().call()


class DisparateImpactDifference(FairnessMetricDifference):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "disparate_impact_difference",
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        super().__init__(
            data,
            SelectionRate,
            label,
            sensitive,
            real_target,
            predicted_target,
            positive_target,
            **{"use_y_true": False},
        )
        super().call()


class EqualOpportunityDifference(FairnessMetricDifference):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "equal_opportunity_difference",
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        super().__init__(
            data,
            ConfusionMatrix,
            label,
            sensitive,
            real_target,
            predicted_target,
            positive_target,
            **{"metrics": {'result': true_positive_rate}},
        )
        super().call()


class FairnessMetricRatio:
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        metric: type[SelectionRate] | type[ConfusionMatrix],
        label: str,
        zero_division: float | str | None = None,
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
        **kwargs,
    ) -> None:
        if isinstance(data, Dataset):
            self.data = data
        else:
            if sensitive == [] or real_target == []:
                raise ValueError(
                    "When providing a DataFrame, 'sensitive' and 'real_target' must be specified."
                )
            self.data = Dataset(
                data, sensitive, real_target, predicted_target, positive_target
            )
        self.metric = metric
        self.kwargs = kwargs
        self.label = label
        self.zero_division = zero_division
        self.targets: Sequence
        self.metric_results: list

    def call(self) -> None:
        metric = self.metric(self.data, **self.kwargs)
        self.targets, self.metric_results = metric()
        self.result = ratio(self.metric_results, self.zero_division)
        self.ratios = self.targets, self.result
        self.summary = {}
        for target in self.targets:
            self.summary[target] = {
                self.label: 1.0,
                "privileged": None,
                "unprivileged": None,
            }
        for key, value in self.result.items():
            if isinstance(value, (float, int)):
                value = [value]
            for ind, target in enumerate(self.targets):
                if value[ind] > 1:
                    value[ind] = 1 / value[ind]
                    key = list(key)
                    key[0], key[1] = key[1], key[0]
                if value[ind] < self.summary[target][self.label]:
                    self.summary[target][self.label] = value[ind]
                    if value[ind] > 1:
                        self.summary[target]["privileged"] = key[0]
                        self.summary[target]["unprivileged"] = key[1]
                    else:
                        self.summary[target]["privileged"] = key[1]
                        self.summary[target]["unprivileged"] = key[0]

    def mean_ratios(self, threshold: float = 0.8) -> dict:
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in range [0, 1]")
        result: dict = {}
        for key, value in self.result.items():
            result.setdefault(key[0], []).append(value)
            result.setdefault(key[1], []).append(1 / value)
        for key, value in result.items():
            stacked_array = np.stack(value)
            result[key] = np.mean(stacked_array, axis=0)
        results: dict = {}
        for target in self.targets:
            results[target] = {
                f"pr_{self.label}": 1.0,
                "most_privileged": None,
                f"unp_{self.label}": 1.0,
                "most_unprivileged": None,
            }
        for key, value in result.items():
            if isinstance(value, (float, int)):
                value = [value]
            for ind, target in enumerate(self.targets):
                if value[ind] < results[target][f"pr_{self.label}"]:
                    results[target][f"pr_{self.label}"] = value[ind]
                    results[target]["most_privileged"] = key
                if value[ind] > results[target][f"unp_{self.label}"]:
                    results[target][f"unp_{self.label}"] = value[ind]
                    results[target]["most_unprivileged"] = key
                if (value[ind] < threshold) or value[ind] > (1 / threshold):
                    results[target]["is_biased"] = True
                else:
                    results[target]["is_biased"] = False

        return results


class DemographicParityRatio(FairnessMetricRatio):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "demographic_parity_ratio",
        zero_division: float | str | None = None,
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        super().__init__(
            data,
            SelectionRate,
            label,
            zero_division,
            sensitive,
            real_target,
            predicted_target,
            positive_target,
            **{"use_y_true": True},
        )
        super().call()


class DisparateImpactRatio(FairnessMetricRatio):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "disparate_impact_ratio",
        zero_division: float | str | None = None,
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        super().__init__(
            data,
            SelectionRate,
            label,
            zero_division,
            sensitive,
            real_target,
            predicted_target,
            positive_target,
            **{"use_y_true": False},
        )
        super().call()
