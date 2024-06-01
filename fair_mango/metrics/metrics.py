from collections.abc import Collection, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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


class Metric:
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


class SelectionRate(Metric):
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


class ConfusionMatrix(Metric):
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


class PerformanceMetric(Metric):
    """Calculate
    - accuracy
    - balanced accuracy
    - precision
    - recall
    - f1 score
    """

    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        metrics: Collection | Sequence | None = None,
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
        if metrics is None:
            self.metrics = {
                "accuracy": accuracy_score,
                "balanced accuracy": balanced_accuracy_score,
                "precision": precision_score,
                "recall": recall_score,
                "f1-score": f1_score,
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
                for metric_name, metric in self.metrics.items():
                    result_for_group.setdefault(metric_name, []).append(
                        metric(real_values, predicted_values)
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
        result_i = np.array(result_per_groups[i]["result"])
        result_j = np.array(result_per_groups[j]["result"])
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
        if metric_type == "performance":
            self.label1 = "privileged"
            self.label2 = "unprivileged"
        elif metric_type == "error":
            self.label1 = "unprivileged"
            self.label2 = "privileged"
        else:
            raise ValueError("Metric type not recognized. accepted values 'performance' or 'error'")
        self.result = None
        self.ranking = None

    def summary(self) -> None:
        metric = self.metric(self.data, **self.kwargs)
        self.targets, self.metric_results = metric()
        self.results = difference(self.metric_results)
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
    
    def rank(self, pr_to_unp: bool = True) -> dict:
        if self.result is None:
            self.summary()
        result: dict = {}
        self.ranking: dict = {}
        for target in self.data.real_target:
            result.setdefault(target, {})
            self.ranking.setdefault(target, {})
        for key, values in self.results.items():
            if isinstance(values, float):
                values = [values]
            for target, value in zip(self.data.real_target, values):
                result[target].setdefault(key[0], []).append(value)
                result[target].setdefault(key[1], []).append(-value)
        for target, target_result in result.items():
            for group, differences in target_result.items():
                difference = np.mean(np.array(differences))
                self.ranking[target].setdefault(group, difference)
            if pr_to_unp:
                self.ranking[target] = dict(sorted(self.ranking[target].items(), key=lambda item: item[1], reverse=True))
            else:
                self.ranking[target] = dict(sorted(self.ranking[target].items(), key=lambda item: item[1], reverse=False))
        return self.ranking
    
    def is_biased(self, threshold: float = 0.1) -> dict:
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in range [0, 1]")
        if self.ranking is None:
            self.rank()
        bias: dict = {}
        for target, dicts in self.ranking.items():
            max_diff, min_diff = list(dicts.values())[0], list(dicts.values())[-1]
            if max_diff > threshold or min_diff < -threshold:
                bias[target] = True
            else:
                bias[target] = False
        return bias


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
            "performance",
            **{"use_y_true": True},
        )


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
            "performance",
            **{"use_y_true": False},
        )


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
            "performance",
            **{"metrics": {"result": true_positive_rate}, "zero_division": np.nan},
        )


class FalsePositiveRateDifference(FairnessMetricDifference):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "false_positive_rate_difference",
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
            "error",
            **{"metrics": {"result": false_positive_rate}, "zero_division": np.nan},
        )


class FairnessMetricRatio:
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        metric: type[SelectionRate] | type[ConfusionMatrix],
        label: str,
        zero_division_: float | str | None = None,
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
        metric_type: str = "performance",
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
        if metric_type == "performance":
            self.label1 = "privileged"
            self.label2 = "unprivileged"
        elif metric_type == "error":
            self.label1 = "unprivileged"
            self.label2 = "privileged"
        self.kwargs = kwargs
        self.label = label
        self.zero_division = zero_division_
        self.targets: Sequence
        self.metric_results: list
        self.result = None
        self.ranking = None

    def summary(self) -> None:
        metric = self.metric(self.data, **self.kwargs)
        self.targets, self.metric_results = metric()
        self.results = ratio(self.metric_results, self.zero_division)
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
                    value[ind] = 1 / value[ind]
                    key = list(key)
                    key[0], key[1] = key[1], key[0]
                if value[ind] < self.result[target][self.label]:
                    self.result[target][self.label] = value[ind]
                    if value[ind] > 1:
                        self.result[target][self.label1] = key[0]
                        self.result[target][self.label2] = key[1]
                    else:
                        self.result[target][self.label1] = key[1]
                        self.result[target][self.label2] = key[0]
        return self.result
    
    def rank(self, pr_to_unp: bool = True) -> dict:
        self.pr_to_unp = pr_to_unp
        if self.result is None:
            self.summary()
        result: dict = {}
        self.ranking: dict = {}
        for target in self.data.real_target:
            result.setdefault(target, {})
            self.ranking.setdefault(target, {})
        for key, values in self.results.items():
            if isinstance(values, float):
                values = [values]
            for target, value in zip(self.data.real_target, values):
                result[target].setdefault(key[0], []).append(value)
                result[target].setdefault(key[1], []).append(1/value)
        for target, target_result in result.items():
            for group, differences in target_result.items():
                difference = np.mean(np.array(differences))
                self.ranking[target].setdefault(group, difference)
            if pr_to_unp:
                self.ranking[target] = dict(sorted(self.ranking[target].items(), key=lambda item: item[1], reverse=False))
            else:
                self.ranking[target] = dict(sorted(self.ranking[target].items(), key=lambda item: item[1], reverse=True))
        return self.ranking
    
    def is_biased(self, threshold: float = 0.8) -> dict:
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in range [0, 1]")
        if self.ranking is None:
            self.rank()
        bias: dict = {}
        for target, dicts in self.ranking.items():
            if self.pr_to_unp:
                min_diff, max_diff = list(dicts.values())[0], list(dicts.values())[-1]
            else:
                max_diff, min_diff = list(dicts.values())[0], list(dicts.values())[-1]
            if max_diff > 1/threshold or min_diff < threshold:
                bias[target] = True
            else:
                bias[target] = False
        return bias
    
    
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
            "performance",
            **{"use_y_true": True},
        )


class DisparateImpactRatio(FairnessMetricRatio):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "disparate_impact_ratio",
        zero_division_: float | str | None = None,
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        super().__init__(
            data,
            SelectionRate,
            label,
            zero_division_,
            sensitive,
            real_target,
            predicted_target,
            positive_target,
            "performance",
            **{"use_y_true": False},
        )


class EqualOpportunityRatio(FairnessMetricRatio):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "equal_opportunity_ratio",
        zero_division_: float | str | None = None,
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        super().__init__(
            data,
            ConfusionMatrix,
            label,
            zero_division_,
            sensitive,
            real_target,
            predicted_target,
            positive_target,
            "performance",
            **{"metrics": {"result": true_positive_rate}, "zero_division": np.nan},
        )


class FalsePositiveRateRatio(FairnessMetricRatio):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "false_positive_rate_ratio",
        zero_division_: float | str | None = None,
        sensitive: Sequence[str] = [],
        real_target: Sequence[str] = [],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        super().__init__(
            data,
            ConfusionMatrix,
            label,
            zero_division_,
            sensitive,
            real_target,
            predicted_target,
            positive_target,
            "error",
            **{"metrics": {"result": false_positive_rate}, "zero_division": np.nan},
        )


class EqualisedOddsDifference:
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
                    "When providing a DataFrame, 'sensitive' and 'real_target' must be specified."
                )
            self.data = Dataset(
                data, sensitive, real_target, predicted_target, positive_target
            )
        self.label = 'equalised_odds_difference'
        tpr = EqualOpportunityDifference(self.data)
        tpr.summary()
        fpr = FalsePositiveRateDifference(self.data)
        fpr.summary()
        self.tpr = tpr.differences[1]
        self.fpr = fpr.differences[1]
        self.ranking = None
        
    def summary(self) -> dict:
        self.result: dict = {}
        for target in self.data.real_target:
            self.result.setdefault(target, 
                                   {self.label: 0.0,
                                    'privileged': None,
                                    'unprivileged': None})
        for (key1, values1), (_, values2) in zip(self.tpr.items(), self.fpr.items()):
            for target, value1, value2 in zip(self.data.real_target, values1, values2):
                if np.abs(value1) > self.result[target][self.label]:
                    self.result[target][self.label] = np.abs(value1)
                    if value1 > 0:
                        self.result[target]['privileged'] = key1[0]
                        self.result[target]['unprivileged'] = key1[1]
                    else:
                        self.result[target]['privileged'] = key1[1]
                        self.result[target]['unprivileged'] = key1[0]
                if np.abs(value2) > self.result[target][self.label]:
                    self.result[target][self.label] = np.abs(value2)
                    if value2 > 0:
                        self.result[target]['privileged'] = key1[1]
                        self.result[target]['unprivileged'] = key1[0]
                    else:
                        self.result[target]['privileged'] = key1[0]
                        self.result[target]['unprivileged'] = key1[1]
        return self.result            
        
    def rank(self) -> dict:
        result: dict = {}
        self.ranking: dict = {}
        for target in self.data.real_target:
            result.setdefault(target, {})
            self.ranking.setdefault(target, {})
        for (key1, values1), (_, values2) in zip(self.tpr.items(), self.fpr.items()):
            for target, value1, value2 in zip(self.data.real_target, values1, values2):
                if np.abs(value1) > np.abs(value2):
                    result[target].setdefault(key1[0], []).append(value1)
                    result[target].setdefault(key1[1], []).append(-value1)
                else:
                    result[target].setdefault(key1[0], []).append(-value2)
                    result[target].setdefault(key1[1], []).append(value2)
        for target, target_result in result.items():
            for group, differences in target_result.items():
                difference = np.mean(np.array(differences))
                self.ranking[target].setdefault(group, difference)
            self.ranking[target] = dict(sorted(self.ranking[target].items(), key=lambda item: item[1], reverse=True))
        return self.ranking
    
    def is_biased(self, threshold: float = 0.1) -> dict:
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in range [0, 1]")
        if self.ranking is None:
            self.rank()
        bias: dict = {}
        for target, dicts in self.ranking.items():
            max_diff, min_diff = list(dicts.values())[0], list(dicts.values())[-1]
            if max_diff > threshold or min_diff < -threshold:
                bias[target] = True
            else:
                bias[target] = False
        return bias
