from collections.abc import Collection, Sequence

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
from fair_mango.metrics.base import (
    FairnessMetricDifference,
    FairnessMetricRatio,
    Metric,
    false_negative_rate,
    false_positive_rate,
    true_negative_rate,
    true_positive_rate,
)


class SelectionRate(Metric):
    """Calculate selection rate for different sensitive groups"""

    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        use_y_true: bool = False,
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
        positive_target: Sequence[int | float | str | bool] | None = None,
    ):
        super().__init__(
            data, sensitive, real_target, predicted_target, positive_target
        )
        self.use_y_true = use_y_true

    def __call__(self) -> tuple[Sequence[str], list[dict]]:
        self.results = []
        if self.use_y_true:
            targets = self.data.real_target
            targets_by_group = self.real_targets_by_group
        else:
            if self.predicted_targets_by_group == []:
                raise ValueError(
                    "No predictions found, provide predicted_target parameter "
                    "when creating the dataset or set use_y_true to True to "
                    "use the real labels"
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

    def all_data(self) -> pd.Series:
        """Compute selection rate for all the dataset

        Returns
        -------
        pd.Series
            the target name and the corresponding selection rate
        """
        if self.use_y_true:
            return self.data.df[self.data.real_target].mean()
        else:
            return self.data.df[self.data.predicted_target].mean()


class ConfusionMatrix(Metric):
    """Calculate confusion matrix related metrics:
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
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        super().__init__(
            data, sensitive, real_target, predicted_target, positive_target
        )
        if self.predicted_targets_by_group == []:
            raise ValueError(
                "No predictions found, provide predicted_target parameter "
                "when creating the dataset"
            )
        self.zero_division = zero_division
        if metrics is None:
            self.metrics = {
                "false_negative_rate": false_negative_rate,  # type: ignore[dict-item]
                "false_positive_rate": false_positive_rate,  # type: ignore[dict-item]
                "true_negative_rate": true_negative_rate,  # type: ignore[dict-item]
                "true_positive_rate": true_positive_rate,  # type: ignore[dict-item]
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

                conf_matrix = confusion_matrix(
                    real_values, predicted_values, labels=[0, 1]
                )
                tn = conf_matrix[0, 0]
                tp = conf_matrix[1, 1]
                fn = conf_matrix[1, 0]
                fp = conf_matrix[0, 1]

                for metric_name, metric in self.metrics.items():
                    result_for_group.setdefault(metric_name, []).append(
                        metric(
                            tn=tn, fp=fp, fn=fn, tp=tp, zero_division=self.zero_division  # type: ignore[call-arg]
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
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        super().__init__(
            data, sensitive, real_target, predicted_target, positive_target
        )

        if self.predicted_targets_by_group == []:
            raise ValueError(
                "No predictions found, provide predicted_target parameter "
                "when creating the dataset"
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


class DemographicParityDifference(FairnessMetricDifference):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "demographic_parity_difference",
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
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
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
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
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
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
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
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


class DemographicParityRatio(FairnessMetricRatio):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "demographic_parity_ratio",
        zero_division: float | str | None = None,
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
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
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
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
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
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
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
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
                    "When providing a DataFrame, 'sensitive' and 'real_target'"
                    " must be specified."
                )

            self.data = Dataset(
                data, sensitive, real_target, predicted_target, positive_target
            )

        self.label = "equalised_odds_difference"
        self.ranking: dict | None = None
        self.tpr: dict | None = None
        self.fpr: dict | None = None

    def _compute(self) -> tuple[dict, dict]:
        tpr = EqualOpportunityDifference(self.data)
        fpr = FalsePositiveRateDifference(self.data)
        tpr.summary()
        fpr.summary()
        tpr_diff = tpr.differences[1]
        fpr_diff = fpr.differences[1]

        return tpr_diff, fpr_diff

    def summary(self) -> dict:
        self.result: dict = {}

        for target in self.data.real_target:
            self.result.setdefault(
                target, {self.label: 0.0, "privileged": None, "unprivileged": None}
            )

        if (self.tpr is None) or (self.fpr is None):
            self.tpr, self.fpr = self._compute()

        for (key1, values1), (_, values2) in zip(self.tpr.items(), self.fpr.items()):
            for target, value1, value2 in zip(self.data.real_target, values1, values2):
                if np.abs(value1) > self.result[target][self.label]:
                    self.result[target][self.label] = np.abs(value1)
                    if value1 > 0:
                        self.result[target]["privileged"] = key1[0]
                        self.result[target]["unprivileged"] = key1[1]
                    else:
                        self.result[target]["privileged"] = key1[1]
                        self.result[target]["unprivileged"] = key1[0]
                if np.abs(value2) > self.result[target][self.label]:
                    self.result[target][self.label] = np.abs(value2)
                    if value2 > 0:
                        self.result[target]["privileged"] = key1[1]
                        self.result[target]["unprivileged"] = key1[0]
                    else:
                        self.result[target]["privileged"] = key1[0]
                        self.result[target]["unprivileged"] = key1[1]

        return self.result

    def rank(self) -> dict:
        result: dict = {}
        self.ranking = {}

        for target in self.data.real_target:
            result.setdefault(target, {})
            self.ranking.setdefault(target, {})

        if (self.tpr is None) or (self.fpr is None):
            self.tpr, self.fpr = self._compute()

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


class EqualisedOddsRatio:
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        zero_division_: float | str | None = None,
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

        self.label = "equalised_odds_ratio"
        self.zero_division = zero_division_
        self.ranking: dict | None = None
        self.tpr: dict | None = None
        self.fpr: dict | None = None

    def _compute(self) -> tuple[dict, dict]:
        tpr = EqualOpportunityRatio(self.data, zero_division_=self.zero_division)
        fpr = FalsePositiveRateRatio(self.data, zero_division_=self.zero_division)
        tpr.summary()
        fpr.summary()
        tpr_ratio = tpr.ratios[1]
        fpr_ratio = fpr.ratios[1]

        return tpr_ratio, fpr_ratio

    def summary(self) -> dict:
        self.result: dict = {}

        for target in self.data.real_target:
            self.result.setdefault(
                target, {self.label: 1.0, "privileged": None, "unprivileged": None}
            )

        if (self.tpr is None) or (self.fpr is None):
            self.tpr, self.fpr = self._compute()

        for (key1, values1), (_, values2) in zip(self.tpr.items(), self.fpr.items()):
            for target, value1, value2 in zip(self.data.real_target, values1, values2):
                if value1 > 1:
                    temp = 1 / value1
                else:
                    temp = value1

                if temp < self.result[target][self.label]:
                    self.result[target][self.label] = temp
                    if value1 > 1:
                        self.result[target]["privileged"] = key1[0]
                        self.result[target]["unprivileged"] = key1[1]
                    else:
                        self.result[target]["privileged"] = key1[1]
                        self.result[target]["unprivileged"] = key1[0]

                if value2 > 1:
                    temp = 1 / value2
                else:
                    temp = value2

                if temp < self.result[target][self.label]:
                    self.result[target][self.label] = temp
                    if value2 > 1:
                        self.result[target]["privileged"] = key1[1]
                        self.result[target]["unprivileged"] = key1[0]
                    else:
                        self.result[target]["privileged"] = key1[0]
                        self.result[target]["unprivileged"] = key1[1]

        return self.result

    def rank(self) -> dict:
        result: dict = {}
        self.ranking = {}

        for target in self.data.real_target:
            result.setdefault(target, {})
            self.ranking.setdefault(target, {})

        if (self.tpr is None) or (self.fpr is None):
            self.tpr, self.fpr = self._compute()

        for (key1, values1), (_, values2) in zip(self.tpr.items(), self.fpr.items()):
            for target, value1, value2 in zip(self.data.real_target, values1, values2):
                if value1 > 1:
                    temp1 = 1 / value1
                else:
                    temp1 = value1

                if value2 > 1:
                    temp2 = 1 / value2
                else:
                    temp2 = value2

                if temp1 < temp2:
                    result[target].setdefault(key1[0], []).append(value1)
                    result[target].setdefault(key1[1], []).append(1 / value1)
                else:
                    result[target].setdefault(key1[0], []).append(value2)
                    result[target].setdefault(key1[1], []).append(1 / value2)

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

    def is_biased(self, threshold: float = 0.1) -> dict:
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
