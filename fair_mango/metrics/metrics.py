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
    """Calculate selection rate for different sensitive groups
    
    Parameters
    ----------
    data : Dataset | pd.DataFrame
        input data
    use_y_true : bool, optional
        if True use the real label else use the predictions, by default False
    sensitive : Sequence[str], optional if data is a Dataset object
        sequence of column names corresponding to sensitive features
        (Ex: gender, race...), by default None
    real_target : Sequence[str], optional if data is a Dataset object
        sequence of column names corresponding to the real targets
        (true labels), by default None
    predicted_target : Sequence[str], optional
        sequence of column names corresponding to the predicted targets,
        by default None
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        sequence of the positive labels corresponding to the provided targets,
        by default None
    """

    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        use_y_true: bool = False,
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
        positive_target: Sequence[int | float | str | bool] | None = None,
        label: str = "result",
    ):
        super().__init__(
            data, sensitive, real_target, predicted_target, positive_target
        )
        self.use_y_true = use_y_true
        self.label = label

    def __call__(self) -> tuple[Sequence[str], list[dict]]:
        """Calculates selection rate for different sensitive groups.

        Returns
        -------
        tuple[Sequence[str], list[dict]]
            a tuple containing two elements:
                targets (Sequence[str]): The target variables used for
                calculation.
                results (list[dict]): A list of dictionaries, where each 
                dictionary has two keys:
                    sensitive: The name of the sensitive group.
                    result: The selection rate for the sensitive group.

        Raises
        ------
        ValueError
            if no predictions are found and `use_y_true` is False.
        
        Examples
        --------
        >>> import pandas as pd
        >>> from fair_mango.metrics.metrics import SelectionRate
        >>> data = {
        ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male'],
        ...     'sensitive_2': ['white', 'white', 'black', 'black', 'black'],
        ...     'real_target_1': [0, 1, 0, 1, 0],
        ...     'real_target_2': ['no', 'yes', 'yes', 'yes', 'no'],
        ...     'predicted_target_1': [0, 1, 1, 0, 0],
        ...     'predicted_target_2': ['no', 'no', 'yes', 'yes', 'yes'],
        ... }
        >>> df = pd.DataFrame(data)
        >>> selection_rate_1 = SelectionRate(
        ...     data=df,
        ...     use_y_true=True,
        ...     sensitive=['sensitive_1'],
        ...     real_target=['real_target_1'],
        ... )
        >>> selection_rate_1()
        (
            ['real_target_1'],
            [
                {
                    'sensitive': array(['male'], dtype=object),
                    'result': array(0.33333333)
                },
                {
                    'sensitive': array(['female'], dtype=object),
                    'result': array(0.5)
                }
            ]
        )
        >>> dataset2 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1', 'sensitive_2'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1'],
        ...     positive_target=[1]
        ... )
        >>> selection_rate_2 = SelectionRate(
        ...     data=dataset2,
        ...     use_y_true=False,
        ... )
        >>> selection_rate_2()
        (
            ['predicted_target_1'],
            [
                {
                    'sensitive': array(['male', 'black'], dtype=object), 
                    'result': array(0.)
                },
                {
                    'sensitive': array(['female', 'black'], dtype=object), 
                    'result': array(1.)
                },
                {
                    'sensitive': array(['female', 'white'], dtype=object),
                    'result': array(1.)
                },
                {
                    'sensitive': array(['male', 'white'], dtype=object),
                    'result': array(0.)
                }
            ]
        )
        >>> dataset3 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_2'],
        ...     real_target=['real_target_1', 'real_target_2'],
        ...     predicted_target=['predicted_target_1', 'predicted_target_2'],
        ...     positive_target=[1, 'yes']
        ... )
        >>> selection_rate_3 = SelectionRate(
        ...     data=dataset3,
        ...     use_y_true=True,
        ... )
        >>> selection_rate_3()
        (
            ['real_target_1', 'real_target_2'],
            [
                {
                    'sensitive': array(['black'], dtype=object),
                    'result': array([0.33333333, 0.66666667])
                },
                {
                    'sensitive': array(['white'], dtype=object),
                    'result': array([0.5, 0.5])
                }
            ]
        )
        """
        results: list = []
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
            results.append({"sensitive": group_, self.label: np.array(y_group.mean())})
        return targets, results

    def all_data(self) -> pd.Series:
        """Compute overall selection rate corresponding to the whole dataset.

        Returns
        -------
        pd.Series
            the target name and the corresponding selection rate.
        
        Examples
        --------
        >>> import pandas as pd
        >>> from fair_mango.metrics.metrics import SelectionRate
        >>> data = {
        ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male'],
        ...     'sensitive_2': ['white', 'white', 'black', 'black', 'black'],
        ...     'real_target_1': [0, 1, 0, 1, 0],
        ...     'real_target_2': ['no', 'yes', 'yes', 'yes', 'no'],
        ...     'predicted_target_1': [0, 1, 1, 0, 0],
        ...     'predicted_target_2': ['no', 'no', 'yes', 'yes', 'yes'],
        ... }
        >>> df = pd.DataFrame(data)
        >>> selection_rate_1 = SelectionRate(
        ...     data=df,
        ...     use_y_true=False,
        ...     sensitive=['sensitive_1'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1']
        ... )
        >>> selection_rate_1.all_data()
        predicted_target_1    0.4
        dtype: float64
        >>> dataset2 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1'],
        ...     real_target=['real_target_1', 'real_target_2'],
        ... )
        >>> selection_rate_2 = SelectionRate(
        ...     data=dataset2,
        ...     use_y_true=True,
        ... )
        >>> selection_rate_2.all_data()
        real_target_1    0.4
        real_target_2    0.6
        dtype: float64
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
    
    Parameters
    ----------
    data : Dataset | pd.DataFrame
        input data
    metrics : Collection | Sequence | None, optional
        a sequence of metrics or a dictionary with keys being custom labels
        and values a callable that takes as input tp, tn, fp, fn which are 
        extracted from the confusion matrix. Available functions in
        fair_mango.metrics.metrics.base are:
        - false_positive_rate()
        - false_negative_rate()
        - true_positive_rate()
        - true_negative_rate()
    sensitive : Sequence[str], optional if data is a Dataset object
        sequence of column names corresponding to sensitive features
        (Ex: gender, race...), by default None
    real_target : Sequence[str], optional if data is a Dataset object
        sequence of column names corresponding to the real targets
        (true labels), by default None
    predicted_target : Sequence[str], optional
        sequence of column names corresponding to the predicted targets,
        by default None
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        sequence of the positive labels corresponding to the provided targets,
        by default None
    
    Raises
        ------
        ValueError
            if the predictions column is not provided.
        KeyError
            if the key of a metric is 'sensitive' which is already reserved
            to the sensitive groups.
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
                "false_negative_rate": false_negative_rate,  # type: ignore[dict-item]
                "false_positive_rate": false_positive_rate,  # type: ignore[dict-item]
                "true_negative_rate": true_negative_rate,  # type: ignore[dict-item]
                "true_positive_rate": true_positive_rate,  # type: ignore[dict-item]
            }
        else:
            if isinstance(metrics, dict):
                if "sensitive" in metrics.keys():
                    raise KeyError(
                        "metric label cannot be 'sensitive'. Change the label " "to fix"
                    )
                self.metrics = metrics
            else:
                metrics = set(metrics)
                self.metrics = {}
                for metric in metrics:
                    if metric.__name__ == "sensitive":
                        raise KeyError(
                            "metric label cannot be 'sensitive'. Rename your "
                            "function or use a dictionary to set a label for it"
                        )
                    self.metrics[metric.__name__] = metric

    def __call__(self) -> tuple[Sequence, list]:
        """Calculate confusion matrix related metrics:
        - false positive rate
        - false negative rate
        - true positive rate
        - true negative rate
        for the different demographic groups present in the sensitive feature.

        Returns
        -------
        tuple[Sequence, list]
            a tuple containing two elements:
                targets (Sequence[str]): The target variables used for
                calculation.
                results (list[dict]): A list of dictionaries, where the keys:
                    sensitive: The name of the sensitive group.
                    label: The corresponding result for the sensitive group.
        
        Examples
        --------
        >>> import pandas as pd
        >>> from fair_mango.metrics.metrics import ConfusionMatrix
        >>> from fair_mango.metrics.base import (
        ... false_positive_rate,
        ... true_negative_rate,
        ... true_positive_rate,
        ... false_negative_rate
        ... )
        >>> data = {
        ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male'],
        ...     'sensitive_2': ['white', 'white', 'black', 'black', 'black'],
        ...     'real_target_1': [0, 1, 0, 1, 0],
        ...     'real_target_2': ['no', 'yes', 'yes', 'yes', 'no'],
        ...     'predicted_target_1': [0, 1, 1, 0, 0],
        ...     'predicted_target_2': ['no', 'no', 'yes', 'yes', 'yes'],
        ... }
        >>> df = pd.DataFrame(data)
        >>> confusion_matrix_1 = ConfusionMartrix(
        ...     data=df,
        ...     sensitive=['sensitive_1'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1]
        ... )
        >>> confusion_matrix_1()
        (
            ['real_target_1'],
            [
                {
                    'sensitive': array(['male'], dtype=object),
                    'false_negative_rate': [1.0],
                    'false_positive_rate': [0.0],
                    'true_negative_rate': [1.0],
                    'true_positive_rate': [0.0]
                },
                {
                    'sensitive': array(['female'], dtype=object),
                    'false_negative_rate': [0.0],
                    'false_positive_rate': [1.0],
                    'true_negative_rate': [0.0],
                    'true_positive_rate': [1.0]
                }
            ]
        )
        >>> dataset2 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1'],
        ...     positive_target=[1]
        ... )
        >>> confusion_matrix_2 = ConfusionMatrix(
        ...     data=dataset2,
        ...     metrics=[true_negative_rate],
        ... )
        >>> confusion_matrix_2()
        (
            ['real_target_1'],
            [
                {
                    'sensitive': array(['male'], dtype=object), 
                    'true_negative_rate': [1.0]
                },
                {
                    'sensitive': array(['female'], dtype=object),
                    'true_negative_rate': [0.0]
                }
            ]
        )
        >>> confusion_matrix_3 = ConfusionMatrix(
        ...     data=dataset2,
        ...     metrics={
        ...         'tpr': true_positive_rate,
        ...         'tnr': true_negative_rate
        ...     }
        ... )
        >>> confusion_matrix_3()
        (
            ['real_target_1'],
            [
                {
                    'sensitive': array(['male'], dtype=object), 
                    'tpr': [0.0], 
                    'tnr': [1.0]
                },
                {
                    'sensitive': array(['female'], dtype=object), 
                    'tpr': [1.0], 
                    'tnr': [0.0]
                }
            ]
        )
        """
        results: list = []
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
                        metric(tn=tn, fp=fp, fn=fn, tp=tp)  # type: ignore[call-arg]
                    )
            results.append(result_for_group)

        return self.data.real_target, results


class PerformanceMetric(Metric):
    """Calculate performance related metrics:
    - accuracy
    - balanced accuracy
    - precision
    - recall
    - f1 score
    
    Parameters
    ----------
    data : Dataset | pd.DataFrame
        input data
    metrics : Collection | Sequence | None, optional
        a sequence of metrics or a dictionary with keys being custom labels
        and values a callable that takes as input y_true and y_pred. default
        functions from sklearn.metrics are:
        - accuracy_score()
        - balanced_accuracy_score()
        - precision_score()
        - recall_score()
        - f1_score_score()
        or any custom metric that takes y_true and y_pred and parameters 
        respectively.
    sensitive : Sequence[str], optional if data is a Dataset object
        sequence of column names corresponding to sensitive features
        (Ex: gender, race...), by default None
    real_target : Sequence[str], optional if data is a Dataset object
        sequence of column names corresponding to the real targets
        (true labels), by default None
    predicted_target : Sequence[str], optional
        sequence of column names corresponding to the predicted targets,
        by default None
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        sequence of the positive labels corresponding to the provided targets,
        by default None
    
    Raises
        ------
        ValueError
            if the predictions column is not provided.
        KeyError
            if the key of a metric is 'sensitive' which is already reserved
            to the sensitive groups.
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
                if "sensitive" in metrics.keys():
                    raise KeyError(
                        "metric label cannot be 'sensitive'. Change the label " "to fix"
                    )
                self.metrics = metrics

            else:
                metrics = set(metrics)
                self.metrics = {}
                for metric in metrics:
                    if metric.__name__ == "sensitive":
                        raise KeyError(
                            "metric label cannot be 'sensitive'. Rename your "
                            "function or use a dictionary to set a label for it"
                        )
                    self.metrics[metric.__name__] = metric

    def __call__(self) -> tuple[Sequence, list]:
        """Calculate performance related metrics:
        - accuracy
        - balanced accuracy
        - precision
        - recall
        - f1 score
        for the different demographic groups present in the sensitive feature.

        Returns
        -------
        tuple[Sequence, list]
            a tuple containing two elements:
                targets (Sequence[str]): The target variables used for
                calculation.
                results (list[dict]): A list of dictionaries, where the keys:
                    sensitive: The name of the sensitive group.
                    label: The corresponding result for the sensitive group.
        
        Examples
        --------
        >>> import pandas as pd
        >>> from fair_mango.metrics.metrics import PerformanceMetric
        >>> from sklearn.metrics import (
        ...     accuracy_score,
        ...     balanced_accuracy_score,
        ...     f1_score,
        ...     precision_score,
        ...     recall_score,
        ... )
        >>> data = {
        ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male'],
        ...     'sensitive_2': ['white', 'white', 'black', 'black', 'black'],
        ...     'real_target_1': [0, 1, 0, 1, 0],
        ...     'real_target_2': ['no', 'yes', 'yes', 'yes', 'no'],
        ...     'predicted_target_1': [0, 1, 1, 0, 0],
        ...     'predicted_target_2': ['no', 'no', 'yes', 'yes', 'yes'],
        ... }
        >>> df = pd.DataFrame(data)
        >>> performance_metric_1 = PerformanceMetric(
        ...     data=df,
        ...     sensitive=['sensitive_1'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1]
        ... )
        >>> performance_metric_1()
        (
            ['real_target_1'],
            [
                {
                    'sensitive': array(['male'], dtype=object),
                    'accuracy': [0.6666666666666666],
                    'balanced accuracy': [0.5],
                    'precision': [0.0],
                    'recall': [0.0],
                    'f1-score': [0.0]
                },
                {
                    'sensitive': array(['female'], dtype=object),
                    'accuracy': [0.5],
                    'balanced accuracy': [0.5],
                    'precision': [0.5],
                    'recall': [1.0],
                    'f1-score': [0.6666666666666666]
                }
            ]
        )
        >>> dataset2 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1'],
        ...     positive_target=[1]
        ... )
        >>> performance_metric_2 = PerformanceMetric(
        ...     data=dataset2,
        ...     metrics=[f1_score],
        ... )
        >>> performance_metric_2()
        (
            ['real_target_1'],
            [
                {
                    'sensitive': array(['male'], dtype=object), 
                    'f1_score': [0.0]
                },
                {
                    'sensitive': array(['female'], dtype=object),
                    'f1_score': [0.6666666666666666]
                }
            ]
        )
        >>> performance_metric_3 = PerformanceMetric(
        ...     data=dataset2,
        ...     metrics={
        ...         'acc': accuracy_score,
        ...         'bal_acc': balanced_accuracy_score
        ...     }
        ... )
        >>> performance_metric_3()
        (
            ['real_target_1'],
            [
                {
                    'sensitive': array(['male'], dtype=object),
                    'acc': [0.6666666666666666],
                    'bal_acc': [0.5]
                },
                {
                    'sensitive': array(['female'], dtype=object),
                    'acc': [0.5],
                    'bal_acc': [0.5]
                }
            ]
        )
        """
        results: list = []
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

            results.append(result_for_group)

        return self.data.real_target, results


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
            **{"metrics": {"result": true_positive_rate}},
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
            **{"metrics": {"result": false_positive_rate}},
        )


class DemographicParityRatio(FairnessMetricRatio):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "demographic_parity_ratio",
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


class DisparateImpactRatio(FairnessMetricRatio):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "disparate_impact_ratio",
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


class EqualOpportunityRatio(FairnessMetricRatio):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "equal_opportunity_ratio",
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
            **{"metrics": {"result": true_positive_rate}},
        )


class FalsePositiveRateRatio(FairnessMetricRatio):
    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        label: str = "false_positive_rate_ratio",
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
            **{"metrics": {"result": false_positive_rate}},
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

    def _compute(
        self,
    ) -> tuple[dict[tuple, np.ndarray[float]], dict[tuple, np.ndarray[float]]]:
        tpr = EqualOpportunityDifference(self.data)
        fpr = FalsePositiveRateDifference(self.data)
        tpr.summary()
        fpr.summary()
        tpr_diff = tpr.differences[1]
        fpr_diff = fpr.differences[1]

        return tpr_diff, fpr_diff

    def summary(self) -> dict[str, dict[str, float | tuple | None]]:
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

    def rank(self) -> dict[str, dict[tuple[str], float]]:
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

    def is_biased(self, threshold: float = 0.1) -> dict[str, bool]:
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
        self.ranking: dict | None = None
        self.tpr: dict | None = None
        self.fpr: dict | None = None

    def _compute(self) -> tuple[dict, dict]:
        tpr = EqualOpportunityRatio(self.data)
        fpr = FalsePositiveRateRatio(self.data)
        tpr.summary()
        fpr.summary()
        tpr_ratio = tpr.ratios[1]
        fpr_ratio = fpr.ratios[1]

        return tpr_ratio, fpr_ratio

    def summary(self) -> dict[str, dict[str, float | tuple | None]]:
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

    def rank(self) -> dict[str, dict[tuple[str], float]]:
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

    def is_biased(self, threshold: float = 0.1) -> dict[str, bool]:
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
