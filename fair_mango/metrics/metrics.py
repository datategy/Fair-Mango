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
    """Calculate selection rate for different sensitive groups.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    use_y_true : bool, optional
        if True use the real label else use the predictions, by default False
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
        A tuple containing two elements:
        - targets (Sequence[str]): The target variables used for
          calculation.
        - results (list[dict]): A list of dictionaries, where each
          dictionary has two keys:
            - sensitive: The name of the sensitive group.
            - result: The selection rate for the sensitive group.

        Raises
        ------
        ValueError
            If no predictions are found and `use_y_true` is False.

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
            The target name and the corresponding selection rate.

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
        Input data.
    metrics : Collection | Sequence | None, optional
        A sequence of metrics or a dictionary with keys being custom labels
        and values a callable that takes as input tp, tn, fp, fn which are
        extracted from the confusion matrix. Available functions in
        fair_mango.metrics.metrics.base are:
        - false_positive_rate().
        - false_negative_rate().
        - true_positive_rate().
        - true_negative_rate().
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
        If the predictions column is not provided.
    KeyError
        If the key of a metric is 'sensitive' which is already reserved
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
        - false positive rate.
        - false negative rate.
        - true positive rate.
        - true negative rate.
        for the different sensitive groups present in the sensitive feature.

        Returns
        -------
        tuple[Sequence, list]
        A tuple containing two elements:
        - targets (Sequence[str]): The target variables used for
          calculation.
        - results (list[dict]): A list of dictionaries, where the keys:
            - sensitive: The name of the sensitive group.
            - label: The corresponding result for the sensitive group.

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
        ...     predicted_target=['predicted_target_1']
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
    - accuracy.
    - balanced accuracy.
    - precision.
    - recall.
    - f1 score.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    metrics : Collection | None, optional
        A sequence of metrics or a dictionary with keys being custom labels
        and values a callable that takes as input y_true and y_pred. default
        functions from sklearn.metrics are:
        - accuracy_score().
        - balanced_accuracy_score().
        - precision_score().
        - recall_score().
        - f1_score_score().
        or any custom metric that takes y_true and y_pred and parameters
        respectively.
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
        If the predictions column is not provided.
    KeyError
        If the key of a metric is 'sensitive' which is already reserved
        to the sensitive groups.
    """

    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        metrics: Collection | None = None,
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
        - accuracy.
        - balanced accuracy.
        - precision.
        - recall.
        - f1 score.
        for the different sensitive groups present in the sensitive feature.

        Returns
        -------
        tuple[Sequence, list]
        A tuple containing two elements:
        - targets (Sequence[str]): The target variables used for
          calculation.
        - results (list[dict]): A list of dictionaries, where the keys:
            - sensitive: The name of the sensitive group.
            - label: The corresponding result for the sensitive group.

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
        ...     predicted_target=['predicted_target_1']
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
    """Calculate Demographic Parity Fairness Metric using "difference" to
    calculate the disparity between the different sensitive groups present
    in the sensitive feature.

    Demographic Parity calculates the "difference" in the Selection Rate in the
    real targets to detect if there is any bias in the **dataset**.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".
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

    Example
    -------
    >>> import pandas as pd
    >>> from fair_mango.metrics.metrics import DemographicParityDifference
    >>> data = {
    ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male', 'male'],
    ...     'sensitive_2': ['white', 'black', 'black', 'black', 'black', 'white'],
    ...     'real_target': [1, 0, 0, 1, 0, 1],
    ...     'predicted_target': [0, 1, 1, 0, 0, 1],
    ... }
    >>> df = pd.DataFrame(data)
    >>> demographic_parity_diff = DemographicParityDifference(
    ...     data=df,
    ...     sensitive=['sensitive_1', 'sensitive_2'],
    ...     real_target=['real_target'],
    ...     predicted_target=['predicted_target']
    ... )
    >>> demographic_parity_diff.summary()
    {
        'real_target': {
            'demographic_parity_difference': np.float64(1.0),
            'privileged': ('male', 'white'),
            'unprivileged': ('female', 'black')
        }
    }
    >>> demographic_parity_diff.rank()
    {
        'real_target': {
            ('male', 'white'): np.float64(0.75),
            ('male', 'black'): np.float64(0.0),
            ('female', 'black'): np.float64(-0.75)
        }
    }
    >>> demographic_parity_diff.is_biased(0.2)
    {
        'real_target': True
    }
    """

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
    """Calculate Disparate Impact Fairness Metric using "difference" to
    calculate the disparity between the different sensitive groups present
    in the sensitive feature.

    Disparate Impact calculates the "difference" in the Selection Rate in the
    predicted targets to detect if there is any bias in the **model**.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".
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

    Example
    -------
    >>> import pandas as pd
    >>> from fair_mango.metrics.metrics import DisparateImpactDifference
    >>> data = {
    ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male', 'male'],
    ...     'sensitive_2': ['white', 'black', 'black', 'black', 'black', 'white'],
    ...     'real_target': [1, 0, 0, 1, 0, 1],
    ...     'predicted_target': [0, 1, 1, 0, 0, 1],
    ... }
    >>> df = pd.DataFrame(data)
    >>> disparate_impact_diff = DisparateImpactDifference(
    ...     data=df,
    ...     sensitive=['sensitive_1', 'sensitive_2'],
    ...     real_target=['real_target'],
    ...     predicted_target=['predicted_target']
    ... )
    >>> disparate_impact_diff.summary()
    {
        'predicted_target': {
            'disparate_impact_difference': np.float64(1.0),
            'privileged': ('female', 'black'),
            'unprivileged': ('male', 'black')
        }
    }
    >>> disparate_impact_diff.rank()
    {
        'real_target': {
            ('female', 'black'): np.float64(0.75),
            ('male', 'white'): np.float64(0.0),
            ('male', 'black'): np.float64(-0.75)
        }
    }
    >>> disparate_impact_diff.is_biased(0.2)
    {
        'real_target': True
    }
    """

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
    """Calculate Equal Opportunity Fairness Metric using "difference" to
    calculate the disparity between the different sensitive groups present
    in the sensitive feature.

    Equal Opportunity calculates the "difference" in the True Positive Rate in
    the targets to detect if there is any bias in the **model**.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".
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

    Example
    -------
    >>> import pandas as pd
    >>> from fair_mango.metrics.metrics import EqualOpportunityDifference
    >>> data = {
    ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male', 'male'],
    ...     'sensitive_2': ['white', 'black', 'black', 'black', 'black', 'white'],
    ...     'real_target': [1, 0, 1, 1, 0, 1],
    ...     'predicted_target': [0, 1, 1, 0, 0, 1],
    ... }
    >>> df = pd.DataFrame(data)
    >>> equal_opportunity_diff = EqualOpportunityDifference(
    ...     data=df,
    ...     sensitive=['sensitive_1', 'sensitive_2'],
    ...     real_target=['real_target'],
    ...     predicted_target=['predicted_target']
    ... )
    >>> equal_opportunity_diff.summary()
    {
        'real_target': {
            'equal_opportunity_difference': np.float64(1.0),
            'privileged': ('female', 'black'),
            'unprivileged': ('male', 'black')
        }
    }
    >>> equal_opportunity_diff.rank()
    {
        'real_target': {
            ('female', 'black'): np.float64(0.75),
            ('male', 'white'): np.float64(0.0),
            ('male', 'black'): np.float64(-0.75)
        }
    }
    >>> equal_opportunity_diff.is_biased(0.2)
    {
        'real_target': True
    }
    """

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
    """Calculate False Positive Rate Parity Fairness Metric using "difference"
    to calculate the disparity between the different sensitive groups present
    in the sensitive feature.

    False Positive Rate Parity calculates the "difference" in the False
    Positive Rate in the targets to detect if there is any bias in the
    **model**.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".
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

    Example
    -------
    >>> import pandas as pd
    >>> from fair_mango.metrics.metrics import FalsePositiveRateDifference
    >>> data = {
    ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male', 'male'],
    ...     'sensitive_2': ['white', 'black', 'black', 'black', 'black', 'white'],
    ...     'real_target': [1, 0, 1, 0, 0, 0],
    ...     'predicted_target': [0, 1, 0, 1, 0, 1],
    ... }
    >>> df = pd.DataFrame(data)
    >>> fpr_parity_diff = FalsePositiveRateDifference(
    ...     data=df,
    ...     sensitive=['sensitive_1', 'sensitive_2'],
    ...     real_target=['real_target'],
    ...     predicted_target=['predicted_target']
    ... )
    >>> fpr_parity_diff.summary()
    {
        'real_target': {
            'false_positive_rate_difference': np.float64(0.5),
            'privileged': ('male', 'black'),
            'unprivileged': ('female', 'black')
        }
    }
    >>> fpr_parity_diff.rank()
    {
        'real_target': {
            ('male', 'black'): np.float64(0.5),
            ('female', 'black'): np.float64(-0.25),
            ('male', 'white'): np.float64(-0.25)
        }
    }
    >>> fpr_parity_diff.is_biased(0.2)
    {
        'real_target': True
    }
    """

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
    """Calculate Demographic Parity Fairness Metric using "ratio" to calculate
    the disparity between the different sensitive groups present in the
    sensitive feature.

    Demographic Parity calculates the "ratio" of the Selection Rate in the
    real targets to detect if there is any bias in the **dataset**.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".
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

    Example
    -------
    >>> import pandas as pd
    >>> from fair_mango.metrics.metrics import DemographicParityRatio
    >>> data = {
    ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male', 'male'],
    ...     'sensitive_2': ['white', 'black', 'black', 'black', 'black', 'white'],
    ...     'real_target': [1, 0, 0, 1, 0, 1],
    ...     'predicted_target': [0, 1, 1, 0, 0, 1],
    ... }
    >>> df = pd.DataFrame(data)
    >>> demographic_parity_ratio = DemographicParityRatio(
    ...     data=df,
    ...     sensitive=['sensitive_1', 'sensitive_2'],
    ...     real_target=['real_target'],
    ...     predicted_target=['predicted_target']
    ... )
    >>> demographic_parity_ratio.summary()
    {
        'real_target': {
            'demographic_parity_ratio': np.float64(0.0),
            'privileged': ('male', 'black'),
            'unprivileged': ('female', 'black')
        }
    }
    >>> demographic_parity_ratio.rank()
    {
        'real_target': {
            ('male', 'white'): np.float64(0.25),
            ('male', 'black'): np.float64(1.0),
            ('female', 'black'): np.float64(inf)
        }
    }
    >>> demographic_parity_ratio.is_biased(0.8)
    {
        'real_target': True
    }
    """

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
    """Calculate Disparate Impact Fairness Metric using "ratio" to calculate
    the disparity between the different sensitive groups present in the
    sensitive feature.

    Disparate Impact calculates the "ratio" of the Selection Rate in the
    predicted targets to detect if there is any bias in the **model**.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".
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

    Example
    -------
    >>> import pandas as pd
    >>> from fair_mango.metrics.metrics import DisparateImpactRatio
    >>> data = {
    ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male', 'male'],
    ...     'sensitive_2': ['white', 'black', 'black', 'black', 'black', 'white'],
    ...     'real_target': [1, 0, 0, 1, 0, 1],
    ...     'predicted_target': [0, 1, 1, 0, 0, 1],
    ... }
    >>> df = pd.DataFrame(data)
    >>> disparate_impact_ratio = DisparateImpactRatio(
    ...     data=df,
    ...     sensitive=['sensitive_1', 'sensitive_2'],
    ...     real_target=['real_target'],
    ...     predicted_target=['predicted_target']
    ... )
    >>> disparate_impact_ratio.summary()
    {
        'predicted_target': {
            'disparate_impact_ratio': np.float64(0.0),
            'privileged': ('female', 'black'),
            'unprivileged': ('male', 'black')
        }
    }
    >>> disparate_impact_ratio.rank()
    {
        'real_target': {
            ('female', 'black'): np.float64(0.25),
            ('male', 'white'): np.float64(1.0),
            ('male', 'black'): np.float64(inf)
        }
    }
    >>> disparate_impact_ratio.is_biased(0.8)
    {
        'real_target': True
    }
    """

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
    """Calculate Equal Opportunity Fairness Metric using "ratio" to calculate
    the disparity between the different sensitive groups present in the
    sensitive feature.

    Equal Opportunity calculates the "ratio" of the True Positive Rate in
    the targets to detect if there is any bias in the **model**.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".
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

    Example
    -------
    >>> import pandas as pd
    >>> from fair_mango.metrics.metrics import EqualOpportunityRatio
    >>> data = {
    ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male', 'male'],
    ...     'sensitive_2': ['white', 'black', 'black', 'black', 'black', 'white'],
    ...     'real_target': [1, 0, 1, 1, 0, 1],
    ...     'predicted_target': [0, 1, 1, 0, 0, 1],
    ... }
    >>> df = pd.DataFrame(data)
    >>> equal_opportunity_ratio = EqualOpportunityRatio(
    ...     data=df,
    ...     sensitive=['sensitive_1', 'sensitive_2'],
    ...     real_target=['real_target'],
    ...     predicted_target=['predicted_target']
    ... )
    >>> equal_opportunity_ratio.summary()
    {
        'real_target': {
            'equal_opportunity_ratio': np.float64(0.0),
            'privileged': ('female', 'black'),
            'unprivileged': ('male', 'black')
        }
    }
    >>> equal_opportunity_ratio.rank()
    {
        'real_target': {
            ('female', 'black'): np.float64(0.25),
            ('male', 'white'): np.float64(1.0),
            ('male', 'black'): np.float64(inf)
        }
    }
    >>> equal_opportunity_ratio.is_biased(0.8)
    {
        'real_target': True
    }
    """

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
    """Calculate False Positive Rate Parity Fairness Metric using "ratio" to
    calculate the disparity between the different sensitive groups present
    in the sensitive feature.

    False Positive Rate Parity calculates the "ratio" of the False Positive
    Rate in the targets to detect if there is any bias in the **model**.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".
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

    Example
    -------
    >>> import pandas as pd
    >>> from fair_mango.metrics.metrics import FalsePositiveRateRatio
    >>> data = {
    ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male', 'male'],
    ...     'sensitive_2': ['white', 'black', 'black', 'black', 'black', 'white'],
    ...     'real_target': [1, 0, 1, 0, 0, 0],
    ...     'predicted_target': [0, 1, 0, 1, 0, 1],
    ... }
    >>> df = pd.DataFrame(data)
    >>> fpr_parity_ratio = FalsePositiveRateRatio(
    ...     data=df,
    ...     sensitive=['sensitive_1', 'sensitive_2'],
    ...     real_target=['real_target'],
    ...     predicted_target=['predicted_target']
    ... )
    >>> fpr_parity_ratio.summary()
    {
        'real_target': {
            'false_positive_rate_ratio': np.float64(0.5),
            'privileged': ('male', 'black'),
            'unprivileged': ('female', 'black')
        }
    }
    >>> fpr_parity_ratio.rank()
    {
        'real_target': {
            ('male', 'black'): np.float64(0.5),
            ('female', 'black'): np.float64(1.5),
            ('male', 'white'): np.float64(1.5)
        }
    }
    >>> fpr_parity_ratio.is_biased(0.2)
    {
        'real_target': True
    }
    """

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
    """Calculate Equalised Odds Fairness Metric using "difference" to calculate
    the disparity between the different sensitive groups present in the
    sensitive feature.

    Equalised Odds calculates the "difference" in the True Positive Rate and
    False Positive Rate in the targets to detect if there is any bias in the
    **model**.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".
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

    Example
    -------
    >>> import pandas as pd
    >>> from fair_mango.metrics.metrics import EqualisedOddsDifference
    >>> data = {
    ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male', 'male'],
    ...     'sensitive_2': ['white', 'black', 'black', 'black', 'black', 'white'],
    ...     'real_target': [1, 0, 1, 0, 0, 0],
    ...     'predicted_target': [0, 1, 0, 1, 0, 1],
    ... }
    >>> df = pd.DataFrame(data)
    >>> equalised_odds_diff = EqualisedOddsDifference(
    ...     data=df,
    ...     sensitive=['sensitive_1', 'sensitive_2'],
    ...     real_target=['real_target'],
    ...     predicted_target=['predicted_target']
    ... )
    >>> equalised_odds_diff.summary()
    {
        'real_target': {
            'equalised_odds_difference': np.float64(0.5),
            'privileged': ('male', 'black'),
            'unprivileged': ('female', 'black')
        }
    }
    >>> equalised_odds_diff.rank()
    {
        'real_target': {
            ('male', 'black'): np.float64(0.5),
            ('female', 'black'): np.float64(-0.25),
            ('male', 'white'): np.float64(-0.25)
        }
    }
    >>> equalised_odds_diff.is_biased(0.2)
    {
        'real_target': False
    }
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

        self.label = "equalised_odds_difference"
        self.ranking: dict | None = None
        self.tpr: dict | None = None
        self.fpr: dict | None = None

    def _compute(
        self,
    ) -> tuple[dict[tuple, np.ndarray[float]], dict[tuple, np.ndarray[float]]]:
        """Calculate the disparity in the True Positive Rate and False Positive
        Rate using "difference" between every possible pair in the provided
        groups.

        Returns
        -------
        tuple[dict[tuple, np.ndarray[float]], dict[tuple, np.ndarray[float]]]
            A tuple with two dictionaries with:
            - keys: tuple with the pair of the sensitive groups labels.
            - values: a numpy array with the corresponding disparity.
        """
        tpr = EqualOpportunityDifference(self.data)
        fpr = FalsePositiveRateDifference(self.data)
        tpr.summary()
        fpr.summary()
        tpr_diff = tpr.differences[1]
        fpr_diff = fpr.differences[1]

        return tpr_diff, fpr_diff

    def summary(self) -> dict[str, dict[str, float | tuple | None]]:
        """Return the Equalised Odds metric value, in other words the biggest
        disparity found in the True Positive Rate and False Positive Rate with
        specifying the priviliged and discriminated groups.

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


class EqualisedOddsRatio:
    """Calculate Equalised Odds Fairness Metric using "ratio" to calculate
    the disparity between the different sensitive groups present in the
    sensitive feature.

    Equalised Odds calculates the "ratio" of the True Positive Rate and False
    Positive Rate in the targets to detect if there is any bias in the
    **model**.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".
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

    Example
    -------
    >>> import pandas as pd
    >>> from fair_mango.metrics.metrics import EqualisedOddsRatio
    >>> data = {
    ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male', 'male'],
    ...     'sensitive_2': ['white', 'black', 'black', 'black', 'black', 'white'],
    ...     'real_target': [1, 0, 1, 0, 0, 0],
    ...     'predicted_target': [0, 1, 0, 1, 0, 1],
    ... }
    >>> df = pd.DataFrame(data)
    >>> equalised_odds_ratio = EqualisedOddsRatio(
    ...     data=df,
    ...     sensitive=['sensitive_1', 'sensitive_2'],
    ...     real_target=['real_target'],
    ...     predicted_target=['predicted_target']
    ... )
    >>> equalised_odds_ratio.summary()
    {
        'real_target': {
            'equalised_odds_ratio': np.float64(0.5),
            'privileged': ('male', 'black'),
            'unprivileged': ('female', 'black')
        }
    }
    >>> equalised_odds_ratio.rank()
    {
        'real_target': {
            ('male', 'black'): np.float64(0.5),
            ('female', 'black'): np.float64(1.5),
            ('male', 'white'): np.float64(1.5)
        }
    }
    >>> equalised_odds_ratio.is_biased(0.2)
    {
        'real_target': True
    }
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

        self.label = "equalised_odds_ratio"
        self.ranking: dict | None = None
        self.tpr: dict | None = None
        self.fpr: dict | None = None

    def _compute(self) -> tuple[dict, dict]:
        """Calculate the disparity in the True Positive Rate and False Positive
        Rate using "ratio" between every possible pair in the provided groups.

        Returns
        -------
        dict[tuple, np.ndarray[float]]
            A dictionary with:
            - keys: tuple with the pair of the sensitive groups labels.
            - values: a numpy array with the corresponding disparity.
        """
        tpr = EqualOpportunityRatio(self.data)
        fpr = FalsePositiveRateRatio(self.data)
        tpr.summary()
        fpr.summary()
        tpr_ratio = tpr.ratios[1]
        fpr_ratio = fpr.ratios[1]

        return tpr_ratio, fpr_ratio

    def summary(self) -> dict[str, dict[str, float | tuple | None]]:
        """Return the Equalised Odds metric value, in other words the biggest
        disparity found in the True Positive Rate and False Positive Rate with
        specifying the priviliged and discriminated groups.

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
            min_ratio, max_ratio = list(dicts.values())[0], list(dicts.values())[-1]
            if max_ratio > (1 / threshold) or min_ratio < threshold:
                bias[target] = True
            else:
                bias[target] = False

        return bias
