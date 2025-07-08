from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd

from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore
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
from fair_mango.typing import (
    DatasetTargetResult,
    DetailedPerformanceMetricsResult,
    MetricResult,
    RankResult,
)


class SelectionRate(Metric):
    """Calculate the selection rates for all the different sensitive groups
    present in the sensitive feature.

    The Selection Rate is the ratio of the number of instances selected
    (predicted as positive) to the total number of instances. It is a measure
    of the proportion of the population that chosen.

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
        Sequence of column names corresponding to the real target
        (true labels), by default None.
    predicted_target : Sequence[str] | None, optional
        Sequence of column names corresponding to the predicted target,
        by default None.
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        Sequence of the positive labels corresponding to the provided target,
        by default None.
    """

    def __init__(
        self,
        data: Dataset,
        use_y_true: bool = False,
        label: str = "result",
    ):
        super().__init__(data)
        self.use_y_true = use_y_true
        self.label = label

    def __call__(self) -> tuple[str, list[MetricResult]]:
        """Calculate the selection rates for all the different sensitive groups
        present in the sensitive feature.

        Returns
        -------
        tuple[Sequence[str], list[MetricResult]]
            A tuple containing the target name and list of MetricResult dictionaries.
            A tuple containing two elements:
            - target (Sequence[str]): The target variables used for
              calculation.
            - results (list[MetricResult]): A list of MetricResult,
              where each dictionary has two keys:
                1. sensitive: The name of the sensitive group.
                2. result: The selection rate for the sensitive group.

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
        results: list[MetricResult] = []
        if self.use_y_true:
            target = self.data.real_target
            target_by_group = self.real_target_by_group
        else:
            if self.data.predicted_target is None:
                raise ValueError(
                    "No predictions found, provide predicted_target parameter "
                    "when creating the dataset or set use_y_true to True to "
                    "use the real labels"
                )
            target= self.data.predicted_target
            target_by_group = self.predicted_target_by_group
        
        for group in target_by_group:
            group_sensitive = group["sensitive"]
            y_group = group["result"] 
            
            results.append({
                "sensitive": group_sensitive,  
                self.label: float(y_group.mean()) 
            })
        return target, results

    def all_data(self) -> dict[str, float]:
        """Compute overall selection rate corresponding to the whole dataset.

        Returns
        -------
        pd.Series
            The target name as the index and the corresponding selection rate
            as the value.

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
            return {self.data.real_target: self.data.df[self.data.real_target].mean()}
        else:
            if self.data.predicted_target is None:
                raise ValueError(
                    "No predictions found, provide predicted_target parameter "
                    "when creating the dataset or set use_y_true to True to "
                    "use the real labels"
                )
            else:
                return {
                    self.data.predicted_target: self.data.df[
                        self.data.predicted_target
                    ].mean()
                }


class ConfusionMatrix(Metric):
    """Calculate the confusion matrix related metrics:
    - false positive rate
    - false negative rate
    - true positive rate
    - true negative rate
    for all the different sensitive groups present in the sensitive feature.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        Input data.
    metrics : Sequence[Callable] | set[Callable] | dict[str, Callable] | None, optional
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
        Sequence of column names corresponding to the real target
        (true labels), by default None.
    predicted_target : Sequence[str] | None, optional
        Sequence of column names corresponding to the predicted target,
        by default None.
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        Sequence of the positive labels corresponding to the provided target,
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
        data: Dataset,
        metrics: Collection | Sequence | None = None,
    ) -> None:
        super().__init__(data)
        if self.predicted_target_by_group == []:
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
                        "metric label cannot be 'sensitive'. Change the label to fix"
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
        """Calculate the confusion matrix related metrics:
        - false positive rate.
        - false negative rate.
        - true positive rate.
        - true negative rate.
        for all the different sensitive groups present in the sensitive feature.

        Returns
        -------
        tuple[Sequence, list]
            A tuple containing two elements:
            - target (Sequence[str]): The target variables used for
              calculation.
            - results (list[dict]): A list of dictionaries, where the keys:
                1. sensitive: The name of the sensitive group.
                2. label: The corresponding result for the sensitive group.

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
            self.real_target_by_group, self.predicted_target_by_group
        ):
            group_ = real_group["sensitive"]
            real_values = real_group["result"]
            predicted_values = predicted_group["result"]
            result_for_group = {"sensitive": group_}

            conf_matrix = confusion_matrix(real_values, predicted_values, labels=[0, 1])
            tn = conf_matrix[0, 0]
            tp = conf_matrix[1, 1]
            fn = conf_matrix[1, 0]
            fp = conf_matrix[0, 1]

            for metric_name, metric in self.metrics.items():
                if metric_name not in result_for_group:
                    result_for_group[metric_name] = []  # type: ignore
                result_for_group[metric_name].append(  # type: ignore
                    metric(tn=tn, fp=fp, fn=fn, tp=tp)  # type: ignore
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
    for all the different sensitive groups present in the sensitive feature.

    Parameters
    ----------
    data : Dataset
        Input data.
    metrics : set[Callable] | dict[str, Callable] | None, optional
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
        data: Dataset,
        metrics: Collection | None = None,
    ) -> None:
        super().__init__(data)

        if self.predicted_target_by_group == []:
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
                        "metric label cannot be 'sensitive'. Change the label to fix"
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
        for all the different sensitive groups present in the sensitive feature.

        Returns
        -------
        tuple[Sequence, list]
            A tuple containing two elements:
            - target (Sequence[str]): The target variables used for
              calculation.
            - results (list[dict]): A list of dictionaries, where the keys:
                1. sensitive: The name of the sensitive group.
                2. label: The corresponding result for the sensitive group.

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
            self.real_target_by_group, self.predicted_target_by_group
        ):
            group_ = real_group["sensitive"]
            real_values = real_group["result"]
            predicted_values = predicted_group["result"]
            result_for_group = {"sensitive": group_}

            for metric_name, metric in self.metrics.items():
                if metric_name not in result_for_group:
                    result_for_group[metric_name] = []  # type: ignore
                result_for_group[metric_name].append(  # type: ignore
                    metric(real_values, predicted_values)
                )

            results.append(result_for_group)

        return self.data.real_target, results


class DemographicParityDifference(FairnessMetricDifference):
    """Calculate Demographic Parity Fairness Metric using "difference" to
    calculate the disparity between the different sensitive groups present
    in the sensitive feature.

    Demographic Parity calculates the "difference" in the Selection Rate in the
    real target to detect if there is any bias in the **dataset**.

    The Selection Rate is the ratio of the number of instances selected
    (predicted as positive) to the total number of instances. It is a measure
    of the proportion of the population that chosen.

    Parameters
    ----------
    data : Dataset
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".

    Examples
    --------
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
        data: Dataset,
        label: str = "demographic_parity_difference",
    ) -> None:
        super().__init__(
            data,
            SelectionRate,
            "performance",
            **{"use_y_true": True, "label": label},
        )


class DisparateImpactDifference(FairnessMetricDifference):
    """Calculate Disparate Impact Fairness Metric using "difference" to
    calculate the disparity between the different sensitive groups present
    in the sensitive feature.

    Disparate Impact calculates the "difference" in the Selection Rate in the
    predicted target to detect if there is any bias in the **model**.

    The Selection Rate is the ratio of the number of instances selected
    (predicted as positive) to the total number of instances. It is a measure
    of the proportion of the population that chosen.

    Parameters
    ----------
    data : Dataset
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".

    Examples
    --------
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
        data: Dataset,
        label: str = "disparate_impact_difference",
    ) -> None:
        super().__init__(
            data,
            SelectionRate,
            "performance",
            **{"use_y_true": False, "label": label},
        )


class EqualOpportunityDifference(FairnessMetricDifference):
    """Calculate Equal Opportunity Fairness Metric using "difference" to
    calculate the disparity between the different sensitive groups present
    in the sensitive feature.

    Equal Opportunity calculates the "difference" in the True Positive Rate in
    the target to detect if there is any bias in the **model**.

    The True Positive Rate (TPR) is the ratio of correctly predicted positive
    observations to all actual positives. It is a measure of a model's ability
    to correctly identify positive instances.

    Parameters
    ----------
    data : Dataset
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".

    Examples
    --------
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
        data: Dataset,
        label: str = "equal_opportunity_difference",
    ) -> None:
        super().__init__(
            data,
            ConfusionMatrix,
            "performance",
            **{"metrics": {"result": true_positive_rate}, "label": label},
        )


class FalsePositiveRateDifference(FairnessMetricDifference):
    """Calculate False Positive Rate Parity Fairness Metric using "difference"
    to calculate the disparity between the different sensitive groups present
    in the sensitive feature.

    False Positive Rate Parity calculates the "difference" in the False
    Positive Rate in the target to detect if there is any bias in the
    **model**.

    The False Positive Rate (FPR) is the ratio of incorrectly predicted
    positive observations to all actual negatives. It is a measure of the
    proportion of negatives that are incorrectly identified as positives by
    the model.

    Parameters
    ----------
    data : Dataset
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".

    Examples
    --------
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
        data: Dataset,
        label: str = "false_positive_rate_difference",
    ) -> None:
        super().__init__(
            data,
            ConfusionMatrix,
            "error",
            **{"metrics": {"result": false_positive_rate}, "label": label},
        )


class DemographicParityRatio(FairnessMetricRatio):
    """Calculate Demographic Parity Fairness Metric using "ratio" to calculate
    the disparity between the different sensitive groups present in the
    sensitive feature.

    Demographic Parity calculates the "ratio" of the Selection Rate in the
    real target to detect if there is any bias in the **dataset**.

    The Selection Rate is the ratio of the number of instances selected
    (predicted as positive) to the total number of instances. It is a measure
    of the proportion of the population that chosen.

    Parameters
    ----------
    data : Dataset
        Input data.

    Examples
    --------
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
        data: Dataset,
        label: str = "demographic_parity_ratio",
    ) -> None:
        super().__init__(
            data,
            SelectionRate,
            "performance",
            **{"use_y_true": True, "label": label},
        )


class DisparateImpactRatio(FairnessMetricRatio):
    """Calculate Disparate Impact Fairness Metric using "ratio" to calculate
    the disparity between the different sensitive groups present in the
    sensitive feature.

    Disparate Impact calculates the "ratio" of the Selection Rate in the
    predicted target to detect if there is any bias in the **model**.

    The Selection Rate is the ratio of the number of instances selected
    (predicted as positive) to the total number of instances. It is a measure
    of the proportion of the population that chosen.

    Parameters
    ----------
    data : Dataset
        Input data.

    Examples
    --------
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
        data: Dataset,
        label: str = "disparate_impact_ratio",
    ) -> None:
        super().__init__(
            data,
            SelectionRate,
            "performance",
            **{"use_y_true": False, "label": label},
        )


class EqualOpportunityRatio(FairnessMetricRatio):
    """Calculate Equal Opportunity Fairness Metric using "ratio" to calculate
    the disparity between the different sensitive groups present in the
    sensitive feature.

    Equal Opportunity calculates the "ratio" of the True Positive Rate in
    the target to detect if there is any bias in the **model**.

    The True Positive Rate (TPR) is the ratio of correctly predicted positive
    observations to all actual positives. It is a measure of a model's ability
    to correctly identify positive instances.

    Parameters
    ----------
    data : Dataset
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".

    Examples
    --------
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
        data: Dataset,
        label: str = "equal_opportunity_ratio",
    ) -> None:
        super().__init__(
            data,
            ConfusionMatrix,
            "performance",
            **{"metrics": {"result": true_positive_rate}, "label": label},
        )


class FalsePositiveRateRatio(FairnessMetricRatio):
    """Calculate False Positive Rate Parity Fairness Metric using "ratio" to
    calculate the disparity between the different sensitive groups present
    in the sensitive feature.

    False Positive Rate Parity calculates the "ratio" of the False Positive
    Rate in the target to detect if there is any bias in the **model**.

    The False Positive Rate (FPR) is the ratio of incorrectly predicted
    positive observations to all actual negatives. It is a measure of the
    proportion of negatives that are incorrectly identified as positives by
    the model.

    Parameters
    ----------
    data : Dataset
        Input data.
    label : str
        The key to give to the result in the different returned dictionaries,
        by default "demographic_parity_difference".

    Examples
    --------
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
        data: Dataset,
        label: str = "false_positive_rate_ratio",
    ) -> None:
        super().__init__(
            data,
            ConfusionMatrix,
            "error",
            **{"metrics": {"result": false_positive_rate}, "label": label},
        )


class EqualisedOddsDifference:
    """Calculate Equalised Odds Fairness Metric using "difference" to calculate
    the disparity between the different sensitive groups present in the
    sensitive feature.

    Equalised Odds calculates the "difference" in the True Positive Rate and
    False Positive Rate in the target to detect if there is any bias in the
    **model**.

    The True Positive Rate (TPR) is the ratio of correctly predicted positive
    observations to all actual positives. It is a measure of a model's ability
    to correctly identify positive instances.

    The False Positive Rate (FPR) is the ratio of incorrectly predicted
    positive observations to all actual negatives. It is a measure of the
    proportion of negatives that are incorrectly identified as positives by
    the model.

    Parameters
    ----------
    data : Dataset
        Input data.

    Examples
    --------
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
        data: Dataset,
    ) -> None:
        self.data = data
        self.label = "equalised_odds_difference"
        self.ranking: dict | None = None
        self.tpr: list[dict] | None = None
        self.fpr: list[dict] | None = None

    def _compute(
        self,
    ) -> tuple[list[dict], list[dict]]:
        """Calculate the disparity in the True Positive Rate and False Positive
        Rate using "difference" between every possible pair in the provided
        groups.

        Returns
        -------
        tuple[list[dict], list[dict]]
            A tuple with two lists of DisparityResult dictionaries.
        """
        tpr = EqualOpportunityDifference(self.data)
        fpr = FalsePositiveRateDifference(self.data)
        tpr.summary()
        fpr.summary()
        tpr_diff = tpr.results  
        fpr_diff = fpr.results  

        return tpr_diff, fpr_diff

    def summary(self) -> dict[str, float | list[str] | None]:
        """Return the Equalised Odds metric value, in other words the biggest
        disparity found in the True Positive Rate and False Positive Rate with
        specifying the priviliged and discriminated groups.

        Returns
        -------
        dict[str, float | list[str] | None]
            A dictionary with keys for the metric value, privileged group,
            and unprivileged group.
        """
        self.result: dict = {}
        target = self.data.real_target

        self.result.setdefault(
            target, {self.label: 0.0, "privileged": None, "unprivileged": None}
        )

        if (self.tpr is None) or (self.fpr is None):
            self.tpr, self.fpr = self._compute()

        # Process the new list-based disparity results
        for tpr_result, fpr_result in zip(self.tpr, self.fpr):
            tpr_disparity = tpr_result["disparity"]
            fpr_disparity = fpr_result["disparity"]
            
            if np.abs(tpr_disparity) > self.result[target][self.label]:
                self.result[target][self.label] = np.abs(tpr_disparity)
                if tpr_disparity > 0:
                    self.result[target]["privileged"] = tpr_result["group_1"]
                    self.result[target]["unprivileged"] = tpr_result["group_2"]
                else:
                    self.result[target]["privileged"] = tpr_result["group_2"]
                    self.result[target]["unprivileged"] = tpr_result["group_1"]
                    
            if np.abs(fpr_disparity) > self.result[target][self.label]:
                self.result[target][self.label] = np.abs(fpr_disparity)
                if fpr_disparity > 0:
                    self.result[target]["privileged"] = fpr_result["group_2"]
                    self.result[target]["unprivileged"] = fpr_result["group_1"]
                else:
                    self.result[target]["privileged"] = fpr_result["group_1"]
                    self.result[target]["unprivileged"] = fpr_result["group_2"]

        return {
            self.label: self.result[target][self.label],
            "privileged_group": self.result[target]["privileged"],
            "unprivileged_group": self.result[target]["unprivileged"],
        }

    def rank(self) -> dict[str, list[RankResult]]:
        """Assign a score to every sensitive group present in the sensitive
        features and rank them from most privileged to most discriminated.
        The score can be interpreted like:
        - ['Male': 0.0314]: Males have on average a score higher by 3.14% than
          the Females.
        - ['White': -0.0628]: Whites have on average a score lower by 6.28% than
          other groups (Black, Asian...).

        Returns
        -------
        dict[str, list[RankResult]]
            A dictionary with:
            - keys: name of the target variable.
            - values: a list of RankResult dictionaries with sensitive group and score.
        """
        result: dict = {}
        self.ranking = {}
        target = self.data.real_target

        result.setdefault(target, {})
        self.ranking.setdefault(target, {})

        if (self.tpr is None) or (self.fpr is None):
            self.tpr, self.fpr = self._compute()

        for tpr_result, fpr_result in zip(self.tpr, self.fpr):
            tpr_disparity = tpr_result["disparity"]
            fpr_disparity = fpr_result["disparity"]
            
            if np.abs(tpr_disparity) > np.abs(fpr_disparity):
                group1_key = tuple(tpr_result["group_1"])
                group2_key = tuple(tpr_result["group_2"])
                result[target].setdefault(group1_key, []).append(tpr_disparity)
                result[target].setdefault(group2_key, []).append(-tpr_disparity)
            else:
                group1_key = tuple(fpr_result["group_1"])
                group2_key = tuple(fpr_result["group_2"])
                result[target].setdefault(group1_key, []).append(-fpr_disparity)
                result[target].setdefault(group2_key, []).append(fpr_disparity)

        target_result = result[target]
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

        rank_results = []
        for group_tuple, score in self.ranking[target].items():
            rank_results.append({
                "sensitive": list(group_tuple),
                "score": float(score)
            })

        return {target: rank_results}

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

        ranking_nested = self.rank()

        bias: dict = {}
        target = self.data.real_target
        rank_list = ranking_nested[target]

        if rank_list:
            scores = [item["score"] for item in rank_list]
            max_diff = scores[0] if scores else 0
            min_diff = scores[-1] if scores else 0
            bias[target] = max_diff > threshold or min_diff < -threshold
        else:
            bias[target] = False

        return bias


class EqualisedOddsRatio:
    """Calculate Equalised Odds Fairness Metric using "ratio" to calculate
    the disparity between the different sensitive groups present in the
    sensitive feature.

    Equalised Odds calculates the "ratio" of the True Positive Rate and False
    Positive Rate in the target to detect if there is any bias in the
    **model**.

    The True Positive Rate (TPR) is the ratio of correctly predicted positive
    observations to all actual positives. It is a measure of a model's ability
    to correctly identify positive instances.

    The False Positive Rate (FPR) is the ratio of incorrectly predicted
    positive observations to all actual negatives. It is a measure of the
    proportion of negatives that are incorrectly identified as positives by
    the model.

    Parameters
    ----------
    data : Dataset
        Input data.

    Examples
    --------
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
        data: Dataset,
    ) -> None:
        self.data = data
        self.label = "equalised_odds_ratio"
        self.ranking: dict | None = None
        self.tpr: list[dict] | None = None
        self.fpr: list[dict] | None = None

    def _compute(self) -> tuple[list[dict], list[dict]]:
        """Calculate the disparity in the True Positive Rate and False Positive
        Rate using "ratio" between every possible pair in the provided groups.

        Returns
        -------
        tuple[list[dict], list[dict]]
            A tuple with two lists of DisparityResult dictionaries.
        """
        tpr = EqualOpportunityRatio(self.data)
        fpr = FalsePositiveRateRatio(self.data)
        tpr.summary()
        fpr.summary()
        tpr_ratio = tpr.results
        fpr_ratio = fpr.results

        return tpr_ratio, fpr_ratio

    def summary(self) -> dict[str, float | list[str] | None]:
        """Return the Equalised Odds metric value, in other words the biggest
        disparity found in the True Positive Rate and False Positive Rate with
        specifying the priviliged and discriminated groups.

        Returns
        -------
        dict[str, float | list[str] | None]
            A dictionary with keys for the metric value, privileged group,
            and unprivileged group.
        """
        self.result: dict = {}
        target = self.data.real_target

        self.result.setdefault(
            target, {self.label: 1.0, "privileged": None, "unprivileged": None}
        )

        if (self.tpr is None) or (self.fpr is None):
            self.tpr, self.fpr = self._compute()

        for tpr_result, fpr_result in zip(self.tpr, self.fpr):
            tpr_ratio = tpr_result["disparity"]
            fpr_ratio = fpr_result["disparity"]
            
            if tpr_ratio > 1:
                temp = 1 / tpr_ratio
            else:
                temp = tpr_ratio

            if temp < self.result[target][self.label]:
                self.result[target][self.label] = temp
                if tpr_ratio > 1:
                    self.result[target]["privileged"] = tpr_result["group_2"]
                    self.result[target]["unprivileged"] = tpr_result["group_1"]
                else:
                    self.result[target]["privileged"] = tpr_result["group_1"]
                    self.result[target]["unprivileged"] = tpr_result["group_2"]

            if fpr_ratio > 1:
                temp = 1 / fpr_ratio
            else:
                temp = fpr_ratio

            if temp < self.result[target][self.label]:
                self.result[target][self.label] = temp
                if fpr_ratio > 1:
                    self.result[target]["privileged"] = fpr_result["group_2"]
                    self.result[target]["unprivileged"] = fpr_result["group_1"]
                else:
                    self.result[target]["privileged"] = fpr_result["group_1"]
                    self.result[target]["unprivileged"] = fpr_result["group_2"]

        # Return flat format with updated key names
        return {
            self.label: self.result[target][self.label],
            "privileged_group": self.result[target]["privileged"],
            "unprivileged_group": self.result[target]["unprivileged"],
        }

    def rank(self) -> dict[str, list[RankResult]]:
        """Assign a score to every sensitive group present in the sensitive
        features and rank them from most privileged to most discriminated.
        The score can be interpreted like:
        - ['Male': 0.814]: Males have on average 81.4% the score of the
          Females.
        - ['White': 1.20]: Whites have on average 120% the score of the
          other groups (Black, Asian...).

        Returns
        -------
        dict[str, list[RankResult]]
            A dictionary with:
            - keys: name of the target variable.
            - values: a list of RankResult dictionaries with sensitive group and score.
        """
        result: dict = {}
        self.ranking = {}

        target = self.data.real_target

        result.setdefault(target, {})
        self.ranking.setdefault(target, {})

        if (self.tpr is None) or (self.fpr is None):
            self.tpr, self.fpr = self._compute()

        for tpr_result, fpr_result in zip(self.tpr, self.fpr):
            tpr_ratio = tpr_result["disparity"]
            fpr_ratio = fpr_result["disparity"]
            
            if tpr_ratio > 1:
                temp1 = 1 / tpr_ratio
            else:
                temp1 = tpr_ratio

            if fpr_ratio > 1:
                temp2 = 1 / fpr_ratio
            else:
                temp2 = fpr_ratio

            if temp1 < temp2:
                group1_key = tuple(tpr_result["group_1"])
                group2_key = tuple(tpr_result["group_2"])
                result[target].setdefault(group1_key, []).append(tpr_ratio)
                if tpr_ratio == 0:
                    result[target].setdefault(group2_key, []).append(np.inf)
                else:
                    result[target].setdefault(group2_key, []).append(1 / tpr_ratio)
            else:
                group1_key = tuple(fpr_result["group_1"])
                group2_key = tuple(fpr_result["group_2"])
                result[target].setdefault(group1_key, []).append(fpr_ratio)
                if fpr_ratio == 0:
                    result[target].setdefault(group2_key, []).append(np.inf)
                else:
                    result[target].setdefault(group2_key, []).append(1 / fpr_ratio)

        target_result = result[target]
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

        rank_results = []
        for group_tuple, ratio in self.ranking[target].items():
            rank_results.append({
                "sensitive": list(group_tuple),
                "score": float(ratio)
            })

        return {target: rank_results}

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

        ranking_nested = self.rank()

        bias: dict = {}
        target = self.data.real_target
        rank_list = ranking_nested[target]

        if rank_list:
            scores = [item["score"] for item in rank_list]
            min_ratio = scores[0] if scores else 1.0
            max_ratio = scores[-1] if scores else 1.0
            bias[target] = max_ratio > (1 / threshold) or min_ratio < threshold
        else:
            bias[target] = False

        return bias
