from collections.abc import Callable, Sequence
from typing import Any

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
from fair_mango.typing import (
    DisparateImpactSummaryResult,
    EqualOpportunitySummaryResult,
    FalsePositiveRateSummaryResult,
    EqualisedOddsSummaryResult,
    DemographicParitySummaryResult,
    RankResult,
)


class SelectionRate(Metric):
    def __init__(self, dataset: Dataset, use_y_true: bool = True) -> None:
        super().__init__(dataset)
        self.use_y_true = use_y_true

    def __call__(self) -> list[dict[str, Any]]:
        if self.use_y_true:
            if self.data.real_target is None:
                msg = "Real target not specified when creating Dataset. Please specify a column name for the real target or set use_y_true to False."
                raise ValueError(msg)
            target = self.data.real_target
            target_by_group = self.real_target_by_group
        else:
            if self.data.predicted_target is None:
                msg = "Predicted target not specified when creating Dataset. Please specify a column name for the predicted target or set use_y_true to True."
                raise ValueError(msg)
            target = self.data.predicted_target
            target_by_group = self.predicted_target_by_group

        result = []
        for group in target_by_group:
            group_sensitive = group["sensitive"]
            y_group = group["result"]
            result.append({"sensitive": group_sensitive, "result": float(y_group.mean())})

        return result


class ConfusionMatrix(Metric):
    def __init__(
        self,
        dataset: Dataset,
        metrics: dict[str, Callable] | Sequence[Callable] | None = None,
    ) -> None:
        super().__init__(dataset)
        if self.predicted_target_by_group == []:
            raise ValueError(
                "No predictions found, provide predicted_target parameter "
                "when creating the dataset"
            )
        if metrics is None:
            self.metrics = {
                "false_negative_rate": false_negative_rate,
                "false_positive_rate": false_positive_rate,
                "true_negative_rate": true_negative_rate,
                "true_positive_rate": true_positive_rate,
            }
        elif isinstance(metrics, dict):
            if "sensitive" in metrics:
                msg = "Cannot use 'sensitive' as a key for metrics"
                raise KeyError(msg)
            self.metrics = metrics
        else:
            self.metrics = {metric.__name__: metric for metric in metrics}

    def __call__(self) -> list[dict[str, Any]]:
        target = self.data.real_target
        result = []
        for real_group, predicted_group in zip(
            self.real_target_by_group, self.predicted_target_by_group
        ):
            group_sensitive = real_group["sensitive"]
            real_values = real_group["result"]
            predicted_values = predicted_group["result"]
            result_for_group: dict[str, Any] = {"sensitive": np.array(group_sensitive)}

            conf_matrix = confusion_matrix(real_values, predicted_values, labels=[0, 1])
            tn = conf_matrix[0, 0]
            tp = conf_matrix[1, 1]
            fn = conf_matrix[1, 0]
            fp = conf_matrix[0, 1]

            for metric_name, metric in self.metrics.items():
                result_for_group[metric_name] = [
                    metric(tn=tn, fp=fp, fn=fn, tp=tp)
                ]

            result.append(result_for_group)

        return result


class PerformanceMetric(Metric):
    def __init__(
        self,
        dataset: Dataset,
        metrics: dict[str, Callable] | Sequence[Callable] | None = None,
    ) -> None:
        super().__init__(dataset)
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
        elif isinstance(metrics, dict):
            if "sensitive" in metrics:
                msg = "Cannot use 'sensitive' as a key for metrics"
                raise KeyError(msg)
            self.metrics = metrics
        else:
            self.metrics = {metric.__name__: metric for metric in metrics}

    def __call__(self) -> list[dict[str, Any]]:
        target = self.data.real_target
        result = []
        for real_group, predicted_group in zip(
            self.real_target_by_group, self.predicted_target_by_group
        ):
            group_sensitive = real_group["sensitive"]
            real_values = real_group["result"]
            predicted_values = predicted_group["result"]
            result_for_group: dict[str, Any] = {"sensitive": np.array(group_sensitive)}

            for metric_name, metric in self.metrics.items():
                result_for_group[metric_name] = [
                    metric(real_values, predicted_values)
                ]

            result.append(result_for_group)

        return result


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

    def rank(self) -> list[dict]:
        """Assign a score to every sensitive group present in the sensitive
        features and rank them from most privileged to most discriminated.

        Returns
        -------
        list[dict]
            List of ranking dictionaries with 'sensitive' and 'score' keys.
        """
        result: dict = {}
        ranking: dict = {}

        if (self.tpr is None) or (self.fpr is None):
            self.tpr, self.fpr = self._compute()

        for tpr_result, fpr_result in zip(self.tpr, self.fpr):
            tpr_disparity = tpr_result["disparity"]
            fpr_disparity = fpr_result["disparity"]
            
            if np.abs(tpr_disparity) > np.abs(fpr_disparity):
                group1_key = tuple(tpr_result["group_1"])
                group2_key = tuple(tpr_result["group_2"])
                result.setdefault(group1_key, []).append(tpr_disparity)
                result.setdefault(group2_key, []).append(-tpr_disparity)
            else:
                group1_key = tuple(fpr_result["group_1"])
                group2_key = tuple(fpr_result["group_2"])
                result.setdefault(group1_key, []).append(-fpr_disparity)
                result.setdefault(group2_key, []).append(fpr_disparity)

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

        # Convert to the expected format
        ranking_list = []
        for group_tuple, score in ranking.items():
            ranking_list.append({
                "sensitive": list(group_tuple),
                "score": float(score)
            })
        
        return ranking_list

    def is_biased(self, threshold: float = 0.1) -> bool:
        """Return a decision of whether there is bias or not
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
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in range [0, 1]")

        rank_results = self.rank()

        if rank_results:
            scores = [item["score"] for item in rank_results]
            max_diff = scores[0] if scores else 0
            min_diff = scores[-1] if scores else 0
            is_biased_result = max_diff > threshold or min_diff < -threshold
        else:
            is_biased_result = False
            
        return is_biased_result


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

    def rank(self) -> list[dict]:
        """Assign a score to every sensitive group present in the sensitive
        features and rank them from most privileged to most discriminated.

        Returns
        -------
        list[dict]
            List of ranking dictionaries with 'sensitive' and 'score' keys.
        """
        result: dict = {}
        ranking: dict = {}

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
                result.setdefault(group1_key, []).append(tpr_ratio)
                if tpr_ratio == 0:
                    result.setdefault(group2_key, []).append(np.inf)
                else:
                    result.setdefault(group2_key, []).append(1 / tpr_ratio)
            else:
                group1_key = tuple(fpr_result["group_1"])
                group2_key = tuple(fpr_result["group_2"])
                result.setdefault(group1_key, []).append(fpr_ratio)
                if fpr_ratio == 0:
                    result.setdefault(group2_key, []).append(np.inf)
                else:
                    result.setdefault(group2_key, []).append(1 / fpr_ratio)

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

        # Convert to the expected format
        ranking_list = []
        for group_tuple, ratio in ranking.items():
            ranking_list.append({
                "sensitive": list(group_tuple),
                "score": float(ratio)
            })

        return ranking_list

    def is_biased(self, threshold: float = 0.1) -> bool:
        """Return a decision of whether there is bias or not
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
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in range [0, 1]")

        rank_results = self.rank()

        if rank_results:
            scores = [item["score"] for item in rank_results]
            min_ratio = scores[0] if scores else 1.0
            max_ratio = scores[-1] if scores else 1.0
            is_biased_result = max_ratio > (1 / threshold) or min_ratio < threshold
        else:
            is_biased_result = False
            
        return is_biased_result
