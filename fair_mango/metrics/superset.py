from abc import ABC
from collections.abc import Sequence
from itertools import chain, combinations

import pandas as pd

from fair_mango.dataset.dataset import Dataset
from fair_mango.metrics.metrics import (
    ConfusionMatrix,
    DemographicParityDifference,
    DemographicParityRatio,
    DisparateImpactDifference,
    DisparateImpactRatio,
    EqualisedOddsDifference,
    EqualisedOddsRatio,
    EqualOpportunityDifference,
    EqualOpportunityRatio,
    FalsePositiveRateDifference,
    FalsePositiveRateRatio,
    PerformanceMetric,
    SelectionRate,
)


class Superset(ABC):
    """An abstract that gets inhereted by other superset classes

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        The dataset containing the data to be evaluated. If a DataFrame object
        is passed, it should contain attributes `sensitive`, `real_target`,
        `predicted_target`, and `positive_target`.
    sensitive : Sequence[str], optional
        A Sequence of sensitive attributes (Ex: gender, race...), by default []
    real_target : Sequence[str] | None, optional
        A Sequence of column names of actual labels for target variables,
        by default None
    predicted_target : Sequence[str] | None, optional
        A Sequence of column names of predicted labels for target variables,
        by default None
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        A Sequence of the positive labels corresponding to the provided
        targets, by default None

    Raises
    ------
    AttributeError
        if data is a pandas dataframe and 'sensitive' parameter is not provided."""

    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        if isinstance(data, Dataset):
            sensitive = data.sensitive
            real_target = data.real_target
            predicted_target = data.predicted_target
            positive_target = data.positive_target
            data = data.df
            if predicted_target == []:
                predicted_target = None

        if sensitive is None:
            raise AttributeError(
                "'sensitive' attribute is required when data is pandas dataframe"
            )

        pairs = chain.from_iterable(
            combinations(sensitive, r) for r in range(1, len(sensitive) + 1)
        )

        self.data = data
        self.sensitive = sensitive
        self.real_target = real_target
        self.predicted_target = predicted_target
        self.positive_target = positive_target
        self.pairs = pairs


class SupersetFairnessMetrics(Superset):
    def __init__(
        self,
        metric: (
            type[DemographicParityDifference]
            | type[DemographicParityRatio]
            | type[DisparateImpactDifference]
            | type[DisparateImpactRatio]
            | type[EqualOpportunityDifference]
            | type[EqualOpportunityRatio]
            | type[EqualisedOddsDifference]
            | type[EqualisedOddsRatio]
            | type[FalsePositiveRateDifference]
            | type[FalsePositiveRateRatio]
        ),
        data: Dataset | pd.DataFrame,
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        """Calculate fairness metrics score for different subsets of sensitive
        attributes and ranks them. Ex:
        [gender, race] → (gender), (race), (gender, race)

        Parameters
        ----------
        metric : type[DemographicParityDifference]  |  type[DemographicParityRatio]
        |  type[DisparateImpactDifference]  |  type[DisparateImpactRatio]  |
        type[EqualOpportunityDifference]  |  type[EqualOpportunityRatio]  |
        type[EqualisedOddsDifference]  |  type[EqualisedOddsRatio]  |
        type[FalsePositiveRateDifference]  |  type[FalsePositiveRateRatio]
            The fairness metric class to be used for evaluation
        data : Dataset | pd.DataFrame
            The dataset containing the data to be evaluated. If a DataFrame object
            is passed, it should contain attributes `sensitive`, `real_target`,
            `predicted_target`, and `positive_target`.
        sensitive : Sequence[str], optional
            A Sequence of sensitive attributes (Ex: gender, race...), by default []
        real_target : Sequence[str] | None, optional
            A Sequence of column names of actual labels for target variables,
            by default None
        predicted_target : Sequence[str] | None, optional
            A Sequence of column names of predicted labels for target variables,
            by default None
        positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
            A Sequence of the positive labels corresponding to the provided
            targets, by default None
        """
        super().__init__(
            data, sensitive, real_target, predicted_target, positive_target
        )
        self.metric = metric

    def rank(self) -> list:
        """Calculate fairness metrics scores for different subsets of sensitive
        attributes and ranks them. Ex:
        [gender, race] → (gender), (race), (gender, race)

        Returns
        -------
        list
            A list of dictionaries, each containing the sensitive attributes
            considered and their corresponding fairness metric result.

        Examples
        --------
        >>> df = pd.DataFrame({
        ...         'gender': ['male', 'male', 'male', 'female', 'female'],
        ...         'race': ['white', 'black', 'black', 'white', 'white'],
        ...         'real_churn': [1,1,0,0,1],
        ...         'pred_churn': [0,1,0,0,1]
        ... })
        >>> super_set_fairness_metrics = SupersetFairnessMetrics(
        ...     metric=DemographicParityDifference,
        ...     data=df,
        ...     sensitive=['gender', 'race'],
        ...     real_target=['real_churn'],
        ...     predicted_target=['pred_churn'],
        ... )
        >>> result = super_set_fairness_metrics.rank()
        >>> result
        [
            {
                'sensitive': ('gender',),
                'result': {
                    'real_churn': {
                        ('male',): 0.16666666666666663,
                        ('female',): -0.16666666666666663
                    }
                }
            },
            {
                'sensitive': ('race',),
                'result': {
                    'real_churn': {
                        ('white',): 0.16666666666666663,
                        ('black',): -0.16666666666666663
                    }
                }
            },
            {
                'sensitive': ('gender', 'race'),
                'result': {
                    'real_churn': {
                        ('male', 'white'): 0.5,
                        ('female', 'white'): -0.25,
                        ('male', 'black'): -0.25
                    }
                }
            }
        ]
        """
        results = []

        for pair in self.pairs:
            result = self.metric(
                data=self.data,
                sensitive=list(pair),
                real_target=self.real_target,
                predicted_target=self.predicted_target,
                positive_target=self.positive_target,
            ).rank()

            results.append(
                {
                    "sensitive": pair,
                    "result": result,
                }
            )

        return results


class SupersetPerformanceMetrics(Superset):
    """Calculate performance evaluation metrics for different subsets of
    sensitive attributes. Ex:
    [gender, race] → (gender), (race), (gender, race)

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        The dataset containing the data to be evaluated. If a DataFrame object
        is passed, it should contain attributes `sensitive`, `real_target`,
        `predicted_target`, and `positive_target`.
    sensitive : Sequence[str], optional
        A Sequence of sensitive attributes (Ex: gender, race...), by default []
    real_target : Sequence[str] | None, optional
        A Sequence of column names of actual labels for target variables,
        by default None
    predicted_target : Sequence[str] | None, optional
        A Sequence of column names of predicted labels for target variables,
        by default None
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        A Sequence of the positive labels corresponding to the provided
        targets, by default None
    """

    def __init__(
        self,
        data: Dataset | pd.DataFrame,
        sensitive: Sequence[str] | None = None,
        real_target: Sequence[str] | None = None,
        predicted_target: Sequence[str] | None = None,
        positive_target: Sequence[int | float | str | bool] | None = None,
    ) -> None:
        super().__init__(
            data, sensitive, real_target, predicted_target, positive_target
        )
        self.metrics = [SelectionRate, PerformanceMetric, ConfusionMatrix]

    def evaluate(self) -> list[dict]:
        """Calculate performance evaluation metrics for different subsets of
        sensitive attributes. Ex:
        [gender, race] → (gender), (race), (gender, race)

        Returns
        -------
        list
            A list of dictionaries, each containing the sensitive attributes
            considered and their corresponding performance evaluation metric
            results.

        Examples
        --------
        >>> df = pd.DataFrame({
        ...         'gender': ['male', 'male', 'male', 'male', 'female', 'female'],
        ...         'race': ['white', 'white', 'black', 'black', 'white', 'white'],
        ...         'real_churn': [1,0,1,0,0,1],
        ...         'pred_churn': [0,0,1,0,0,1]
        ... })
        >>> result = super_set_performance_metrics(
        ...     data=df,
        ...     sensitive=['gender', 'race'],
        ...     real_target=['real_churn'],
        ...     predicted_target=['pred_churn'],
        ... )
        >>> result
        [
            {'sensitive': ('gender',),
            'result': (['real_churn'],
            [
                {
                    'sensitive': array(['male'], dtype=object),
                    'selection_rate_in_data': array(0.5),
                    'selection_rate_in_predictions': array(0.25),
                    'accuracy': [0.75],
                    'balanced accuracy': [0.75],
                    'precision': [1.0],
                    'recall': [0.5],
                    'f1-score': [0.6666666666666666],
                    'false_negative_rate': [0.5],
                    'false_positive_rate': [0.0],
                    'true_negative_rate': [1.0],
                    'true_positive_rate': [0.5]
                },
                {
                    'sensitive': array(['female'], dtype=object),
                    ...
                    'f1-score': [0.0],
                    'false_negative_rate': [1.0],
                    'false_positive_rate': [0.0],
                    'true_negative_rate': [1.0],
                    'true_positive_rate': [0.0]
                }
            ]
            )
            }
        ]
        """
        results = []

        for pair in self.pairs:
            concatenated_results = SelectionRate(
                data=self.data,
                use_y_true=True,
                sensitive=list(pair),
                real_target=self.real_target,
                predicted_target=self.predicted_target,
                positive_target=self.positive_target,
                label="selection_rate_in_data",
            )()

            for metric in self.metrics:
                if metric is SelectionRate:
                    result = SelectionRate(
                        data=self.data,
                        use_y_true=False,
                        sensitive=list(pair),
                        real_target=self.real_target,
                        predicted_target=self.predicted_target,
                        positive_target=self.positive_target,
                        label="selection_rate_in_predictions",
                    )()[1]
                else:
                    result = metric(
                        data=self.data,
                        sensitive=list(pair),
                        real_target=self.real_target,
                        predicted_target=self.predicted_target,
                        positive_target=self.positive_target,
                    )()[1]

                for concatenated_result, res in zip(concatenated_results[1], result):
                    concatenated_result.update(res)

            results.append(
                {
                    "sensitive": pair,
                    "result": concatenated_results,
                }
            )

        return results
