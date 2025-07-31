import logging
from abc import ABC
from itertools import chain, combinations

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
from fair_mango.metrics.constants import DEFAULT_BIAS_THRESHOLDS
from fair_mango.typing import (
    DemographicParitySummaryResult,
    DisparateImpactSummaryResult,
    EqualOpportunitySummaryResult,
    EqualisedOddsSummaryResult,
    FalsePositiveRateSummaryResult,
    SupersetBiasResult,
    SupersetFairnessRankingResult,
    SupersetFairnessSummaryResult,
    SupersetPerformanceMetricsResult,
)

# Configure logger
logger = logging.getLogger(__name__)


class MetricCalculationError(Exception):
    """Raised when a metric calculation fails for a specific dataset pair."""
    
    def __init__(self, metric_name: str, pair: str, original_error: Exception):
        self.metric_name = metric_name
        self.pair = pair
        self.original_error = original_error
        super().__init__(f"Failed to calculate {metric_name} for {pair}: {original_error}")


class Superset(ABC):
    """An abstract that gets inhereted by other superset classes.

    Parameters
    ----------
    data : Dataset | pd.DataFrame
        The dataset containing the data to be evaluated. If a DataFrame object
        is passed, it should contain attributes `sensitive`, `real_target`,
        `predicted_target`, and `positive_target`.
    sensitive : Sequence[str], optional
        Sequence of sensitive attributes (Ex: gender, race...), by default []
    real_target : Sequence[str] | None, optional.
        Sequence of column names of actual labels for target variables,
        by default None.
    predicted_target : Sequence[str] | None, optional
        Sequence of column names of predicted labels for target variables,
        by default None.
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        Sequence of the positive labels corresponding to the provided
        targets, by default None.

    Raises
    ------
    AttributeError
        If data is a pandas dataframe and 'sensitive_group' parameter is not provided.
    """

    def __init__(
        self,
        data: Dataset,
    ) -> None:
        sensitive = data.sensitive
        real_target = data.real_target
        predicted_target = data.predicted_target
        positive_target = data.positive_target
        df = data.df
        if predicted_target == []:
            predicted_target = None
        if sensitive is None:
            raise AttributeError(
                "'sensitive_group' attribute is required when data is pandas dataframe"
            )

        pairs = list(chain.from_iterable(
            combinations(sensitive, r) for r in range(1, len(sensitive) + 1)
        ))

        self.df = df
        self.sensitive = sensitive
        self.real_target = real_target
        self.predicted_target = predicted_target
        self.positive_target = positive_target
        self.pairs = pairs


class SupersetFairnessMetrics(Superset):
    """Calculate fairness metrics score for all combinations of sensitive
    attributes and ranks them. This class computes all applicable fairness metrics across different
    subsets of sensitive attributes. Ex:
    [gender, race] → (gender), (race), (gender, race)

    Parameters
    ----------
    data : Dataset
        The dataset containing the data to be evaluated. It should contain
        attributes `sensitive`, `real_target`, `predicted_target`, and `positive_target`.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...         'gender': ['male', 'male', 'male', 'female', 'female'],
    ...         'race': ['white', 'black', 'black', 'white', 'white'],
    ...         'real_churn': [1,1,0,0,1],
    ...         'pred_churn': [0,1,0,0,1]
    ... })
    >>> dataset = Dataset(
    ...     df=df,
    ...     sensitive=['gender', 'race'],
    ...     real_target=['real_churn'],
    ...     predicted_target=['pred_churn'],
    ...     positive_target=[1]
    ... )
    >>> super_set_fairness_metrics = SupersetFairnessMetrics(data=dataset)
    >>> result = super_set_fairness_metrics.rank()
    """

    def __init__(
        self,
        data: Dataset,
    ) -> None:
        super().__init__(data)

        self._dataset_metrics: dict[str, type] = {
            "demographic_parity_difference": DemographicParityDifference,
        }

        self._model_metrics: dict[str, type] = {
            "demographic_parity_ratio": DemographicParityRatio,
            "disparate_impact_difference": DisparateImpactDifference,
            "disparate_impact_ratio": DisparateImpactRatio,
            "equal_opportunity_difference": EqualOpportunityDifference,
            "equal_opportunity_ratio": EqualOpportunityRatio,
            "equalised_odds_difference": EqualisedOddsDifference,
            "equalised_odds_ratio": EqualisedOddsRatio,
            "false_positive_rate_difference": FalsePositiveRateDifference,
            "false_positive_rate_ratio": FalsePositiveRateRatio,
        }

    def rank(self) -> list[SupersetFairnessRankingResult]:
        """Calculate fairness metrics rankings for all combinations of sensitive
        attributes and all applicable fairness metrics.

        Returns
        -------
        list[SupersetFairnessRankingResult]
            A list of SupersetFairnessRankingResult objects, each containing:
            - sensitive_group: List of sensitive attribute names for this combination
            - rankings: Dictionary mapping metric names to their ranking results
        """
        results = []

        for pair in self.pairs:
            dataset = Dataset(
                self.df,
                list(pair),
                self.real_target,
                self.predicted_target,
                self.positive_target,
            )

            rankings = {}

            for metric_name, metric_class in self._dataset_metrics.items():
                try:
                    metric = metric_class(dataset)
                    rankings[metric_name] = metric.rank()
                except MetricCalculationError:
                    raise
                except Exception as e:
                    logger.warning("Could not calculate %s for %s: %s", metric_name, pair, e)
                    continue

            if self.predicted_target is not None:
                for metric_name, metric_class in self._model_metrics.items():
                    try:
                        metric = metric_class(dataset)
                        rankings[metric_name] = metric.rank()
                    except MetricCalculationError:
                        raise
                    except Exception as e:
                        logger.warning("Could not calculate %s for %s: %s", metric_name, pair, e)
                        continue

            results.append(
                SupersetFairnessRankingResult(
                    sensitive_group=list(pair),
                    rankings=rankings,
                )
            )

        return results

    def summary(self) -> list[SupersetFairnessSummaryResult]:
        """Calculate fairness metrics summaries for all combinations of sensitive
        attributes and all applicable fairness metrics.

        Returns
        -------
        list[SupersetFairnessSummaryResult]
            A list of SupersetFairnessSummaryResult objects, each containing:
            - sensitive_group: List of sensitive attribute names for this combination
            - summaries: Dictionary mapping metric names to their summary results
        """
        results = []

        for pair in self.pairs:
            dataset = Dataset(
                self.df,
                list(pair),
                self.real_target,
                self.predicted_target,
                self.positive_target,
            )

            summaries: dict[
                str,
                DemographicParitySummaryResult
                | DisparateImpactSummaryResult
                | EqualOpportunitySummaryResult
                | EqualisedOddsSummaryResult
                | FalsePositiveRateSummaryResult,
            ] = {}

            for metric_name, metric_class in self._dataset_metrics.items():
                try:
                    metric = metric_class(dataset)
                    summaries[metric_name] = metric.summary()
                except MetricCalculationError:
                    raise
                except Exception as e:
                    logger.warning("Could not calculate %s for %s: %s", metric_name, pair, e)
                    continue

            if self.predicted_target is not None:
                for metric_name, metric_class in self._model_metrics.items():
                    try:
                        metric = metric_class(dataset)
                        summaries[metric_name] = metric.summary()
                    except MetricCalculationError:
                        raise
                    except Exception as e:
                        logger.warning("Could not calculate %s for %s: %s", metric_name, pair, e)
                        continue

            results.append(
                SupersetFairnessSummaryResult(
                    summaries=summaries,
                )
            )

        return results

    def is_biased(
        self, thresholds: dict[str, float] | None = None
    ) -> list[SupersetBiasResult]:
        """Determine bias for all combinations of sensitive attributes and all
        applicable fairness metrics.

        Parameters
        ----------
        thresholds : dict[str, float] | None, optional
            Dictionary mapping metric names to their bias thresholds.
            If None, uses default thresholds for each metric.

        Returns
        -------
        list[SupersetBiasResult]
            A list of SupersetBiasResult objects, each containing:
            - sensitive_group: List of sensitive attribute names for this combination
            - bias_results: Dictionary mapping metric names to their bias decisions
        """
        if thresholds is None:
            thresholds = {}

        effective_thresholds = DEFAULT_BIAS_THRESHOLDS | thresholds

        results = []

        for pair in self.pairs:
            dataset = Dataset(
                self.df,
                list(pair),
                self.real_target,
                self.predicted_target,
                self.positive_target,
            )

            bias_results = {}

            for metric_name, metric_class in self._dataset_metrics.items():
                try:
                    metric = metric_class(dataset)
                    threshold = effective_thresholds[metric_name]
                    bias_results[metric_name] = metric.is_biased(threshold)
                except MetricCalculationError:
                    raise
                except Exception as e:
                    logger.warning("Could not calculate %s for %s: %s", metric_name, pair, e)
                    continue

            if self.predicted_target is not None:
                for metric_name, metric_class in self._model_metrics.items():
                    try:
                        metric = metric_class(dataset)
                        threshold = effective_thresholds[metric_name]
                        bias_results[metric_name] = metric.is_biased(threshold)
                    except MetricCalculationError:
                        raise
                    except Exception as e:
                        logger.warning("Could not calculate %s for %s: %s", metric_name, pair, e)
                        continue

            results.append(
                SupersetBiasResult(
                    sensitive_group=list(pair),
                    bias_results=bias_results,
                )
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
        Sequence of sensitive attributes (Ex: gender, race...), by default [].
    real_target : Sequence[str] | None, optional
        Sequence of column names of actual labels for target variables,
        by default None.
    predicted_target : Sequence[str] | None, optional
        Sequence of column names of predicted labels for target variables,
        by default None.
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        Sequence of the positive labels corresponding to the provided
        targets, by default None.
    """

    def __init__(
        self,
        data: Dataset,
    ):
        super().__init__(data)
        self.metrics = [SelectionRate, PerformanceMetric, ConfusionMatrix]

    def evaluate(self) -> list[SupersetPerformanceMetricsResult]:
        """Calculate performance evaluation metrics for different subsets of
        sensitive attributes. Ex:
        [gender, race] → (gender), (race), (gender, race)

        Returns
        -------
        list[SupersetPerformanceMetricsResult]
            A list of SupersetPerformanceMetricsResult dictionaries, , each containing the sensitive attributes
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
            {'sensitive_group': ('gender',),
            'result': (['real_churn'],
            [
                {
                    'sensitive_group': array(['male'], dtype=object),
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
                    'sensitive_group': array(['female'], dtype=object),
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
            dataset = self._create_dataset_for_pair(pair)
            concatenated_results = self._initialize_base_results(dataset)
            self._process_metrics_for_dataset(dataset, concatenated_results)
            
            results.append(
                SupersetPerformanceMetricsResult(
                    sensitive_group=pair,
                    data=concatenated_results,
                )
            )

        return results
    
    def _create_dataset_for_pair(self, pair):
        """Create a Dataset instance for a given pair of sensitive attributes."""
        return Dataset(
            self.df,
            list(pair),
            self.real_target,
            self.predicted_target,
            self.positive_target,
        )
    
    def _initialize_base_results(self, dataset):
        """Initialize base results with selection rate data."""
        concatenated_results = SelectionRate(
            dataset,
            use_y_true=True,
        )()
        
        for group_result in concatenated_results:
            group_result.selection_rate_in_data = group_result.data
            
        return concatenated_results
    
    def _process_metrics_for_dataset(self, dataset, concatenated_results):
        """Process all metrics for a given dataset and update concatenated results."""
        for metric in self.metrics:
            if metric is SelectionRate:
                self._process_selection_rate_metric(dataset, concatenated_results)
            else:
                self._process_other_metric(metric, dataset, concatenated_results)
    
    def _process_selection_rate_metric(self, dataset, concatenated_results):
        """Process SelectionRate metric specifically."""
        result = SelectionRate(
            dataset,
            use_y_true=False,
        )()
        
        for group_result in result:
            group_result.selection_rate_in_predictions = group_result.data
            
        for concatenated_result, res in zip(concatenated_results, result):
            self._merge_metric_results(concatenated_result, res)
    
    def _process_other_metric(self, metric, dataset, concatenated_results):
        """Process non-SelectionRate metrics."""
        result = metric(dataset)()
        
        for concatenated_result, res in zip(concatenated_results, result):
            self._merge_metric_results(concatenated_result, res)
    
    def _merge_metric_results(self, concatenated_result, res):
        """Merge individual metric results into concatenated results."""
        if hasattr(res, "metrics"):
            self._merge_metrics_dict(concatenated_result, res)
        elif hasattr(res, "selection_rate_in_predictions"):
            concatenated_result.selection_rate_in_predictions = (
                res.selection_rate_in_predictions
            )
        else:
            self._merge_other_attributes(concatenated_result, res)
    
    def _merge_metrics_dict(self, concatenated_result, res):
        """Merge metrics dictionary into concatenated result."""
        for key, value in res.metrics.items():
            setattr(concatenated_result, key, value)
    
    def _merge_other_attributes(self, concatenated_result, res):
        """Merge other attributes from result into concatenated result."""
        for attr_name in dir(res):
            if (
                not attr_name.startswith("_")
                and attr_name != "sensitive_group"
            ):
                attr_value = getattr(res, attr_name)
                if not callable(attr_value):
                    setattr(concatenated_result, attr_name, attr_value)
