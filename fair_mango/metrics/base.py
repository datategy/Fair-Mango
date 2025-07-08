from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd

from fair_mango.dataset.dataset import Dataset
from fair_mango.typing import (
    FairnessSummaryResult,
    FairnessRatioSummaryResult,
    RankResult,
)


def is_binary(y: pd.Series) -> bool:
    """Check if a data contains binary values (0/1) or can be treated as binary.
    
    Parameters
    ----------
    y : pd.Series
        Input data.

    Returns
    -------
    bool
        True if data contains binary values or can be treated as binary, False otherwise.
    """
    unique_values = y.unique()
    
    if y.nunique() == 2:
        return True
    
    elif y.nunique() == 1 and len(unique_values) == 1:
        single_value = unique_values[0]
        if single_value in [0, 1]:
            return True
    
    return False


def encode_target(data: Dataset, col: str | Hashable) -> None:
    """Encode target as [0,1]
    
    Parameters
    ----------
    data : Dataset
        Dataset object.
    col : str | Hashable
        Column name.

    Raises
    ------
    ValueError
        If the positive target parameter was not provided when creating the dataset.
    KeyError
        If the positive target value does not exist in the column.
    """
    unique_values = data.df[col].unique()
    
    if data.positive_target is None:
        raise ValueError(
            f"Calculations failed because target '{col}' has values different "
            "than [0,1]. Provide the positive_target parameter when creating "
            "the dataset to solve this issue."
        )
    else:
        if data.positive_target in unique_values:
            
            mapping = {data.positive_target: 1}
            data.df[col] = data.df[col].map(mapping).fillna(0).astype(int)
        else:
            
            if len(unique_values) == 1 and unique_values[0] not in [0, 1]:
                single_value = unique_values[0]
                if single_value == data.positive_target:
                    data.df[col] = 1
                else:
                    data.df[col] = 0
            else:
                raise KeyError(
                    "Positive target value provided does not exist in the column. "
                    f"{data.positive_target} does not exist in column {col}: "
                    f"{unique_values}"
                )
def false_negative_rate(fn: int, tp: int, **_) -> float:
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


def false_positive_rate(tn: int, fp: int, **_) -> float:
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


def true_negative_rate(tn: int, fp: int, **_) -> float:
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


def true_positive_rate(fn: int, tp: int, **_) -> float:
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
    data : Dataset
        Input data.

    Raises
    ------
    ValueError
        - If data is a DataFrame and the parameters 'sensitive' and 'real_target'
          are not provided.
        - If the target variable is not binary (has two unique values).
    """

    def __init__(self, data: Dataset) -> None:
        self.data = data
        self.predicted_target_by_group = []
        y: pd.Series = self.data.df[self.data.real_target]

        if is_binary(y):
            if (np.unique(y) != [0, 1]).all():
                encode_target(self.data, y.name)
            self.real_target_by_group = self.data.get_real_target_for_all_groups()

            if self.data.predicted_target is not None:
                y = self.data.df[self.data.predicted_target]
                if is_binary(y) and (np.unique(y) != [0, 1]).all():
                    encode_target(self.data, y.name)
                self.predicted_target_by_group = (
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
    def __call__(self):
        pass


def calculate_disparity(
    result_per_groups: list[dict], method: Literal["difference", "ratio"]
) -> list:
    """Calculate the disparity in the scores between every possible pair in
    the provided groups using two available methods:
    - difference (Example: for three groups a, b, c:
      `[score_a - score_b], [score_a - score_c], [score_b - score_c]`).
    - ratio (Example: for three groups a, b, c:
      `[score_a / score_b], [score_a / score_c], [score_b / score_c]`).

    Parameters
    ----------
    result_per_groups : list[dict]
        List of dictionaries with the sensitive group and the corresponding
        score.
    method : Literal['difference', 'ratio']
        Method used to calculate the disparity. Either 'difference' or 'ratio'.

    Returns
    -------
    list[dict]
        A list of dictionaries with JSON-compatible structure:
        - group_1: list[str] with the first group's sensitive attributes
        - group_2: list[str] with the second group's sensitive attributes  
        - disparity: float with the calculated disparity value

    Raises
    ------
    AttributeError
        If method is not 'difference' or 'ratio'.
    """
    result = []
    pairs = combinations(range(len(result_per_groups)), 2)

    for i, j in pairs:
        group_i = result_per_groups[i]["sensitive"]
        group_j = result_per_groups[j]["sensitive"]
        
        if not isinstance(group_i, list):
            group_i = [str(x) for x in group_i]
        if not isinstance(group_j, list):
            group_j = [str(x) for x in group_j]

        key_i = 'data' if 'data' in result_per_groups[i] else 'result'
        key_j = 'data' if 'data' in result_per_groups[j] else 'result'
        
        data_i = result_per_groups[i][key_i]
        data_j = result_per_groups[j][key_j]
        
        if isinstance(data_i, pd.Series):
            if len(data_i) == 1:
                result_i = float(data_i.iloc[0])
            else:
                result_i = float(data_i.mean())
        elif isinstance(data_i, (list, np.ndarray)):
            if len(data_i) == 1:
                result_i = float(data_i[0])
            else:
                result_i = float(np.mean(data_i))
        else:
            result_i = float(data_i)
            
        if isinstance(data_j, pd.Series):
            if len(data_j) == 1:
                result_j = float(data_j.iloc[0])
            else:
                result_j = float(data_j.mean())
        elif isinstance(data_j, (list, np.ndarray)):
            if len(data_j) == 1:
                result_j = float(data_j[0])
            else:
                result_j = float(np.mean(data_j))
        else:
            result_j = float(data_j)

        if method == "difference":
            disparity_value = result_i - result_j
        elif method == "ratio":
            disparity_value = result_i / result_j if result_j != 0 else float('inf')
        else:
            raise AttributeError(
                f"method {method} not recognised. Use 'difference' or 'ratio' instead."
            )

        result.append({
            "group_1": group_i,
            "group_2": group_j,
            "disparity": float(disparity_value)
        })

    return result


class FairnessMetricDifference(ABC):
    """An abstract class that is inherited by every fairness metric that is
    based on the 'difference' to calculate disparity between the sensitive
    groups present in the sensitive feature.

    Parameters
    ----------
    data : Dataset
        Input data.
    metric : type[Metric]
        A sequence of metrics or a dictionary with keys being custom labels
        and values a callable that calculates the score.
    label : str
        The key to give to the result in the different returned dictionaries.
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
        data: Dataset,
        metric: type[Metric],
        metric_type: str = "performance",
        **kwargs,
    ) -> None:
        self.metric = metric
        self.kwargs = kwargs
        self.target: str
        self.metric_results: list
        self.metric_type = metric_type
        self.data = data
        if metric_type == "performance":
            self.label1 = "privileged"
            self.label2 = "unprivileged"
        elif metric_type == "error":
            self.label1 = "unprivileged"
            self.label2 = "privileged"
        else:
            raise AttributeError(
                "Metric type not recognized. accepted values 'performance' or 'error'"
            )

        self.result: dict | None = None
        self.ranking: dict | None = None
        self.results: list[dict] | None = None

    def _compute(self) -> list[dict]:
        """Calculate the disparity in the scores between every possible pair in
        the provided groups.

        Returns
        -------
        dict[tuple, np.ndarray[float]]
            A dictionary with:
            - keys: tuple with the pair of the sensitive groups labels.
            - values: a numpy array with the corresponding disparity.
        """
       
        filtered_kwargs = {k: v for k, v in self.kwargs.items() if k != 'label'}
        metric = self.metric(self.data, **filtered_kwargs)
        self.target, self.metric_results = metric()
        results = calculate_disparity(self.metric_results, "difference")
        
       
        self.differences = (self.target, results)
        
        return results

    def summary(self) -> FairnessSummaryResult:
        """Return the fairness metric value, in other words the biggest
        disparity found with specifying the privileged and discriminated
        groups.

        Returns
        -------
        FairnessSummaryResult
            A single summary result dictionary with:
            - disparity: The maximum absolute disparity value found
            - privileged_group: List of strings identifying the privileged group
            - unprivileged_group: List of strings identifying the unprivileged group
        """
        if self.results is None:
            self.results = self._compute()
        
        max_disparity = 0.0
        privileged_group = None
        unprivileged_group = None

        for disparity_result in self.results:
            abs_disparity = abs(disparity_result["disparity"])
            
            if abs_disparity > max_disparity:
                max_disparity = abs_disparity
                if disparity_result["disparity"] > 0:
                    privileged_group = disparity_result["group_1"]
                    unprivileged_group = disparity_result["group_2"]
                else:
                    privileged_group = disparity_result["group_2"]
                    unprivileged_group = disparity_result["group_1"]

        label = self.kwargs.get('label', getattr(self, 'label', 'disparity'))
        
        # Return the JSON-compatible single summary result
        return {
            label: max_disparity,
            "privileged_group": privileged_group,
            "unprivileged_group": unprivileged_group,
        }

    def rank(self) -> dict[str, list[RankResult]]:
        """Assign a score to every sensitive group present in the sensitive
        features and rank them from most privileged to most discriminated.
        The score can be interpreted like:
        - [('Male',): 0.0314]: Males have on average a score higher by 3.14% than
          the Females.
        - [('White',): -0.0628]: Whites have on average a score lower by 6.28% than
          other groups (Black, Asian...).

        Returns
        -------
        dict[str, list[RankResult]]
            A dictionary with:
            - keys: name of the target variable.
            - values: a list of RankResult dictionaries with sensitive group and score.
        """
        result: dict = {}
        ranking: dict[tuple, float] = {}

        if self.results is None:
            self.results = self._compute()


        for disparity_result in self.results:
            group1_key = tuple(disparity_result["group_1"])
            group2_key = tuple(disparity_result["group_2"])
            disparity_value = disparity_result["disparity"]
            
            if self.metric_type == "performance":
                result.setdefault(group1_key, []).append(disparity_value)
                result.setdefault(group2_key, []).append(-disparity_value)
            elif self.metric_type == "error":
                result.setdefault(group1_key, []).append(-disparity_value)
                result.setdefault(group2_key, []).append(disparity_value)
        
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

       
        ranking_str = {}
        for group, score in ranking.items():
            group_str = tuple(str(x) for x in group)
            ranking_str[group_str] = score

       
        rank_results = []
        for group_tuple, score in ranking.items():
            rank_results.append({
                "sensitive": list(group_tuple),
                "score": float(score)
            })

        return {self.data.real_target: rank_results}

    def is_biased(self, threshold: float = 0.1) -> dict[str, bool]:
        """Return a decision of whether there is bias or not
        depending on the provided threshold.

        Parameters
        ----------
        threshold : float, optional
            The threshold to make the decision of whether there is bias or not,
            by default 0.1.

        Returns
        -------
        dict[str, bool]
            Dictionary with target name as key and bias decision as value.

        Raises
        ------
        ValueError
            If threshold parameter is not in the range of [0, 1].
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in range [0, 1]")

        ranking_nested = self.rank()
        
        if not ranking_nested:
            return {self.data.real_target: False}

        
        ranking = list(ranking_nested.values())[0]
        
        if not ranking:
            return {self.data.real_target: False}

        scores = [item["score"] for item in ranking]
        max_diff = scores[0] if scores else 0
        min_diff = scores[-1] if scores else 0
        is_biased = max_diff > threshold or min_diff < -threshold
        
        
        return {self.data.real_target: is_biased}


class FairnessMetricRatio(ABC):
    """An abstract class that is inherited by every fairness metric that is
    based on the 'ratio' to calculate disparity between the sensitive groups
    present in the sensitive feature.

    Parameters
    ----------
    data : Dataset
        Input data.
    metric : type[Metric]
        A sequence of metrics or a dictionary with keys being custom labels
        and values a callable that calculates the score.
    label : str
        The key to give to the result in the different returned dictionaries.
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
        data: Dataset,
        metric: type[Metric],
        metric_type: str = "performance",
        **kwargs,
    ) -> None:
        self.data = data
        self.metric = metric
        self.label = kwargs.pop('label', 'difference') 

        if metric_type == "performance":
            self.label1 = "privileged"
            self.label2 = "unprivileged"
        elif metric_type == "error":
            self.label1 = "unprivileged"
            self.label2 = "privileged"
        else:
            raise AttributeError(
                "Metric type not recognized. accepted values 'performance' or 'error'"
            )

        self.kwargs = kwargs
        self.metric_type = metric_type
        self.target: Sequence
        self.metric_results: list
        self.result: dict | None = None
        self.ranking: dict | None = None
        self.results: list[dict] | None = None
        self.ratios: tuple | None = None  
        self.differences: tuple | None = None  

    def _compute(self) -> list[dict]:
        """Calculate the disparity in the scores between every possible pair in
        the provided groups.

        Returns
        -------
        list[dict]
            A list of DisparityResult dictionaries with:
            - group_1: First group as list[str]
            - group_2: Second group as list[str] 
            - disparity: The ratio value between groups
        """
        
        filtered_kwargs = {k: v for k, v in self.kwargs.items() if k != 'label'}
        metric = self.metric(self.data, **filtered_kwargs)
        self.target, self.metric_results = metric()
        results = calculate_disparity(self.metric_results, "ratio")
        
        return results

    def summary(self) -> FairnessRatioSummaryResult:
        """Return the fairness metric value, in other words the biggest
        disparity found with specifying the privileged and discriminated
        groups.

        Returns
        -------
        FairnessRatioSummaryResult
            A single summary result dictionary with:
            - ratio: The minimum ratio value found (closest to 0)
            - privileged_group: List of strings identifying the privileged group
            - unprivileged_group: List of strings identifying the unprivileged group
        """
        if self.results is None:
            self.results = self._compute()

        min_ratio = 1.0
        privileged_group = None
        unprivileged_group = None


        for disparity_result in self.results:
            ratio_value = disparity_result["disparity"]
            

            if ratio_value > 1:
                adjusted_ratio = 1 / ratio_value
                temp_privileged = disparity_result["group_1"] 
                temp_unprivileged = disparity_result["group_2"]
            else:
                adjusted_ratio = ratio_value
                temp_privileged = disparity_result["group_2"]  
                temp_unprivileged = disparity_result["group_1"] 

            if adjusted_ratio < min_ratio:
                min_ratio = adjusted_ratio
                privileged_group = temp_privileged
                unprivileged_group = temp_unprivileged

        label = self.kwargs.get('label', getattr(self, 'label', 'ratio'))
        
        return {
            label: min_ratio,
            "privileged_group": privileged_group,
            "unprivileged_group": unprivileged_group,
        }

    def rank(self) -> dict[str, list[RankResult]]:
        """Assign a score to every sensitive group present in the sensitive
        features and rank them from most privileged to most discriminated.
        The score can be interpreted like:
        - [('Male',): 0.814]: Males have on average 81.4% the score of the
          Females.
        - [('White',): 1.20]: Whites have on average 120% the score of the
          other groups (Black, Asian...).

        Returns
        -------
        dict[str, list[RankResult]]
            A dictionary with target name as key and list of RankResult as values.
        """
        result: dict = {}
        ranking: dict[tuple, float] = {}

        if self.results is None:
            self.results = self._compute()

        for disparity_result in self.results:
            group1_key = tuple(disparity_result["group_1"])
            group2_key = tuple(disparity_result["group_2"])
            ratio_value = disparity_result["disparity"]
            
            if self.metric_type == "performance":
                result.setdefault(group1_key, []).append(np.inf if ratio_value == 0 else 1 / ratio_value)
                result.setdefault(group2_key, []).append(ratio_value)
            elif self.metric_type == "error":
                result.setdefault(group1_key, []).append(ratio_value)
                result.setdefault(group2_key, []).append(np.inf if ratio_value == 0 else 1 / ratio_value)
        
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

        rank_results = []
        for group_tuple, score in ranking.items():
            rank_results.append({
                "sensitive": list(group_tuple),
                "score": float(score)
            })

        return {self.data.real_target: rank_results}

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

        ranking_nested = self.rank()

        if not ranking_nested:
            return {self.data.real_target: False}

        ranking = list(ranking_nested.values())[0]
        
        if not ranking:
            return {self.data.real_target: False}

        scores = [item["score"] for item in ranking]
        min_ratio = scores[0] if scores else 1.0
        max_ratio = scores[-1] if scores else 1.0
        is_biased = max_ratio > (1 / threshold) or min_ratio < threshold
        
        return {self.data.real_target: is_biased}
