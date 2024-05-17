from typing import Sequence, Callable, Collection, Mapping
from sklearn.metrics import confusion_matrix
import numpy as np
from fair_mango.dataset.dataset import Dataset
import pandas as pd

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
        if y.nunique()==2:
            return True
        else:
            return False
    except ValueError:
        if (y.nunique()==2).all():
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
        raise ValueError(f"Calculations failed because target '{col}' has values \
different than [0,1]. Provide the positive_target parameter when creating the dataset to solve this issue.")
    else:
        if data.positive_target[ind] in data.df[col].unique():
            mapping = {data.positive_target[ind]: 1}
            data.df[col] = data.df[col].map(mapping).fillna(0).astype(int)
        else:
            raise KeyError(f"Positive target value provided does not exist in the column. {data.positive_target[ind]} does not exist in column {col}: {data.df[col].unique()}")

def false_negative_rate(fn: int, tp: int, zero_division: float | str | None, **_) -> float | str | None:
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

def false_positive_rate(tn: int, fp: int, zero_division: float | str | None, **_) -> float | str | None:
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

def true_negative_rate(tn: int, fp: int, zero_division: float | str | None, **_) -> float | str | None:
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
    
def true_positive_rate(fn: int, tp: int, zero_division: float | str | None, **_) -> float | str | None:
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
    """A class for the different fairness metrics and performance evaluation in different groups
    """
    def __init__(self, data: Dataset):
        self.data = data
        self.predicted_targets_by_group = None
        y = self.data.df[data.real_target]
        if len(data.real_target)>1:
            y = y.squeeze()
        if is_binary(y):
            for ind, col in enumerate(y):
                if (np.unique(y[col])!=[0,1]).all():
                    encode_target(self.data, ind, col)
            self.real_targets_by_group = self.data.get_real_target_for_all_groups()
            if self.data.predicted_target is not None:
                self.predicted_targets_by_group = self.data.get_predicted_target_for_all_groups()
            self.results = []
        else:
            raise(ValueError(f"target variable needs to be binary. Found {y.nunique()} unique values"))

    
    def __call__(self):
        ...

        
class SelectionRate(Metric):
    """Calculate selection rate for different sensitive groups
    """
    def __init__(self, data: Dataset, use_y_true: bool = False):
        super().__init__(data)
        self.use_y_true = use_y_true
    
    def __call__(self):     
        if self.use_y_true:
            targets = self.data.real_target
            targets_by_group = self.real_targets_by_group
        else:
            if self.predicted_targets_by_group is None:
                raise ValueError("No predictions found, provide predicted_target parameter when creating the dataset or set use_y_true to True to use the real labels")
            targets = self.data.predicted_target
            targets_by_group = self.predicted_targets_by_group
        for group in targets_by_group:
            group_ = group['sensitive']
            y_group = group['data']
            self.results.append({"sensitive": group_, "result": np.array(y_group.mean())})
        return targets, self.results


class ConfusionMatrix(Metric):
    """Calculate
    - false positive rate
    - false negative rate
    - true positive rate
    - true negative rate
    """
    def __init__(self, data: Dataset, metrics: Collection | None = None, zero_division: float | str | None = None) -> None:
        super().__init__(data)
        if self.predicted_targets_by_group is None:
            raise ValueError("No predictions found, provide predicted_target parameter when creating the dataset")
        self.zero_division = zero_division
        if metrics is None:
            self.metrics = {'false_negative_rate': false_negative_rate,
                            'false_positive_rate': false_positive_rate,
                            'true_negative_rate': true_negative_rate,
                            'true_positive_rate': true_positive_rate,
                            }
        else:
            if isinstance(metrics, (dict, Mapping)):
                self.metrics = metrics
            else:
                metrics = set(metrics)
                self.metrics = {}
                for metric in metrics:
                    self.metrics[metric.__name__] = metric
    
    def __call__(self) -> tuple[list]:
        for real_group, predicted_group in zip(self.real_targets_by_group, self.predicted_targets_by_group):
            group_ = real_group['sensitive']
            real_y_group = real_group['data']
            predicted_y_group = predicted_group['data']
            result_for_group = {'sensitive': group_}
            for real_col, predicted_col in zip(self.data.real_target, self.data.predicted_target):
                if len(self.data.real_target)!=1:
                    real_values = real_y_group[real_col]
                    predicted_values = predicted_y_group[predicted_col]
                else:
                    real_values = real_y_group
                    predicted_values = predicted_y_group
                if (np.unique(real_values) == 1).all() and (np.unique(predicted_values) == 1).all():
                    tp = len(real_values)
                    tn = fp = fn = 0
                elif (np.unique(real_values) == 0).all() and (np.unique(predicted_values) == 0).all():
                    tn = len(real_values)
                    tp = fp = fn = 0
                elif (np.unique(real_values) == 1).all() and (np.unique(predicted_values) == 0).all():
                    fn = len(real_values)
                    fp = tn = tp = 0
                elif (np.unique(real_values) == 0).all() and (np.unique(predicted_values) == 1).all():
                    fp = len(real_values)
                    fn = tn = tp = 0
                else:
                    conf_matrix = confusion_matrix(real_values, predicted_values)
                    tn = conf_matrix[0, 0]
                    tp = conf_matrix[1, 1]
                    fn = conf_matrix[1, 0]
                    fp = conf_matrix[0, 1]
                for metric_name, metric in self.metrics.items():
                    result_for_group.setdefault(metric_name, []).append(metric(tn=tn, fp=fp, fn=fn, tp=tp, zero_division=self.zero_division))
            self.results.append(result_for_group)
        return self.data.real_target, self.results
