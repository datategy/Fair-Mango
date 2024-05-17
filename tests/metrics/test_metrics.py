from fair_mango.dataset.dataset import Dataset
from fair_mango.metrics.metrics import SelectionRate, ConfusionMatrix, false_positive_rate, false_negative_rate, true_negative_rate
from fair_mango.metrics.metrics import is_binary, encode_target
import pytest
from typing import Sequence, Collection
import pandas as pd
import numpy as np
from _pytest.python_api import RaisesContext


df = pd.read_csv('tests/data/heart_data.csv')

dataset1 = Dataset(df, ['Sex'], ['HeartDisease'])

dataset2 = Dataset(df, ['Sex'], ['HeartDisease'], ['HeartDiseasePred'])

dataset3 = Dataset(df, ['Sex', 'ChestPainType'], ['HeartDisease'], ['HeartDiseasePred'])

dataset4 = Dataset(df, ['Sex'], ['HeartDisease', 'ExerciseAngina'], None, [1, 'Y'])

dataset5 = Dataset(df, ['Sex'], ['HeartDisease', 'ExerciseAngina'], ['HeartDiseasePred', 'ExerciseAngina'], [1, 'Y'])

dataset6 = Dataset(df, ['Sex', 'ChestPainType'], ['HeartDisease', 'ExerciseAngina'], ['HeartDiseasePred', 'ExerciseAngina'], [1, 'Y'])


@pytest.mark.parametrize(
    "y, expected_result",
    [
        (df['Sex'], True),
        (df['ExerciseAngina'], True),
        (df['ChestPainType'], False),
        (df[['ExerciseAngina', 'Sex']], True),
    ],
)
def test_is_binary(y: pd.Series | pd.DataFrame, expected_result: bool):
    assert is_binary(y) == expected_result


@pytest.mark.parametrize(
    "data, ind, col, expected_result",
    [
        (dataset4, 0, 'HeartDisease', None),
        (dataset5, 1, 'ExerciseAngina', None),
        (dataset5, 1, 'HeartDiseasePred', pytest.raises(KeyError)),
        (dataset3, 0, 'ExerciseAngina', pytest.raises(ValueError)),
    ],
)
def test_encode_target(data: Dataset, ind: int, col: str, expected_result: None | RaisesContext):
    if expected_result is None:
        encode_target(data, ind, col)
        assert sorted(data.df[col].unique()) == [0,1]
    else:
        with expected_result:
            encode_target(data, ind, col)


@pytest.mark.parametrize(
    "data, use_y_true, expected_groups, expected_result",
    [
        (dataset1, True, ['M', 'F'], [0.63172414, 0.25906736]),
        (dataset1, False, [], pytest.raises(ValueError)),
        (dataset3, True, [['M', 'ASY'],['M', 'NAP'],['M', 'ATA'],['F', 'ASY'],
                           ['F', 'ATA'],['F', 'NAP'],['M', 'TA'],['F', 'TA']], 
                          [0.8286385, 0.44, 0.17699115, 0.55714286, 0.06666667, 0.11320755, 0.52777778, 0.1]),
        (dataset3, False, [['M', 'ASY'],['M', 'NAP'],['M', 'ATA'],['F', 'ASY'],
                           ['F', 'ATA'],['F', 'NAP'],['M', 'TA'],['F', 'TA']], 
                          [0.81924883, 0.44, 0.17699115, 0.58571429, 0.08333333, 0.0754717, 0.52777778, 0.1]),
        (dataset5, True, ['M', 'F'], [[0.63172414, 0.45241379], [0.25906736, 0.22279793]]),
        (dataset5, False, ['M', 'F'], [[0.6262069 , 0.45241379], [0.2642487 , 0.22279793]]),
    ],
)
def test_selectionrate(data: Dataset, use_y_true: bool, expected_groups: Sequence[str], expected_result: Sequence[float] | RaisesContext):
    if isinstance(expected_result, Sequence):
        sr = SelectionRate(data, use_y_true)
        result = sr()
        if use_y_true:
            assert result[0] == data.real_target
        else:
            assert result[0] == data.predicted_target
        for i, res in enumerate(result[1]):
            assert (res['sensitive'] == expected_groups[i]).all()
            assert (np.isclose(res['result'], expected_result[i], atol=0.0000002)).all()        
    else:
        with expected_result:
            sr = SelectionRate(data, use_y_true)
            sr()


expected_result_2 = [{'sensitive': np.array(['M']),
                    'false_negative_rate': [0.021834061135371178],
                    'false_positive_rate': [0.02247191011235955],
                    'true_negative_rate': [0.9775280898876404],
                    'true_positive_rate': [0.9781659388646288]},
                    {'sensitive': np.array(['F']),
                    'false_negative_rate': [0.06],
                    'false_positive_rate': [0.027972027972027972],
                    'true_negative_rate': [0.972027972027972],
                    'true_positive_rate': [0.94]}]

expected_result_3 = [{'sensitive': np.array(['M', 'ASY']),
                    'fpr': [0.0136986301369863]},
                    {'sensitive': np.array(['M', 'NAP']),
                    'fpr': [0.011904761904761904]},
                    {'sensitive': np.array(['M', 'ATA']),
                    'fpr': [0.010752688172043012]},
                    {'sensitive': np.array(['F', 'ASY']),
                    'fpr': [0.0967741935483871]},
                    {'sensitive': np.array(['F', 'ATA']),
                    'fpr': [0.017857142857142856]},
                    {'sensitive': np.array(['F', 'NAP']), 'fpr': [0.0]},
                    {'sensitive': np.array(['M', 'TA']),
                    'fpr': [0.17647058823529413]},
                    {'sensitive': np.array(['F', 'TA']), 'fpr': [0.0]}]

expected_result_6 = [{'sensitive': np.array(['M', 'ASY']),
                    'true_negative_rate': [0.9863013698630136, 1.0],
                    'false_negative_rate': [0.014164305949008499, 0.0]},
                    {'sensitive': np.array(['M', 'NAP']),
                    'true_negative_rate': [0.9880952380952381, 1.0],
                    'false_negative_rate': [0.015151515151515152, 0.0]},
                    {'sensitive': np.array(['M', 'ATA']),
                    'true_negative_rate': [0.989247311827957, 1.0],
                    'false_negative_rate': [0.05, 0.0]},
                    {'sensitive': np.array(['F', 'ASY']),
                    'true_negative_rate': [0.9032258064516129, 1.0],
                    'false_negative_rate': [0.02564102564102564, 0.0]},
                    {'sensitive': np.array(['F', 'ATA']),
                    'true_negative_rate': [0.9821428571428571, 1.0],
                    'false_negative_rate': [0.0, 0.0]},
                    {'sensitive': np.array(['F', 'NAP']),
                    'true_negative_rate': [1.0, 1.0],
                    'false_negative_rate': [0.3333333333333333, 0.0]},
                    {'sensitive': np.array(['M', 'TA']),
                    'true_negative_rate': [0.8235294117647058, 1.0],
                    'false_negative_rate': [0.15789473684210525, 0.0]},
                    {'sensitive': np.array(['F', 'TA']),
                    'true_negative_rate': [1.0, 1.0],
                    'false_negative_rate': [0.0, 'ZERO']}]

@pytest.mark.parametrize(
    "data, metrics, zero_division, expected_result",
    [
        (dataset1, None, None, pytest.raises(ValueError)),
        (dataset2, None, None, expected_result_2),
        (dataset3, {'fpr': false_positive_rate}, None, expected_result_3),
        (dataset6, [false_negative_rate, true_negative_rate], "ZERO", expected_result_6),
    ],
)
def test_confusionmatrix(data: Dataset, metrics: Collection | None, zero_division: float | str | None, expected_result: Sequence[float] | RaisesContext):
    if isinstance(expected_result, Sequence):
        cf = ConfusionMatrix(data, metrics, zero_division)
        result = cf()
        assert result[0] == data.real_target
        for i, res in enumerate(result[1]):
            for key, expected_key in zip(res.keys(), expected_result[i].keys()):
                assert key == expected_key
            for val, expected_val in zip(res.values(), expected_result[i].values()):
                if isinstance(val[0], object):
                    try:
                        assert (val == expected_val).all()
                    except AttributeError:
                        assert val == expected_val
                else:
                    assert (np.isclose(val, expected_val, atol=0.0000002)).all()        
    else:
        with expected_result:
            cf = ConfusionMatrix(data, metrics, zero_division)
            cf()
