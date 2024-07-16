import pandas as pd
import pytest
from _pytest.python_api import RaisesContext

from fair_mango.dataset.dataset import Dataset
from fair_mango.metrics.base import encode_target, is_binary

df = pd.read_csv("tests/data/heart_data.csv")

dataset1 = Dataset(df, ["Sex"], ["HeartDisease"])

dataset2 = Dataset(df, ["Sex"], ["HeartDisease"], ["HeartDiseasePred"])

dataset3 = Dataset(df, ["Sex", "ChestPainType"], ["HeartDisease"], ["HeartDiseasePred"])

dataset4 = Dataset(df, ["Sex"], ["HeartDisease", "ExerciseAngina"], None, [1, "Y"])

dataset5 = Dataset(
    df,
    ["Sex"],
    ["HeartDisease", "ExerciseAngina"],
    ["HeartDiseasePred", "ExerciseAngina"],
    [1, "Y"],
)

dataset6 = Dataset(
    df,
    ["Sex", "ChestPainType"],
    ["HeartDisease", "ExerciseAngina"],
    ["HeartDiseasePred", "ExerciseAngina"],
    [1, "Y"],
)


@pytest.mark.parametrize(
    "y, expected_result",
    [
        (df["Sex"], True),
        (df["ExerciseAngina"], True),
        (df["ChestPainType"], False),
        (df[["ExerciseAngina", "Sex"]], True),
    ],
)
def test_is_binary(y: pd.Series | pd.DataFrame, expected_result: bool):
    assert is_binary(y) == expected_result


@pytest.mark.parametrize(
    "data, ind, col, expected_result",
    [
        (dataset4, 0, "HeartDisease", None),
        (dataset5, 1, "ExerciseAngina", None),
        (dataset5, 1, "HeartDiseasePred", pytest.raises(KeyError)),
        (dataset3, 0, "ExerciseAngina", pytest.raises(ValueError)),
    ],
)
def test_encode_target(
    data: Dataset, ind: int, col: str, expected_result: None | RaisesContext
):
    if expected_result is None:
        encode_target(data, ind, col)
        assert sorted(data.df[col].unique()) == [0, 1]
    else:
        with expected_result:
            encode_target(data, ind, col)
