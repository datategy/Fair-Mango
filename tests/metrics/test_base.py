from contextlib import AbstractContextManager

import pandas as pd
import pytest

from fair_mango.dataset.dataset import Dataset
from fair_mango.metrics.base import encode_target, is_binary

df = pd.read_csv("tests/data/heart_data.csv")

dataset1 = Dataset(df, ["Sex"], "HeartDisease")

dataset2 = Dataset(df, ["Sex"], "HeartDisease", "HeartDiseasePred")

dataset3 = Dataset(df, ["Sex", "ChestPainType"], "HeartDisease", "HeartDiseasePred")

dataset4 = Dataset(df, ["Sex"], "HeartDisease", None, 1)

dataset5 = Dataset(
    df,
    ["Sex"],
    "ExerciseAngina",
    "ExerciseAnginaPred",
    "Y",
)

dataset6 = Dataset(
    df,
    ["Sex", "ChestPainType"],
    "HeartDisease",
    "HeartDiseasePred",
    1,
)


@pytest.mark.parametrize(
    "y, expected_result",
    [
        (df["Sex"], True),
        (df["ExerciseAngina"], True),
        (df["ChestPainType"], False),
    ],
)
def test_is_binary(y: pd.Series, expected_result: bool):
    assert is_binary(y) == expected_result


@pytest.mark.parametrize(
    "data, col, expected_result",
    [
        (dataset4, "HeartDisease", None),
        (dataset5, "ExerciseAngina", None),
        (dataset5, "HeartDiseasePred", pytest.raises(KeyError)),
        (dataset3, "ExerciseAngina", pytest.raises(ValueError)),
    ],
)
def test_encode_target(
    data: Dataset, col: str, expected_result: None | AbstractContextManager
):
    if expected_result is None:
        encode_target(data, col)
        assert sorted(data.df[col].unique()) == [0, 1]
    else:
        with expected_result:
            encode_target(data, col)