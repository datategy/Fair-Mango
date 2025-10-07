from contextlib import AbstractContextManager

import pandas as pd
import pytest

from fair_mango.dataset.dataset import Dataset
from fair_mango.metrics.base import calculate_disparity, encode_target, is_binary
from fair_mango.typing import BaseMetricResult, DisparityResultDict

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


@pytest.mark.parametrize(
    "data, method, expected_result",
    [
        (
            [
                BaseMetricResult(["A"], 0),
                BaseMetricResult(["B"], 0),
                BaseMetricResult(["C"], 0),
            ],
            "difference",
            [
                {"group_1": ["A"], "group_2": ["B"], "disparity": 0},
                {"group_1": ["A"], "group_2": ["C"], "disparity": 0},
                {"group_1": ["B"], "group_2": ["C"], "disparity": 0},
            ],
        ),
        (
            [
                BaseMetricResult(["A"], 0),
                BaseMetricResult(["B"], 0),
                BaseMetricResult(["C"], 0),
            ],
            "ratio",
            [
                {"group_1": ["A"], "group_2": ["B"], "disparity": 1},
                {"group_1": ["A"], "group_2": ["C"], "disparity": 1},
                {"group_1": ["B"], "group_2": ["C"], "disparity": 1},
            ],
        ),
        (
            [
                BaseMetricResult(["A"], 2),
                BaseMetricResult(["B"], 1),
                BaseMetricResult(["C"], 0),
            ],
            "difference",
            [
                {"group_1": ["A"], "group_2": ["B"], "disparity": 1},
                {"group_1": ["A"], "group_2": ["C"], "disparity": 2},
                {"group_1": ["B"], "group_2": ["C"], "disparity": 1},
            ],
        ),
        # check that changing the order will not change the results
        (
            [
                BaseMetricResult(["C"], 0),
                BaseMetricResult(["B"], 1),
                BaseMetricResult(["A"], 2),
            ],
            "difference",
            [
                {"group_1": ["B"], "group_2": ["C"], "disparity": 1},
                {"group_1": ["A"], "group_2": ["C"], "disparity": 2},
                {"group_1": ["A"], "group_2": ["B"], "disparity": 1},
            ],
        ),
        # check that ratio disparity is always < 1
        (
            [
                BaseMetricResult(["C"], 0),
                BaseMetricResult(["B"], 1),
                BaseMetricResult(["A"], 2),
            ],
            "ratio",
            [
                {"group_1": ["C"], "group_2": ["B"], "disparity": 0},
                {"group_1": ["C"], "group_2": ["A"], "disparity": 0},
                {"group_1": ["B"], "group_2": ["A"], "disparity": 0.5},
            ],
        ),
        # check that changing the order results in the same results
        (
            [
                BaseMetricResult(["A"], 2),
                BaseMetricResult(["B"], 1),
                BaseMetricResult(["C"], 0.1),
            ],
            "ratio",
            [
                {"group_1": ["B"], "group_2": ["A"], "disparity": 0.5},
                {"group_1": ["C"], "group_2": ["A"], "disparity": 0.05},
                {"group_1": ["C"], "group_2": ["B"], "disparity": 0.1},
            ],
        ),
    ],
)
def test_calculate_disparity(data, method, expected_result: DisparityResultDict):
    assert calculate_disparity(data, method) == expected_result
