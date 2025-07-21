from collections.abc import Sequence
from contextlib import AbstractContextManager

import numpy as np
import pandas as pd
import pytest

from fair_mango.dataset.dataset import (
    Dataset,
    check_column_existence_in_df,
    validate_columns,
)

df = pd.read_csv("tests/data/heart_data.csv")

df2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

dataset1 = Dataset(df, "Sex", "HeartDisease")

dataset2 = Dataset(df, "Sex", "HeartDisease", "HeartDiseasePred")

dataset3 = Dataset(df, ["Sex", "ChestPainType"], "HeartDisease", "HeartDiseasePred")

dataset4 = Dataset(df, "Sex", "HeartDisease", None, 1)

dataset5 = Dataset(
    df,
    "Sex",
    "ExerciseAngina",
    "ExerciseAnginaPred",
    "Y",
)


@pytest.mark.parametrize(
    "df, columns, expected_result",
    [
        (df2, ["A"], None),
        (df2, ["A", "B"], None),
        (df2, ["D"], pytest.raises(KeyError)),
    ],
)
def test_check_column_existence_in_df(
    df: pd.DataFrame,
    columns: Sequence,
    expected_result: None | AbstractContextManager,
):
    if expected_result is not None:
        with expected_result:
            check_column_existence_in_df(df, columns)
    else:
        check_column_existence_in_df(df, columns)


@pytest.mark.parametrize(
    "df, sensitive, real_target, predicted_target, positive_target, group_count, expected_result",
    [
        (df, "Sex", "HeartDisease", None, None, (725, 193), None),
        (df, "Sex", "HeartDisease", "HeartDiseasePred", None, (725, 193), None),
        (
            df,
            ["Sex", "ChestPainType"],
            "HeartDisease",
            "HeartDiseasePred",
            None,
            [426, 150, 113, 70, 60, 53, 36, 10],
            None,
        ),
        (df, "Sex", "HeartDisease", "HeartDiseasePred", None, (725, 193), None),
        (df, "Sex", "ExerciseAngina", None, None, (725, 193), None),
        (df, "Sex", "ExerciseAngina", None, "Y", (725, 193), None),
        (
            df,
            "Sex",
            "HeartDisease",
            ["HeartDiseasePred", "ExerciseAnginaPred"],
            [1, "Y"],
            (725, 193),
            pytest.raises(TypeError),
        ),
        (
            df,
            ["NewColumn"],
            "HeartDisease",
            None,
            None,
            (725, 193),
            pytest.raises(KeyError),
        ),
        (df, "Sex", "NewColumn", None, None, (725, 193), pytest.raises(KeyError)),
        (
            df,
            "Sex",
            "HeartDisease",
            "NewColumn",
            None,
            (725, 193),
            pytest.raises(KeyError),
        ),
        (
            df,
            "Sex",
            "HeartDisease",
            ["HeartDisease", "HeartDiseasePred"],
            None,
            (725, 193),
            pytest.raises(TypeError),
        ),
    ],
)
def test_dataset_init(
    df: pd.DataFrame,
    sensitive: Sequence[str],
    real_target: str,
    predicted_target: str,
    positive_target: int | float | str | bool | None,
    group_count: Sequence[int],
    expected_result: None | AbstractContextManager,
):
    if expected_result is None:
        dataset = Dataset(df, sensitive, real_target, predicted_target, positive_target)
        assert type(dataset) is Dataset
        for val in dataset.groups["Sex"].unique():
            assert val in ["M", "F"]
        if "ChestPainType" in dataset.groups:
            for val in dataset.groups["ChestPainType"].unique():
                assert val in ["ASY", "NAP", "ATA", "TA"]
        assert np.all(dataset.groups["Count"].values == group_count)
    else:
        with expected_result:
            Dataset(df, sensitive, real_target, predicted_target, positive_target)


@pytest.mark.parametrize(
    "dataset, sensitive_groups, data_shape",
    [
        (dataset2, [["M"], ["F"]], [(725, 14), (193, 14)]),
        (
            dataset3,
            [
                ["M", "ASY"],
                ["M", "NAP"],
                ["M", "ATA"],
                ["F", "ASY"],
                ["F", "ATA"],
                ["F", "NAP"],
                ["M", "TA"],
                ["F", "TA"],
            ],
            [
                (426, 14),
                (150, 14),
                (113, 14),
                (70, 14),
                (60, 14),
                (53, 14),
                (36, 14),
                (10, 14),
            ],
        ),
    ],
)
def test_get_data_for_all_groups(
    dataset: Dataset, sensitive_groups: Sequence, data_shape: Sequence
):
    results = dataset.get_data_for_all_groups()
    assert isinstance(results, list)
    for i, result in enumerate(results):
        assert isinstance(result, dict)
        for group, expected_group in zip(result["sensitive_group"], sensitive_groups[i]):
            assert group == expected_group
        assert isinstance(result["data"], pd.DataFrame)


@pytest.mark.parametrize(
    "dataset, group, expected_data_shape",
    [
        (dataset1, ["M"], (725, 14)),
        (dataset3, ["M", "ATA"], (113, 14)),
        (dataset3, ["F", "TA"], (10, 14)),
        (dataset4, ["F"], (193, 14)),
    ],
)
def test_get_data_for_one_group(
    dataset: Dataset,
    group: Sequence,
    expected_data_shape: Sequence,
):
    result = dataset.get_data_for_one_group(group)
    assert isinstance(result, pd.DataFrame)
    assert np.all(result[dataset.sensitive].nunique().values == 1)
    assert result.shape == expected_data_shape


@pytest.mark.parametrize(
    "dataset, sensitive_groups, expected_data_type, expected_data_shape",
    [
        (dataset1, [["M"], ["F"]], pd.Series, [(725,), (193,)]),
        (dataset2, [["M"], ["F"]], pd.Series, [(725,), (193,)]),
        (
            dataset3,
            [
                ["M", "ASY"],
                ["M", "NAP"],
                ["M", "ATA"],
                ["F", "ASY"],
                ["F", "ATA"],
                ["F", "NAP"],
                ["M", "TA"],
                ["F", "TA"],
            ],
            pd.Series,
            [(426,), (150,), (113,), (70,), (60,), (53,), (36,), (10,)],
        ),
        (dataset4, [["M"], ["F"]], pd.Series, [(725,), (193,)]),
        (dataset5, [["M"], ["F"]], pd.Series, [(725,), (193,)]),
    ],
)
def test_get_real_target_for_all_groups(
    dataset: Dataset,
    sensitive_groups: Sequence,
    expected_data_type: type[pd.DataFrame],
    expected_data_shape: Sequence,
):
    results = dataset.get_real_target_for_all_groups()
    assert isinstance(results, list)
    for i, result in enumerate(results):
        assert isinstance(result, dict)
        for group, expected_group in zip(result["sensitive_group"], sensitive_groups[i]):
            assert group == expected_group
        assert isinstance(result["data"], expected_data_type)
        assert result["data"].shape in expected_data_shape


@pytest.mark.parametrize(
    "dataset, sensitive_groups, expected_data_type, expected_data_shape, exception",
    [
        (
            dataset1,
            [["M"], ["F"]],
            pd.Series,
            [(725,), (193,)],
            pytest.raises(ValueError),
        ),
        (dataset2, [["M"], ["F"]], pd.Series, [(725,), (193,)], None),
        (
            dataset3,
            [
                ["M", "ASY"],
                ["M", "NAP"],
                ["M", "ATA"],
                ["F", "ASY"],
                ["F", "ATA"],
                ["F", "NAP"],
                ["M", "TA"],
                ["F", "TA"],
            ],
            pd.Series,
            [(426,), (150,), (113,), (70,), (60,), (53,), (36,), (10,)],
            None,
        ),
        (
            dataset4,
            [["M"], ["F"]],
            pd.Series,
            [(725, 2), (193, 2)],
            pytest.raises(ValueError),
        ),
        (dataset5, [["M"], ["F"]], pd.Series, [(725,), (193,)], None),
    ],
)
def test_get_predicted_target_for_all_groups(
    dataset: Dataset,
    sensitive_groups: Sequence,
    expected_data_type: type[pd.DataFrame],
    expected_data_shape: Sequence,
    exception: None | AbstractContextManager,
):
    if exception is None:
        results = dataset.get_predicted_target_for_all_groups()
        assert isinstance(results, list)
        for i, result in enumerate(results):
            assert isinstance(result, dict)
            for group, expected_group in zip(result["sensitive_group"], sensitive_groups[i]):
                assert group == expected_group
            assert isinstance(result["data"], expected_data_type)
            assert result["data"].shape in expected_data_shape
    else:
        with exception:
            dataset.get_predicted_target_for_all_groups()


@pytest.mark.parametrize(
    "dataset, sensitive_groups, expected_data_type, expected_data_shape",
    [
        (dataset1, ["M"], pd.Series, (725,)),
        (dataset4, "F", pd.Series, (193,)),
        (dataset3, ["M", "ASY"], pd.Series, (426,)),
        (dataset3, ["F", "ASY"], pd.Series, (70,)),
    ],
)
def test_get_real_target_for_one_group(
    dataset: Dataset,
    sensitive_groups: Sequence,
    expected_data_type: type[pd.Series],
    expected_data_shape: Sequence,
):
    result = dataset.get_real_target_for_one_group(sensitive_groups)
    assert isinstance(result, expected_data_type)
    assert result.shape == expected_data_shape


@pytest.mark.parametrize(
    "dataset, sensitive_groups, expected_data_type, expected_data_shape, exception",
    [
        (dataset1, ["M"], pd.Series, (725,), pytest.raises(ValueError)),
        (dataset4, "F", pd.Series, (193,), pytest.raises(ValueError)),
        (dataset3, ["M", "ASY"], pd.Series, (426,), None),
        (dataset3, ["F", "ASY"], pd.Series, (70,), None),
        (dataset5, "F", pd.Series, (193,), None),
    ],
)
def test_get_predicted_target_for_one_group(
    dataset: Dataset,
    sensitive_groups: Sequence,
    expected_data_type: type[pd.Series],
    expected_data_shape: Sequence,
    exception: None | AbstractContextManager,
):
    if exception is None:
        result = dataset.get_predicted_target_for_one_group(sensitive_groups)
        assert isinstance(result, expected_data_type)
        assert result.shape == expected_data_shape
    else:
        with exception:
            dataset.get_predicted_target_for_one_group(sensitive_groups)


@pytest.mark.parametrize(
    "sensitive, real_target, predicted_target, expected_result",
    [
        (["gender"], "r1", None, None),
        (["gender", "race"], "r1", "p1", None),
        (["race"], "race", None, pytest.raises(AttributeError)),
        (["gender", "race"], "r1", "gender", pytest.raises(AttributeError)),
    ],
)
def test_validate_columns(
    sensitive: Sequence[str],
    real_target: str,
    predicted_target: str | None,
    expected_result: None | AbstractContextManager,
):
    if expected_result is not None:
        with expected_result:
            validate_columns(sensitive, real_target, predicted_target)
    else:
        validate_columns(sensitive, real_target, predicted_target)