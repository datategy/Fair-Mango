from fair_mango.dataset.dataset import check_column_in_df, check_real_and_predicted_target_match
from fair_mango.dataset.dataset import Dataset
import pytest
from typing import Sequence
import pandas as pd
from _pytest.python_api import RaisesContext

df = pd.read_csv('tests/data/heart_data.csv')

df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

dataset1 = Dataset(df, ['Sex'], ['HeartDisease'])

dataset2 = Dataset(df, ['Sex'], ['HeartDisease'], ['HeartDiseasePred'])

dataset3 = Dataset(df, ['Sex', 'ChestPainType'], ['HeartDisease'], ['HeartDiseasePred'])

dataset4 = Dataset(df, ['Sex'], ['HeartDisease', 'ExerciseAngina'], None, [1, 'Y'])

dataset5 = Dataset(df, ['Sex'], ['HeartDisease', 'ExerciseAngina'], ['HeartDiseasePred', 'ExerciseAngina'], [1, 'Y'])


@pytest.mark.parametrize(
    "df, columns, expected_result",
    [
        (df2, ["A"], None),
        (df2, ["A", "B"], None),
        (df2, ["D"], pytest.raises(KeyError)),
    ],
)
def test_check_column_in_df(
    df: pd.DataFrame, 
    columns: Sequence, 
    expected_result: None | RaisesContext,
    ):
    if expected_result is None:
        assert check_column_in_df(df, columns) is expected_result
    else:
        with expected_result:
            check_column_in_df(df, columns)


@pytest.mark.parametrize(
    "real_target, predicted_target, expected_result",
    [
        (["A"], ["a"], None),
        (['a', 'b'], ["A", "B"], None),
        ([], [], None),
        (["D"], [], pytest.raises(ValueError)),
        ([], ["D"], pytest.raises(ValueError)),
        (['C', 'c'], ["D"], pytest.raises(ValueError)),
    ],
)
def test_check_real_and_predicted_target_match(
    real_target: Sequence[str], 
    predicted_target: Sequence[str], 
    expected_result: None | RaisesContext,
    ):
    if expected_result is None:
        assert check_real_and_predicted_target_match(real_target, predicted_target) is expected_result
    else:
        with expected_result:
            check_real_and_predicted_target_match(real_target, predicted_target)


@pytest.mark.parametrize(
    "df, sensitive, real_target, predicted_target, positive_target, group_count, expected_result",
    [
        (df, ['Sex'], ['HeartDisease'], None, None, (725, 193),None),
        (df, ['Sex'], ['HeartDisease'], ['HeartDiseasePred'], None, (725, 193), None),
        (df, ['Sex', 'ChestPainType'], ['HeartDisease'], ['HeartDiseasePred'], None, [426, 150, 113,  70,  60,  53,  36,  10],None),
        (df, 'Sex', 'HeartDisease', 'HeartDiseasePred', None, (725, 193), None),
        (df, ['Sex'], ['ExerciseAngina'], None, None, (725, 193), None),
        (df, ['Sex'], ['ExerciseAngina'], None, ['Y'], (725, 193), None),
        (df, ['Sex'], ['HeartDisease','ExerciseAngina'], ['HeartDiseasePred','ExerciseAngina'], [1, 'Y'], (725, 193), None),
        (df, ['NewColumn'], ['HeartDisease'], None, None, (725, 193), pytest.raises(KeyError)),
        (df, ['Sex'], ['NewColumn'], None, None, (725, 193), pytest.raises(KeyError)),
        (df, ['Sex'], ['HeartDisease'], ['NewColumn'], None, (725, 193), pytest.raises(KeyError)),
        (df, ['Sex'], ['HeartDisease'], [], None, (725, 193), pytest.raises(ValueError)),
        (df, ['Sex'], ['HeartDisease'], ['HeartDisease', 'HeartDiseasePred'], None, (725, 193), pytest.raises(ValueError)),
    ],
)
def test_dataset_init(df: pd.DataFrame,
                      sensitive: Sequence[str],
                      real_target: Sequence[str],
                      predicted_target: Sequence[str] | None,
                      positive_target: Sequence[int | float | str | bool] | None,
                      group_count: Sequence[int],
                      expected_result: None | RaisesContext,
                      ):
    if expected_result is None:
        dataset = Dataset(df, sensitive, real_target, predicted_target, positive_target)
        assert type(dataset) is Dataset
        for val in dataset.groups['Sex'].unique():
            assert val in ['M', 'F']
        if 'ChestPainType' in dataset.groups:
            for val in dataset.groups['ChestPainType'].unique():
                assert val in ['ASY', 'NAP', 'ATA', 'TA']
        assert (dataset.groups['Count'].values == group_count).all()
    else:
        with expected_result:
            Dataset(df, sensitive, real_target, predicted_target, positive_target)


@pytest.mark.parametrize(
    "dataset, sensitive_groups, data_shape",
    [
        (dataset2, [['M'], ['F']], [(725, 13), (193, 13)]),
        (dataset3, [['M', 'ASY'],['M', 'NAP'],['M', 'ATA'],['F', 'ASY'],
                    ['F', 'ATA'],['F', 'NAP'],['M', 'TA'],['F', 'TA']],
                   [(426, 13),(150, 13),(113, 13),(70, 13),
                    (60, 13),(53, 13),(36, 13),(10, 13)]),
    ],
)
def test_get_data_for_all_groups(dataset: Dataset,
                                 sensitive_groups: Sequence,
                                 data_shape: Sequence):
    results = dataset.get_data_for_all_groups()
    assert isinstance(results, list)
    for i, result in enumerate(results):
        assert isinstance(result, dict)
        for group, expected_group in zip(result['sensitive'], sensitive_groups[i]):
            assert group == expected_group
        assert isinstance(result['data'], pd.DataFrame)
        for shape, expected_shape in zip(result['data'].shape, data_shape[i]):
            assert shape == expected_shape


@pytest.mark.parametrize(
    "dataset, group, expected_data_shape",
    [
        (dataset1, ['M'], (725, 13)),
        (dataset3, ['M', 'ATA'], (113, 13)),
        (dataset3, ['F', 'TA'], (10, 13)),
        (dataset4, ['F'], (193, 13)),
    ],
)
def test_get_data_for_one_group(dataset: Dataset,
                                group: Sequence,
                                expected_data_shape: Sequence,
                                ):
    result = dataset.get_data_for_one_group(group)
    assert isinstance(result, pd.DataFrame)
    assert (result[dataset.sensitive].nunique().values == 1).all()
    assert result.shape == expected_data_shape


@pytest.mark.parametrize(
    "dataset, sensitive_groups, expected_data_type, expected_data_shape",
    [
        (dataset1, [['M'], ['F']], pd.Series, [(725,), (193,)]),
        (dataset2, [['M'], ['F']], pd.Series, [(725,), (193,)]),
        (dataset3, [['M', 'ASY'],['M', 'NAP'],['M', 'ATA'],['F', 'ASY'],
                    ['F', 'ATA'],['F', 'NAP'],['M', 'TA'],['F', 'TA']], pd.Series, [(426,),(150,),(113,),(70,),(60,),(53,),(36,),(10,)]),
        (dataset4, [['M'], ['F']], pd.DataFrame, [(725, 2), (193, 2)]),
        (dataset5, [['M'], ['F']], pd.DataFrame, [(725, 2), (193, 2)]),
    ],
)
def test_get_real_target_for_all_groups(dataset: Dataset,
                                        sensitive_groups: Sequence,
                                        expected_data_type: pd.Series | pd.DataFrame,
                                        expected_data_shape: Sequence,
                                        ):
    results = dataset.get_real_target_for_all_groups()
    assert isinstance(results, list)
    for i, result in enumerate(results):
        assert isinstance(result, dict)
        for group, expected_group in zip(result['sensitive'], sensitive_groups[i]):
            assert group == expected_group
        assert isinstance(result['data'], expected_data_type)
        assert result['data'].shape in expected_data_shape


@pytest.mark.parametrize(
    "dataset, sensitive_groups, expected_data_type, expected_data_shape, exception",
    [
        (dataset1, [['M'], ['F']], pd.Series, [(725,), (193,)], pytest.raises(ValueError)),
        (dataset2, [['M'], ['F']], pd.Series, [(725,), (193,)], None),
        (dataset3, [['M', 'ASY'],['M', 'NAP'],['M', 'ATA'],['F', 'ASY'],
                    ['F', 'ATA'],['F', 'NAP'],['M', 'TA'],['F', 'TA']], pd.Series, [(426,),(150,),(113,),(70,),(60,),(53,),(36,),(10,)], None),
        (dataset4, [['M'], ['F']], pd.DataFrame, [(725, 2), (193, 2)], pytest.raises(ValueError)),
        (dataset5, [['M'], ['F']], pd.DataFrame, [(725, 2), (193, 2)], None),
    ],
)
def test_get_predicted_target_for_all_groups(dataset: Dataset,
                                             sensitive_groups: Sequence,
                                             expected_data_type: pd.Series | pd.DataFrame,
                                             expected_data_shape: Sequence,
                                             exception: RaisesContext | None,
                                             ):
    if exception is None:
        results = dataset.get_predicted_target_for_all_groups()
        assert isinstance(results, list)
        for i, result in enumerate(results):
            assert isinstance(result, dict)
            for group, expected_group in zip(result['sensitive'], sensitive_groups[i]):
                assert group == expected_group
            assert isinstance(result['data'], expected_data_type)
            assert result['data'].shape in expected_data_shape
    else:
        with exception:
            dataset.get_predicted_target_for_all_groups()


@pytest.mark.parametrize(
    "dataset, sensitive_groups, expected_data_type, expected_data_shape",
    [
        (dataset1, ['M'], pd.Series, (725,)),
        (dataset4, 'F', pd.DataFrame, (193,2)),
        (dataset3, ['M', 'ASY'], pd.Series, (426,)),
        (dataset3, ['F', 'ASY'], pd.Series,(70,)),
    ],
)
def test_get_real_target_for_one_group(dataset: Dataset,
                                       sensitive_groups: Sequence,
                                       expected_data_type: pd.Series | pd.DataFrame,
                                       expected_data_shape: Sequence,
                                       ):
    result = dataset.get_real_target_for_one_group(sensitive_groups)
    assert isinstance(result, expected_data_type)
    assert result.shape == expected_data_shape


@pytest.mark.parametrize(
    "dataset, sensitive_groups, expected_data_type, expected_data_shape, exception",
    [
        (dataset1, ['M'], pd.Series, (725,), pytest.raises(ValueError)),
        (dataset4, 'F', pd.DataFrame, (193,2), pytest.raises(ValueError)),
        (dataset3, ['M', 'ASY'], pd.Series, (426,), None),
        (dataset3, ['F', 'ASY'], pd.Series,(70,), None),
        (dataset5, 'F', pd.DataFrame, (193,2), None),
    ],
)
def test_get_predicted_target_for_one_group(dataset: Dataset,
                                       sensitive_groups: Sequence,
                                       expected_data_type: pd.Series | pd.DataFrame,
                                       expected_data_shape: Sequence,
                                       exception: RaisesContext | None
                                       ):
    if exception is None:
        result = dataset.get_predicted_target_for_one_group(sensitive_groups)
        assert isinstance(result, expected_data_type)
        assert result.shape == expected_data_shape
    else:
        with exception:
            dataset.get_predicted_target_for_one_group(sensitive_groups)
