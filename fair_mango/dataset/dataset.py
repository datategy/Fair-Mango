from collections.abc import Sequence

import pandas as pd

from fair_mango.typing import DatasetGroupResult, DatasetTargetResult, SensitiveGroupT


def check_column_existence_in_df(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate the columns existence in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to check.
    columns : Sequence[str]
        Sequence of column names to check if they exist in the dataframe.

    Raises
    ------
    KeyError
        If the one of the columns does not exist in the dataframe.
    """
    for column in columns:
        if column not in df.columns:
            raise (
                KeyError(f"{column} column does not exist in the dataframe provided")
            )


def validate_columns(
    sensitive: Sequence[str],
    real_target: str,
    predicted_target: str | None = None,
) -> None:
    """Make sure that the columns provided as parameters are different.
    A column cannot be a sensitive column and a target at the same time.
    A column cannot be a real target and a predicted target at the same time.

    Parameters
    ----------
    sensitive : Sequence[str]
        Sequence of column names corresponding to sensitive features
        (Ex: gender, race...).
    real_target : Sequence[str]
        Sequence of column names corresponding to the real target
        (true labels). Every target will be processed independently.
    predicted_target : Sequence[str], optional
        Sequence of column names corresponding to the predicted target,
        by default None.

    Raises
    ------
    AttributeError
        If the same column is assigned to different parameters at the same time.
    """
    sensitive_set = set(sensitive)

    overlap = set()
    if real_target in sensitive_set:
        overlap.add(real_target)

    if predicted_target is not None:
        if predicted_target in sensitive_set:
            overlap.add(predicted_target)
        if predicted_target == real_target:
            overlap.add(predicted_target)

    if overlap:
        raise AttributeError(
            f"Columns must be different, You provided the same column in {overlap}"
        )


def convert_to_list(variable: SensitiveGroupT | str) -> list[str]:
    """Convert a variable of type str or SensitiveGroupT to a list of strings.

    Parameters
    ----------
    variable : SensitiveGroupT | str
        Sequence of values or a single string value.

    Returns
    -------
    list[str]
        List of the values converted to strings.
    """
    if isinstance(variable, str):
        return [variable]
    else:
        return [str(item) for item in variable]


def df_filtration(
    df: pd.DataFrame, sensitive_group: Sequence[str], sensitive: Sequence[str]
) -> pd.DataFrame:
    """Filters a DataFrame to only take the sensitive groups

    Parameters
    ----------
    df : pd.DataFrame
        df of the dataset

    sensitive : Sequence[str]
        Sequence of sensitive values must be in the same order as `sensitive`
        attribute, and so `sensitive_group` must be the same length as
        `sensitive`. For instance, if your `sensitive` attributes were
        `["race", "gender"]`, you can pass `sensitive_group=["white", "male"]`.

    Returns
    -------
    pd.Dataframe
    """
    mask = pd.Series(True, index=df.index)
    for column, value in zip(sensitive, sensitive_group, strict=True):
        mask &= df[column] == value
    filtered_df = df[mask]
    return filtered_df


class Dataset:
    """A class for handling datasets with sensitive attributes and target
    variables.

    This class separates a dataframe into different sensitive groups present
    in the dataframe. Any object of this class will serve as a building block
    for evaluating the performance of different sensitive groups and
    calculating the fairness metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sensitive : Sequence[str]
        Sequence of column names corresponding to sensitive features
        (Ex: gender, race...).
    real_target : str
        The column name corresponding to the real target
        (true label).
    predicted_target : str, optional
        The column name corresponding to the predicted target,
        by default None.
    positive_target : int  |  float  |  str  |  bool | None, optional
        The positive label corresponding to the provided target,
        by default None.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sensitive: SensitiveGroupT | str,
        real_target: str,
        predicted_target: str | None = None,
        positive_target: int | float | str | bool | None = None,
    ):
        self.sensitive = convert_to_list(sensitive)
        check_column_existence_in_df(df, self.sensitive)
        self.real_target = real_target
        check_column_existence_in_df(df, [self.real_target])
        self.predicted_target = predicted_target
        if self.predicted_target is not None:
            check_column_existence_in_df(df, [self.predicted_target])
        validate_columns(self.sensitive, self.real_target, self.predicted_target)
        self.df = df.copy()
        self.shape = df.shape
        self.positive_target = positive_target
        if len(self.sensitive) == 1:
            self.groups = (
                df[self.sensitive]
                .groupby(self.sensitive)
                .size()
                .reset_index(name="Count")
                .sort_values("Count", ascending=False)
            )
        else:
            self.groups = (
                df[self.sensitive]
                .groupby(list(self.sensitive))
                .size()
                .reset_index(name="Count")
                .sort_values("Count", ascending=False)
            )
        self.n_groups: int = len(self.groups)
        self.groups_data: list[DatasetGroupResult] = []
        self.groups_real_target: list[DatasetTargetResult] | None = None
        self.groups_predicted_target: list[DatasetTargetResult] | None = None

    def get_data_for_all_groups(self) -> list[DatasetGroupResult]:
        """Retrieve data corresponding to each sensitive group present in
        the sensitive features.

        Tip
        ---
        If you have two sensitive attributes `gender` (male, female) and `race`
        (white, black), this function would return the data for the combination
        of the two sensitive features; Hence, all of the following groups:
        (male, white), (female, white), (female, black), (male, black)

        Returns
        -------
        list[dict[str, np.ndarray | pd.DataFrame]]
            List of dictionaries with the sensitive group as keys and the
            corresponding dataframe as value.

        Examples
        --------
        >>> import pandas as pd
        >>> from fair_mango.dataset.dataset import Dataset
        >>> data = {
        ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male'],
        ...     'sensitive_2': ['white', 'white', 'black', 'black', 'black'],
        ...     'col-a': ['a', 'A', 'a', 'A', 'a'],
        ...     'col-b': ['B', 'B', 'b', 'B', 'b'],
        ...     'real_target_1': [0, 1, 0, 1, 0],
        ...     'real_target_2': ['no', 'yes', 'yes', 'yes', 'no'],
        ...     'predicted_target_1': [0, 1, 1, 0, 0],
        ...     'predicted_target_2': ['no', 'no', 'yes', 'yes', 'yes'],
        ... }
        >>> df = pd.DataFrame(data)
        >>> dataset1 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1'],
        ...     positive_target=[1]
        ... )
        >>> dataset1.get_data_for_all_groups()
        [
            {
                'sensitive_group': array(['male'], dtype=object),
                'data':   sensitive_1 sensitive_2  ... predicted_target_1 predicted_target_2
                    0        male       white      ...         0                 no
                    3        male       black      ...         0                yes
                    4        male       black      ...         0                yes

                [3 rows x 8 columns]
            },
            {
                'sensitive_group': array(['female'], dtype=object),
                'data':   sensitive_1 sensitive_2  ... predicted_target_1 predicted_target_2
                     1      female       white     ...        1                 no
                     2      female       black     ...        1                yes

                [2 rows x 8 columns]
            }
        ]
        >>> dataset2 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1', 'sensitive_2'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1'],
        ...     positive_target=[1]
        ... )
        >>> dataset2.get_data_for_all_groups()
        [
            {
                'sensitive_group': array(['male', 'black'], dtype=object),
                'data':   sensitive_1 sensitive_2  ... predicted_target_1 predicted_target_2
                    3        male       black      ...         0                yes
                    4        male       black      ...         0                yes

                [2 rows x 8 columns]
            },
            {
                'sensitive_group': array(['female', 'black'], dtype=object),
                'data':   sensitive_1 sensitive_2  ... predicted_target_1 predicted_target_2
                    2      female       black      ...         1                yes

                [1 rows x 8 columns]
            },
            {
                'sensitive_group': array(['female', 'white'], dtype=object),
                'data':   sensitive_1 sensitive_2  ... predicted_target_1 predicted_target_2
                    1      female       white      ...         1                 no

                [1 rows x 8 columns]
            },
            {
                'sensitive_group': array(['male', 'white'], dtype=object),
                'data':   sensitive_1 sensitive_2  ... predicted_target_1 predicted_target_2
                    0        male       white      ...         0                 no

                [1 rows x 8 columns]
            }
        ]
        """
        for row in self.groups.values:
            result = self.df
            for i in range(len(self.sensitive)):
                result = result[result[self.sensitive[i]] == row[i]]
            assert isinstance(result, pd.DataFrame)
            self.groups_data.append(
                DatasetGroupResult(
                    sensitive_group=[str(x) for x in row[:-1]], data=result
                )
            )
        return self.groups_data

    def get_data_for_one_group(self, sensitive_group: Sequence[str]) -> pd.DataFrame:
        """Retrieve data corresponding to a specific sensitive group present
        in the sensitive features.

        Tip
        ---
        If you have two sensitive attributes `gender` (male, female) and `race`
        (white, black), this function would return the data for the combination
        of the two sensitive features; Hence, it expects the `sensitive_group`
        parameter to match the `sensitive` parameter when  creating the
        `Dataset`. For example: `sensitive = ['Sex', 'Race']` then
        `sensitive_group = ['male', 'Asian']` (The order of the values matters
        and exchanging the places will not work!)

        Parameters
        ----------
        sensitive_group : Sequence[str]
            Sequence of sensitive values must be in the same order as `sensitive`
            attribute, and so `sensitive_group` must be the same length as
            `sensitive`. For instance, if your `sensitive` attributes were
            `["race", "gender"]`, you can pass `sensitive_group=["white", "male"]`.

        Returns
        -------
        pd.DataFrame
            The dataframe corresponding to the sensitive group specified.

        Examples
        --------
        >>> import pandas as pd
        >>> from fair_mango.dataset.dataset import Dataset
        >>> data = {
        ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male'],
        ...     'sensitive_2': ['white', 'white', 'black', 'black', 'black'],
        ...     'col-a': ['a', 'A', 'a', 'A', 'a'],
        ...     'col-b': ['B', 'B', 'b', 'B', 'b'],
        ...     'real_target_1': [0, 1, 0, 1, 0],
        ...     'real_target_2': ['no', 'yes', 'yes', 'yes', 'no'],
        ...     'predicted_target_1': [0, 1, 1, 0, 0],
        ...     'predicted_target_2': ['no', 'no', 'yes', 'yes', 'yes'],
        ... }
        >>> df = pd.DataFrame(data)
        >>> dataset1 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1'],
        ...     positive_target=[1]
        ... )
        >>> dataset1.get_data_for_one_group(['female'])
            sensitive_1 sensitive_2  ... predicted_target_1 predicted_target_2
        1      female       white    ...         1                 no
        2      female       black    ...         1                yes

        [2 rows x 8 columns]
        >>> dataset2 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1', 'sensitive_2'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1'],
        ...     positive_target=[1]
        ... )
        >>> dataset2.get_data_for_one_group(['male', 'black'])
            sensitive_1 sensitive_2  ... predicted_target_1 predicted_target_2
        3        male       black    ...         0                yes
        4        male       black    ...         0                yes

        [2 rows x 8 columns]
        """
        result = None
        if self.groups_data == []:
            result = df_filtration(self.df, sensitive_group, self.sensitive)
        else:
            for item in self.groups_data:
                if set(item.sensitive_group) == set(sensitive_group):
                    result = item.data
        if result is None:
            raise (
                ValueError(f"{sensitive_group} group does not exist in the dataframe")
            )
        return result

    def get_real_target_for_all_groups(
        self,
    ) -> list[DatasetTargetResult]:
        """Retrieve the real target corresponding to each sensitive group
        present in the sensitive features.

        Tip
        ---
        If you have two sensitive attributes `gender` (male, female) and `race`
        (white, black), this function would return the real target for the
        combination of the two sensitive features; Hence, all of the following
        groups: (male, white), (female, white), (female, black), (male, black)

        Returns
        -------
        list[DatasetTargetResult]
            List of DatasetTargetResult dictionaries with standardized structure.

        Examples
        --------
        >>> import pandas as pd
        >>> from fair_mango.dataset.dataset import Dataset
        >>> data = {
        ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male'],
        ...     'sensitive_2': ['white', 'white', 'black', 'black', 'black'],
        ...     'col-a': ['a', 'A', 'a', 'A', 'a'],
        ...     'col-b': ['B', 'B', 'b', 'B', 'b'],
        ...     'real_target_1': [0, 1, 0, 1, 0],
        ...     'real_target_2': ['no', 'yes', 'yes', 'yes', 'no'],
        ...     'predicted_target_1': [0, 1, 1, 0, 0],
        ...     'predicted_target_2': ['no', 'no', 'yes', 'yes', 'yes'],
        ... }
        >>> df = pd.DataFrame(data)
        >>> dataset1 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1'],
        ...     positive_target=[1]
        ... )
        >>> dataset1.get_real_target_for_all_groups()
        [
            {
                'sensitive_group': array(['male'], dtype=object),
                'data': 0    0
                        3    1
                        4    0
                        Name: real_target_1, dtype: int64
            },
            {
                'sensitive_group': array(['female'], dtype=object),
                'data': 1    1
                        2    0
                        Name: real_target_1, dtype: int64
            }
        ]
        >>> dataset2 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1', 'sensitive_2'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1'],
        ...     positive_target=[1]
        ... )
        >>> dataset2.get_real_target_for_all_groups()
        [
            {
                'sensitive_group': array(['male', 'black'], dtype=object),
                'data': 3    1
                        4    0
                        Name: real_target_1, dtype: int64
            },
            {
                'sensitive_group': array(['female', 'black'], dtype=object),
                'data': 0
            },
            {
                'sensitive_group': array(['female', 'white'], dtype=object),
                'data': 1
            },
            {
                'sensitive_group': array(['male', 'white'], dtype=object),
                'data': 0
            }
        ]
        """
        self.groups_real_target = []
        for row in self.groups.values:
            result = self.df
            for i in range(len(self.sensitive)):
                result = result[result[self.sensitive[i]] == row[i]]

            sensitive_group = [str(x) for x in row[:-1]]
            target_data = result[self.real_target]

            self.groups_real_target.append(
                DatasetTargetResult(sensitive_group=sensitive_group, data=target_data)
            )
        return self.groups_real_target

    def get_real_target_for_one_group(
        self, sensitive_group: SensitiveGroupT | str
    ) -> pd.Series:
        """Retrieve the real target corresponding to a specific sensitive
        group present in the sensitive features.

        Tip
        ---
        If you have two sensitive attributes `gender` (male, female) and `race`
        (white, black), this function would return the real target for the
        combination of the two sensitive features; Hence, it expects the
        `sensitive_group` parameter to match the `sensitive` parameter when
        creating the `Dataset`. For example: `sensitive = ['Sex', 'Race']` then
        `sensitive_group = ['male', 'Asian']` (The order of the values matters
        and exchanging the places will not work!)

        Parameters
        ----------
        sensitive : Sequence[str]
            Sequence of sensitive values must be in the same order as `sensitive`
            attribute, and so `sensitive_group` must be the same length as
            `sensitive`. For instance, if your `sensitive` attributes were
            `["race", "gender"]`, you can pass `sensitive_group=["white", "male"]`.

        Returns
        -------
        pd.DataFrame
            The pandas dataframe corresponding to the sensitive group.

        Examples
        --------
        >>> import pandas as pd
        >>> from fair_mango.dataset.dataset import Dataset
        >>> data = {
        ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male'],
        ...     'sensitive_2': ['white', 'white', 'black', 'black', 'black'],
        ...     'col-a': ['a', 'A', 'a', 'A', 'a'],
        ...     'col-b': ['B', 'B', 'b', 'B', 'b'],
        ...     'real_target_1': [0, 1, 0, 1, 0],
        ...     'real_target_2': ['no', 'yes', 'yes', 'yes', 'no'],
        ...     'predicted_target_1': [0, 1, 1, 0, 0],
        ...     'predicted_target_2': ['no', 'no', 'yes', 'yes', 'yes'],
        ... }
        >>> df = pd.DataFrame(data)
        >>> dataset1 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1'],
        ...     positive_target=[1]
        ... )
        >>> dataset1.get_real_target_for_one_group(['female'])
        1    1
        2    0
        Name: real_target_1, dtype: int64
        >>> dataset2 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1', 'sensitive_2'],
        ...     real_target=['real_target_1', 'real_target_2'],
        ...     predicted_target=['predicted_target_1', 'predicted_target_2'],
        ...     positive_target=[1, 'yes']
        ... )
        >>> dataset2.get_real_target_for_one_group(['male', 'black'])
            real_target_1 real_target_2
        3         1           yes
        4         0            no
        """
        sensitive_group = convert_to_list(sensitive_group)

        result = None
        if self.groups_real_target is None:
            filtered_df = df_filtration(self.df, sensitive_group, self.sensitive)
            result = filtered_df[self.real_target]
        else:
            for item in self.groups_real_target:
                if set(item.sensitive_group) == set(sensitive_group):
                    result = item.data
        if result is None:
            raise (
                ValueError(f"{sensitive_group} group does not exist in the dataframe")
            )
        return result

    def get_predicted_target_for_all_groups(
        self,
    ) -> list[DatasetTargetResult]:
        """Retrieve the predicted target corresponding to each sensitive
        group present in the sensitive features.

        Tip
        ---
        If you have two sensitive attributes `gender` (male, female) and `race`
        (white, black), this function would return the predicted target for the
        combination of the two sensitive features; Hence, all of the following
        groups: (male, white), (female, white), (female, black), (male, black)

        Returns
        -------
        list[DatasetTargetResult]
            List of DatasetTargetResult dictionaries with standardized structure.

        Examples
        --------
        >>> import pandas as pd
        >>> from fair_mango.dataset.dataset import Dataset
        >>> data = {
        ...     'sensitive_1': ['male', 'female', 'female', 'male', 'male'],
        ...     'sensitive_2': ['white', 'white', 'black', 'black', 'black'],
        ...     'col-a': ['a', 'A', 'a', 'A', 'a'],
        ...     'col-b': ['B', 'B', 'b', 'B', 'b'],
        ...     'real_target_1': [0, 1, 0, 1, 0],
        ...     'real_target_2': ['no', 'yes', 'yes', 'yes', 'no'],
        ...     'predicted_target_1': [0, 1, 1, 0, 0],
        ...     'predicted_target_2': ['no', 'no', 'yes', 'yes', 'yes'],
        ... }
        >>> df = pd.DataFrame(data)
        >>> dataset1 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1'],
        ...     positive_target=[1]
        ... )
        >>> dataset1.get_predicted_target_for_all_groups()
        [
            {
                'sensitive_group': array(['male'], dtype=object),
                'data': 0    0
                        3    0
                        4    0
                        Name: predicted_target_1, dtype: int64
            },
            {
                'sensitive_group': array(['female'], dtype=object),
                'data': 1    1
                        2    1
                        Name: predicted_target_1, dtype: int64
            }
        ]
        >>> dataset2 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1', 'sensitive_2'],
        ...     real_target=['real_target_1'],
        ...     predicted_target=['predicted_target_1'],
        ...     positive_target=[1]
        ... )
        >>> dataset2.get_predicted_target_for_all_groups()
        [
            {
                'sensitive_group': array(['male', 'black'], dtype=object),
                'data': 3    0
                        4    0
                        Name: predicted_target_1, dtype: int64
            },
            {
                'sensitive_group': array(['female', 'black'], dtype=object),
                'data': 1
            },
            {
                'sensitive_group': array(['female', 'white'], dtype=object),
                'data': 1
            },
            {
                'sensitive_group': array(['male', 'white'], dtype=object),
                'data': 0
            }
        ]
        """
        if self.predicted_target is None:
            raise ValueError(
                "predicted_target parameter is required when creating the dataset"
            )
        self.groups_predicted_target = []
        for row in self.groups.values:
            result = self.df
            for i in range(len(self.sensitive)):
                result = result[result[self.sensitive[i]] == row[i]]

            sensitive_group = [str(x) for x in row[:-1]]
            target_data = result[self.predicted_target]

            self.groups_predicted_target.append(
                DatasetTargetResult(sensitive_group=sensitive_group, data=target_data)
            )
        return self.groups_predicted_target
