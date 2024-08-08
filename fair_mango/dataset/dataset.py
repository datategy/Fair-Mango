from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def check_column_existence_in_df(df: pd.DataFrame, columns: Sequence) -> None:
    """validate the columns existence in the dataset

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to check.
    columns : Sequence | None
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


def check_real_and_predicted_target_match(
    real_target: Sequence[str], predicted_target: Sequence[str]
) -> None:
    """Check that the number of real targets and number of predicted targets
    match.

    Parameters
    ----------
    real_target : Sequence[str]
        Sequence of column names corresponding to the real targets
        (true labels).
    predicted_target : Sequence[str]
        Sequence of column names corresponding to the predicted targets.

    Raises
    ------
    ValueError
        If the number of real targets and predicted targets does not match.
    """
    if len(real_target) != len(predicted_target):
        raise ValueError("real_target and predicted_target does not match")


def validate_columns(
    sensitive: Sequence[str],
    real_target: Sequence[str],
    predicted_target: Sequence[str] | None = None,
) -> None:
    """Make sure that the columns provided as parameters are different.
    A column cannot be a sensitive column and a target at the same time.
    A column cannot be a real target and a predicted target at the same time.

    Parameters
    ----------
    sensitive : Sequence[str]
        sequence of column names corresponding to sensitive features
        (Ex: gender, race...).
    real_target : Sequence[str]
        sequence of column names corresponding to the real targets
        (true labels). Every target will be processed independently.
    predicted_target : Sequence[str], optional
        sequence of column names corresponding to the predicted targets,
        by default None

    Raises
    ------
    AttributeError
        if the same column is assigned to different parameters at the same time.
    """
    overlap = set(sensitive).intersection(real_target)

    if predicted_target is not None:
        overlap.update(set(sensitive).intersection(predicted_target))
        overlap.update(set(real_target).intersection(predicted_target))

    if overlap:
        raise AttributeError(
            "Same column name can't be assigned to multiple" f" parameters {overlap}"
        )


def convert_to_list(variable: Sequence[str]) -> Sequence:
    """Convert a variable of type str to a list

    Parameters
    ----------
    variable : Sequence[str]
        sequence of values.

    Returns
    -------
    Sequence
        sequence of the values (not str).
    """
    if isinstance(variable, str):
        return [variable]
    else:
        return variable


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
    real_target : Sequence[str]
        Sequence of column names corresponding to the real targets
        (true labels).
    predicted_target : Sequence[str], optional
        Sequence of column names corresponding to the predicted targets,
        by default None.
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        Sequence of the positive labels corresponding to the provided targets,
        by default None.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sensitive: Sequence[str],
        real_target: Sequence[str],
        predicted_target: Sequence[str] | None = None,
        positive_target: Sequence[int | float | str | bool] | None = None,
    ):
        self.sensitive = convert_to_list(sensitive)
        check_column_existence_in_df(df, self.sensitive)
        self.real_target = convert_to_list(real_target)
        check_column_existence_in_df(df, self.real_target)
        if predicted_target is not None:
            self.predicted_target = convert_to_list(predicted_target)
            check_column_existence_in_df(df, self.predicted_target)
            check_real_and_predicted_target_match(
                self.real_target, self.predicted_target
            )
        else:
            self.predicted_target = []
        validate_columns(self.sensitive, self.real_target, self.predicted_target)
        self.df = df.copy()
        self.shape = df.shape
        self.positive_target = positive_target
        if isinstance(sensitive, str):
            self.groups = (
                df[[sensitive]]
                .groupby([sensitive])
                .size()
                .reset_index(name="Count")
                .sort_values("Count", ascending=False)
            )
        else:
            self.groups = (
                df[sensitive]
                .groupby(sensitive)
                .size()
                .reset_index(name="Count")
                .sort_values("Count", ascending=False)
            )
        self.n_groups: int = len(self.groups)
        self.groups_data: list[dict[str, np.ndarray | pd.DataFrame]] = []
        self.groups_real_target: (
            list[dict[str, np.ndarray | pd.Series | pd.DataFrame]] | None
        ) = None
        self.groups_predicted_target: (
            list[dict[str, np.ndarray | pd.Series | pd.DataFrame]] | None
        ) = None
        plt.style.use("fivethirtyeight")

    def plot_groups(
        self,
        sensitive: Sequence[str] = [],
        figsize: tuple[int, int] = (16, 6),
        dpi: int = 200,
    ):
        """Plot the distribution of the sensitive groups found within the
        sensitive features.

        Parameters
        ----------
        sensitive : Sequence[str], optional
            Sequence of column names corresponding to sensitive features
            (Ex: gender, race...), by default [].
        figsize : tuple[int, int], optional
            Figure size, by default (16, 6).
        dpi : int, optional
            Density of pixels per inch, by default 200.
        """
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if sensitive == []:
            sensitive = self.sensitive
        if len(sensitive) == 1:
            sns.barplot(x=sensitive[0], y="Count", data=self.groups, ax=ax)
            plt.show()
        elif len(sensitive) == 2:
            # Use the column with the least group as the color variable
            sensitive = (
                self.groups[sensitive].nunique().sort_values(ascending=False).index
            )
            sns.barplot(
                x=sensitive[0], y="Count", hue=sensitive[1], data=self.groups, ax=ax
            )
            plt.show()

    def get_data_for_all_groups(self) -> list[dict[str, np.ndarray | pd.DataFrame]]:
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
                'sensitive': array(['male'], dtype=object),
                'data':   sensitive_1 sensitive_2  ... predicted_target_1 predicted_target_2
                    0        male       white      ...         0                 no
                    3        male       black      ...         0                yes
                    4        male       black      ...         0                yes
                [3 rows x 8 columns]
            },
            {
                'sensitive': array(['female'], dtype=object),
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
                'sensitive': array(['male', 'black'], dtype=object),
                'data':   sensitive_1 sensitive_2  ... predicted_target_1 predicted_target_2
                    3        male       black      ...         0                yes
                    4        male       black      ...         0                yes
                [2 rows x 8 columns]
            },
            {
                'sensitive': array(['female', 'black'], dtype=object),
                'data':   sensitive_1 sensitive_2  ... predicted_target_1 predicted_target_2
                    2      female       black      ...         1                yes
                [1 rows x 8 columns]
            },
            {
                'sensitive': array(['female', 'white'], dtype=object),
                'data':   sensitive_1 sensitive_2  ... predicted_target_1 predicted_target_2
                    1      female       white      ...         1                 no
                [1 rows x 8 columns]
            },
            {
                'sensitive': array(['male', 'white'], dtype=object),
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
            self.groups_data.append({"sensitive": row[:-1], "data": result})
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
        sensitive : Sequence[str]
            Sequence of sensitive values must be in the same order as `sensitive`
            attribute, and so `sensitive_group` must be the same length as
            `sensitive`. For instance, if your `sensitive` attribute were
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
            result = self.df
            for i in range(len(sensitive_group)):
                result = result[result[self.sensitive[i]].isin(sensitive_group)]
        else:
            for item in self.groups_data:
                if (all(e1 in item["sensitive"] for e1 in sensitive_group)) and (
                    all(e2 in sensitive_group for e2 in item["sensitive"])
                ):
                    result = item["data"]
        if result is None:
            raise (
                ValueError(f"{sensitive_group} group does not exist in the dataframe")
            )
        return result

    def get_real_target_for_all_groups(
        self,
    ) -> list[dict[str, np.ndarray | pd.Series | pd.DataFrame]]:
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
        list[dict[str, np.ndarray | pd.Series | pd.DataFrame]]
            List of dictionaries with the sensitive group as keys and the
            corresponding real target as value.

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
                'sensitive': array(['male'], dtype=object),
                'data': 0    0
                        3    1
                        4    0
                        Name: real_target_1, dtype: int64
            },
            {
                'sensitive': array(['female'], dtype=object),
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
                'sensitive': array(['male', 'black'], dtype=object),
                'data': 3    1
                        4    0
                        Name: real_target_1, dtype: int64
            },
            {
                'sensitive': array(['female', 'black'], dtype=object),
                'data': 0
            },
            {
                'sensitive': array(['female', 'white'], dtype=object),
                'data': 1
            },
            {
                'sensitive': array(['male', 'white'], dtype=object),
                'data': 0
            }
        ]
        """
        self.groups_real_target = []
        for row in self.groups.values:
            result = self.df
            for i in range(len(self.sensitive)):
                result = result[result[self.sensitive[i]] == row[i]]
            self.groups_real_target.append(
                {"sensitive": row[:-1], "data": result[self.real_target].squeeze()}
            )
        return self.groups_real_target

    def get_real_target_for_one_group(
        self, sensitive_group: Sequence[str]
    ) -> pd.DataFrame:
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
            `sensitive`. For instance, if your `sensitive` attribute were
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
        result = None
        if self.groups_real_target is None:
            result = self.df
            for i in range(len(sensitive_group)):
                result = result[result[self.sensitive[i]].isin(sensitive_group)]
            result = result[self.real_target]
        else:
            for item in self.groups_real_target:
                if (item["sensitive"] == sensitive_group).all() or (
                    item["sensitive"] == sensitive_group[::-1]
                ).all():
                    result = item["data"]
        if result is None:
            raise (
                ValueError(f"{sensitive_group} group does not exist in the dataframe")
            )
        return result

    def get_predicted_target_for_all_groups(
        self,
    ) -> list[dict[str, np.ndarray | pd.Series | pd.DataFrame]]:
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
        list[dict[str, np.ndarray | pd.Series | pd.DataFrame]]
            List of dictionaries with the sensitive group as keys and the
            corresponding predicted target as value.

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
                'sensitive': array(['male'], dtype=object),
                'data': 0    0
                        3    0
                        4    0
                        Name: predicted_target_1, dtype: int64
            },
            {
                'sensitive': array(['female'], dtype=object),
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
                'sensitive': array(['male', 'black'], dtype=object),
                'data': 3    0
                        4    0
                        Name: predicted_target_1, dtype: int64
            },
            {
                'sensitive': array(['female', 'black'], dtype=object),
                'data': 1
            },
            {
                'sensitive': array(['female', 'white'], dtype=object),
                'data': 1
            },
            {
                'sensitive': array(['male', 'white'], dtype=object),
                'data': 0
            }
        ]
        """
        if self.predicted_target == []:
            raise ValueError(
                "predicted_target parameter is required when creating the dataset"
            )
        self.groups_predicted_target = []
        for row in self.groups.values:
            result = self.df
            for i in range(len(self.sensitive)):
                result = result[result[self.sensitive[i]] == row[i]]
            self.groups_predicted_target.append(
                {"sensitive": row[:-1], "data": result[self.predicted_target].squeeze()}
            )
        return self.groups_predicted_target

    def get_predicted_target_for_one_group(
        self, sensitive_group: Sequence[str]
    ) -> pd.DataFrame:
        """Retrieve the predicted target corresponding to a specific sensitive
        group present in the sensitive features.

        Tip
        ---
        If you have two sensitive attributes `gender` (male, female) and `race`
        (white, black), this function would return the predicted target for the
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
            `sensitive`. For instance, if your `sensitive` attribute were
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
        >>> dataset1.get_predicted_target_for_one_group(['female'])
        1    1
        2    1
        Name: predicted_target_1, dtype: int64
        >>> dataset2 = Dataset(
        ...     df=df,
        ...     sensitive=['sensitive_1', 'sensitive_2'],
        ...     real_target=['real_target_1', 'real_target_2'],
        ...     predicted_target=['predicted_target_1', 'predicted_target_2'],
        ...     positive_target=[1, 'yes']
        ... )
        >>> dataset2.get_real_target_for_one_group(['male', 'black'])
            predicted_target_1 predicted_target_2
        3           0                yes
        4           0                yes
        """
        if self.predicted_target == []:
            raise ValueError(
                "predicted_target parameter is required when creating the dataset"
            )
        result = None
        if self.groups_predicted_target is None:
            result = self.df
            for i in range(len(sensitive_group)):
                result = result[result[self.sensitive[i]].isin(sensitive_group)]
            result = result[self.predicted_target]
        else:
            for item in self.groups_predicted_target:
                if (item["sensitive"] == sensitive_group).all() or (
                    item["sensitive"] == sensitive_group[::-1]
                ).all():
                    result = item["data"]
        if result is None:
            raise (
                ValueError(f"{sensitive_group} group does not exist in the dataframe")
            )
        return result
