from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def check_column_in_df(df: pd.DataFrame, columns: Sequence) -> None:
    """validate the columns existance in the dataset

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe to check
    columns : Sequence | None
        sequence of column names to check if they exist in the dataframe.

    Raises
    ------
    KeyError
        If the one of the columns does not exist in the dataframe.
    """

    if isinstance(columns, str):
        columns = [columns]
    if columns != []:
        for column in columns:
            if column not in df.columns:
                raise (
                    KeyError(
                        f"{column} column does not exist in the dataframe provided"
                    )
                )


def check_real_and_predicted_target_match(
    real_target: Sequence[str], predicted_target: Sequence[str]
) -> None:
    """check that the number of real targets and number of predicted targets
    match.

    Parameters
    ----------
    real_target : Sequence[str]
        sequence of column names corresponding to the real targets
        (true labels).
    predicted_target : Sequence[str]
        sequence of column names corresponding to the predicted targets.

    Raises
    ------
    ValueError
        if the number of real targets and predicted targets does not match.
    """
    if isinstance(real_target, str) and isinstance(predicted_target, str):
        return None
    elif isinstance(real_target, str):
        if len(predicted_target) != 1:
            raise ValueError("real_target and predicted_target does not match")
    elif isinstance(predicted_target, str):
        if len(real_target) != 1:
            raise ValueError("real_target and predicted_target does not match")
    elif len(real_target) != len(predicted_target):
        raise ValueError("real_target and predicted_target does not match")


class Dataset:
    """A class for handling datasets with sensitive attributes and target
    variables.

    This class separates a dataframe into different demographic groups present
    in the dataframe. Any object of this class will serve as a building block
    for evaluating the performance of different demographic groups and
    calculating the fairness metrics.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    sensitive : Sequence[str]
        sequence of column names corresponding to sensitive features
        (Ex: gender, race...).
    real_target : Sequence[str]
        sequence of column names corresponding to the real targets
        (true labels). Every target will be processed independently.
    predicted_target : Sequence[str], optional
        sequence of column names corresponding to the predicted targets,
        by default None
    positive_target : Sequence[int  |  float  |  str  |  bool] | None, optional
        sequence of the positive labels corresponding to the provided targets,
        by default None
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sensitive: Sequence[str],
        real_target: Sequence[str],
        predicted_target: Sequence[str] | None = None,
        positive_target: Sequence[int | float | str | bool] | None = None,
    ):
        check_column_in_df(df, sensitive)
        check_column_in_df(df, real_target)
        if predicted_target is not None:
            check_column_in_df(df, predicted_target)
            check_real_and_predicted_target_match(real_target, predicted_target)
            self.predicted_target = predicted_target
        else:
            self.predicted_target = []
        self.df = df.copy()
        self.shape = df.shape
        self.sensitive = sensitive
        self.real_target = real_target
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
        self.groups_data: list[dict] = []
        self.groups_real_target: list[dict] | None = None
        self.groups_predicted_target: list[dict] | None = None
        plt.style.use("fivethirtyeight")

    def plot_groups(
        self,
        sensitive: Sequence[str] = [],
        figsize: tuple[int, int] = (16, 6),
        dpi: int = 200,
    ):
        """Plot the distribution of the demographic groups found within the
        sensitive features.

        Parameters
        ----------
        sensitive : Sequence[str], optional
            sequence of column names corresponding to sensitive features
            (Ex: gender, race...), by default []
        figsize : tuple[int, int], optional
            figure size, by default (16, 6)
        dpi : int, optional
            density of pixels per inch, by default 200
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

    def get_data_for_all_groups(self) -> list[dict]:
        """Retrieve data corresponding to each demographic group present in
        the sensitive features.

        Returns
        -------
        list[dict]
            list of dictionaries with the sensitive group as keys and the
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

    def get_data_for_one_group(self, sensitive: Sequence[str]) -> pd.DataFrame:
        """Retrieve data corresponding to a specific demographic group present
        in the sensitive features.

        Parameters
        ----------
        sensitive : Sequence[str]
            sequence of column names corresponding to sensitive features
            (Ex: gender, race...).

        Returns
        -------
        pd.DataFrame
            the dataframe corresponding to the sensitive group specified.

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
            for i in range(len(sensitive)):
                result = result[result[self.sensitive[i]].isin(sensitive)]
        else:
            for item in self.groups_data:
                if (all(e1 in item["sensitive"] for e1 in sensitive)) and (
                    all(e2 in sensitive for e2 in item["sensitive"])
                ):
                    result = item["data"]
        if result is None:
            raise (ValueError(f"{sensitive} group does not exist in the dataframe"))
        return result

    def get_real_target_for_all_groups(self) -> list[dict]:
        """Retrieve the real target corresponding to each demographic group
        present in the sensitive features.

        Returns
        -------
        list[dict]
            list of dictionaries with the sensitive group as keys and the
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
        self, sensitive: Sequence[str]
    ) -> pd.Series | pd.DataFrame:
        """Retrieve the real target corresponding to a specific demographic
        group present in the sensitive features.

        Parameters
        ----------
        sensitive : Sequence[str]
            sequence of column names corresponding to sensitive features
            (Ex: gender, race...).

        Returns
        -------
        pd.Seies | pd.DataFrame
            the pandas series or dataframe corresponding to the sensitive group

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
            for i in range(len(sensitive)):
                result = result[result[self.sensitive[i]].isin(sensitive)]
            result = result[self.real_target]
        else:
            for item in self.groups_real_target:
                if (item["sensitive"] == sensitive).all() or (
                    item["sensitive"] == sensitive[::-1]
                ).all():
                    result = item["data"]
        if result is None:
            raise (ValueError(f"{sensitive} group does not exist in the dataframe"))
        return result.squeeze()

    def get_predicted_target_for_all_groups(self) -> list[dict]:
        """Retrieve the predicted target corresponding to each demographic
        group present in the sensitive features.

        Returns
        -------
        list[dict]
            list of dictionaries with the sensitive group as keys and the
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
        self, sensitive: Sequence[str]
    ) -> pd.Series | pd.DataFrame:
        """Retrieve the predicted target corresponding to a specific
        demographic group present in the sensitive features.

        Parameters
        ----------
        sensitive : Sequence[str]
            sequence of column names corresponding to sensitive features
            (Ex: gender, race...).

        Returns
        -------
        pd.Seies | pd.DataFrame
            the pandas series or dataframe corresponding to the sensitive group

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
            for i in range(len(sensitive)):
                result = result[result[self.sensitive[i]].isin(sensitive)]
            result = result[self.predicted_target]
        else:
            for item in self.groups_predicted_target:
                if (item["sensitive"] == sensitive).all() or (
                    item["sensitive"] == sensitive[::-1]
                ).all():
                    result = item["data"]
        if result is None:
            raise (ValueError(f"{sensitive} group does not exist in the dataframe"))
        return result.squeeze()
