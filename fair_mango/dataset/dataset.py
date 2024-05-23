from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def check_column_in_df(df: pd.DataFrame, columns: Sequence) -> None:
    """validate the columns existance in the dataset

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe
    columns : Sequence | None
        a list of column names
    """
    if columns != []:
        if isinstance(columns, str):
            if columns not in df.columns:
                raise (
                    KeyError(
                        f"{columns} column does not exist in the dataframe provided"
                    )
                )
        else:
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
    """check that number of real targets and number of predicted targets match

    Parameters
    ----------
    real_target : Sequence[str]
        real targets
    predicted_target : Sequence[str]
        predicted targets

    Raises
    ------
    ValueError
        if the number of targets and predicted targets does not match
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
    """A class for handling datasets with sensitive attributes and target variables."""

    def __init__(
        self,
        df: pd.DataFrame,
        sensitive: Sequence[str],
        real_target: Sequence[str],
        predicted_target: Sequence[str] = [],
        positive_target: Sequence[int | float | str | bool] | None = None,
    ):
        check_column_in_df(df, sensitive)
        check_column_in_df(df, real_target)
        check_column_in_df(df, predicted_target)
        if predicted_target != []:
            check_real_and_predicted_target_match(real_target, predicted_target)
        self.df = df.copy()
        self.shape = df.shape
        self.sensitive = sensitive
        self.real_target = real_target
        self.predicted_target = predicted_target
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
        self.groups_predicted_target: list[dict] = []
        plt.style.use("fivethirtyeight")

    def plot_groups(
        self,
        sensitive: Sequence[str] = [],
        figsize: tuple[int, int] = (16, 6),
        dpi: int = 200,
    ):
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
        """Retrieve data for all unique groups.

        Returns
        -------
        list[dict]
            list of dictionaries with the sensitive group and the corresponding dataframe.
        """
        for row in self.groups.values:
            result = self.df
            for i in range(len(self.sensitive)):
                result = result[result[self.sensitive[i]] == row[i]]
            self.groups_data.append({"sensitive": row[:-1], "data": result})
        return self.groups_data

    def get_data_for_one_group(self, sensitive: Sequence[str]) -> pd.DataFrame:
        """Retrieve data for a specific group

        Parameters
        ----------
        sensitive : Sequence[str]
            the sensitive group

        Returns
        -------
        pd.DataFrame
            the dataframe corresponding to the sensitive group
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
        """Retrieve real target for all unique groups.

        Returns
        -------
        list[dict]
            list of dictionaries with the sensitive group and the corresponding dataframe.
        """
        self.groups_real_target: list[dict] = []
        for row in self.groups.values:
            result = self.df
            for i in range(len(self.sensitive)):
                result = result[result[self.sensitive[i]] == row[i]]
            self.groups_real_target.append(
                {"sensitive": row[:-1], "data": result[self.real_target].squeeze()}
            )
        return self.groups_real_target

    def get_real_target_for_one_group(self, sensitive: Sequence[str]) -> pd.DataFrame:
        """Retrieve real target for a specific group

        Parameters
        ----------
        sensitive : Sequence[str]
            the sensitive group

        Returns
        -------
        pd.DataFrame
            the dataframe corresponding to the sensitive group
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
        """Retrieve predicted target for all unique groups.

        Returns
        -------
        list[dict]
            list of dictionaries with the sensitive group and the corresponding dataframe.
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
    ) -> pd.DataFrame:
        """Retrieve predicted target for a specific group

        Parameters
        ----------
        sensitive : Sequence[str]
            the sensitive group

        Returns
        -------
        pd.DataFrame
            the dataframe corresponding to the sensitive group
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
            result = result[self.real_target]
        else:
            for item in self.groups_predicted_target:
                if (item["sensitive"] == sensitive).all() or (
                    item["sensitive"] == sensitive[::-1]
                ).all():
                    result = item["data"]
        if result is None:
            raise (ValueError(f"{sensitive} group does not exist in the dataframe"))
        return result.squeeze()
