from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import pytest
from _pytest.python_api import RaisesContext
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from fair_mango.dataset.dataset import Dataset
from fair_mango.metrics.metrics import (
    ConfusionMatrix,
    DemographicParityDifference,
    DemographicParityRatio,
    DisparateImpactDifference,
    DisparateImpactRatio,
    EqualOpportunityDifference,
    PerformanceMetric,
    SelectionRate,
    encode_target,
    false_negative_rate,
    false_positive_rate,
    is_binary,
    true_negative_rate,
)

df = pd.read_csv("tests/data/heart_data.csv")

dataset1 = Dataset(df, ["Sex"], ["HeartDisease"])

dataset2 = Dataset(df, ["Sex"], ["HeartDisease"], ["HeartDiseasePred"])

dataset3 = Dataset(df, ["Sex", "ChestPainType"], ["HeartDisease"], ["HeartDiseasePred"])

dataset4 = Dataset(df, ["Sex"], ["HeartDisease", "ExerciseAngina"], [], [1, "Y"])

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


@pytest.mark.parametrize(
    "data, use_y_true, expected_groups, expected_result",
    [
        (dataset1, True, ["M", "F"], [0.63172414, 0.25906736]),
        (dataset1, False, [], pytest.raises(ValueError)),
        (
            dataset3,
            True,
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
                0.8286385,
                0.44,
                0.17699115,
                0.55714286,
                0.06666667,
                0.11320755,
                0.52777778,
                0.1,
            ],
        ),
        (
            dataset3,
            False,
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
                0.81924883,
                0.44,
                0.17699115,
                0.58571429,
                0.08333333,
                0.0754717,
                0.52777778,
                0.1,
            ],
        ),
        (
            dataset5,
            True,
            ["M", "F"],
            [[0.63172414, 0.45241379], [0.25906736, 0.22279793]],
        ),
        (
            dataset5,
            False,
            ["M", "F"],
            [[0.6262069, 0.45241379], [0.2642487, 0.22279793]],
        ),
    ],
)
def test_selectionrate(
    data: Dataset,
    use_y_true: bool,
    expected_groups: Sequence[str],
    expected_result: Sequence[float] | RaisesContext,
):
    if isinstance(expected_result, Sequence):
        sr = SelectionRate(data, use_y_true)
        result = sr()
        if use_y_true:
            assert result[0] == data.real_target
        else:
            assert result[0] == data.predicted_target
        for i, res in enumerate(result[1]):
            assert (res["sensitive"] == expected_groups[i]).all()
            assert (np.isclose(res["result"], expected_result[i], atol=0.0000002)).all()
    else:
        with expected_result:
            sr = SelectionRate(data, use_y_true)
            sr()


expected_result_2 = [
    {
        "sensitive": np.array(["M"]),
        "false_negative_rate": [0.021834061135371178],
        "false_positive_rate": [0.02247191011235955],
        "true_negative_rate": [0.9775280898876404],
        "true_positive_rate": [0.9781659388646288],
    },
    {
        "sensitive": np.array(["F"]),
        "false_negative_rate": [0.06],
        "false_positive_rate": [0.027972027972027972],
        "true_negative_rate": [0.972027972027972],
        "true_positive_rate": [0.94],
    },
]


expected_result_3 = [
    {"sensitive": np.array(["M", "ASY"]), "fpr": [0.0136986301369863]},
    {"sensitive": np.array(["M", "NAP"]), "fpr": [0.011904761904761904]},
    {"sensitive": np.array(["M", "ATA"]), "fpr": [0.010752688172043012]},
    {"sensitive": np.array(["F", "ASY"]), "fpr": [0.0967741935483871]},
    {"sensitive": np.array(["F", "ATA"]), "fpr": [0.017857142857142856]},
    {"sensitive": np.array(["F", "NAP"]), "fpr": [0.0]},
    {"sensitive": np.array(["M", "TA"]), "fpr": [0.17647058823529413]},
    {"sensitive": np.array(["F", "TA"]), "fpr": [0.0]},
]


expected_result_6 = [
    {
        "sensitive": np.array(["M", "ASY"]),
        "true_negative_rate": [0.9863013698630136, 1.0],
        "false_negative_rate": [0.014164305949008499, 0.0],
    },
    {
        "sensitive": np.array(["M", "NAP"]),
        "true_negative_rate": [0.9880952380952381, 1.0],
        "false_negative_rate": [0.015151515151515152, 0.0],
    },
    {
        "sensitive": np.array(["M", "ATA"]),
        "true_negative_rate": [0.989247311827957, 1.0],
        "false_negative_rate": [0.05, 0.0],
    },
    {
        "sensitive": np.array(["F", "ASY"]),
        "true_negative_rate": [0.9032258064516129, 1.0],
        "false_negative_rate": [0.02564102564102564, 0.0],
    },
    {
        "sensitive": np.array(["F", "ATA"]),
        "true_negative_rate": [0.9821428571428571, 1.0],
        "false_negative_rate": [0.0, 0.0],
    },
    {
        "sensitive": np.array(["F", "NAP"]),
        "true_negative_rate": [1.0, 1.0],
        "false_negative_rate": [0.3333333333333333, 0.0],
    },
    {
        "sensitive": np.array(["M", "TA"]),
        "true_negative_rate": [0.8235294117647058, 1.0],
        "false_negative_rate": [0.15789473684210525, 0.0],
    },
    {
        "sensitive": np.array(["F", "TA"]),
        "true_negative_rate": [1.0, 1.0],
        "false_negative_rate": [0.0, "ZERO"],
    },
]


@pytest.mark.parametrize(
    "data, metrics, zero_division, expected_result",
    [
        (dataset1, None, None, pytest.raises(ValueError)),
        (dataset2, None, None, expected_result_2),
        (dataset3, {"fpr": false_positive_rate}, None, expected_result_3),
        (
            dataset6,
            [true_negative_rate, false_negative_rate],
            "ZERO",
            expected_result_6,
        ),
    ],
)
def test_confusionmatrix(
    data: Dataset,
    metrics: dict[str, Callable] | None,
    zero_division: float | str | None,
    expected_result: Sequence[dict[str, Sequence]] | RaisesContext,
):
    if isinstance(expected_result, Sequence):
        cf = ConfusionMatrix(data, metrics, zero_division)
        result = cf()
        assert result[0] == data.real_target
        for i, res in enumerate(result[1]):
            for key in res.keys():
                if isinstance(res[key][0], object):
                    try:
                        assert (res[key] == expected_result[i][key]).all()
                    except AttributeError:
                        assert res[key] == expected_result[i][key]
                else:
                    assert (
                        np.isclose(res[key], expected_result[i][key], atol=0.0000002)
                    ).all()

    else:
        with expected_result:
            cf = ConfusionMatrix(data, metrics, zero_division)
            cf()


expected_result_2 = [
    {
        "HeartDisease": {
            "dpd": 0.3726567804180811,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {
        "HeartDisease": {
            "pr_dpd": 0.3726567804180811,
            "most_privileged": ("M",),
            "unp_dpd": -0.3726567804180811,
            "most_unprivileged": ("F",),
            "is_biased": True,
        }
    },
]


expected_result_3 = [
    {
        "HeartDisease": {
            "demographic_parity_difference": 0.7619718309859155,
            "privileged": ("M", "ASY"),
            "unprivileged": ("F", "ATA"),
        }
    },
    {
        "HeartDisease": {
            "pr_demographic_parity_difference": 0.5455262120526406,
            "most_privileged": ("M", "ASY"),
            "unp_demographic_parity_difference": -0.32529873764554856,
            "most_unprivileged": ("F", "ATA"),
            "is_biased": True,
        }
    },
]


expected_result_6 = [
    {
        "HeartDisease": {
            "demographic_parity_difference": 0.7619718309859155,
            "privileged": ("M", "ASY"),
            "unprivileged": ("F", "ATA"),
        },
        "ExerciseAngina": {
            "demographic_parity_difference": 0.6197183098591549,
            "privileged": ("M", "ASY"),
            "unprivileged": ("F", "TA"),
        },
    },
    {
        "HeartDisease": {
            "pr_demographic_parity_difference": 0.5455262120526406,
            "most_privileged": ("M", "ASY"),
            "unp_demographic_parity_difference": -0.32529873764554856,
            "most_unprivileged": ("F", "ATA"),
            "is_biased": True,
        },
        "ExerciseAngina": {
            "pr_demographic_parity_difference": 0.44419980257312147,
            "most_privileged": ("M", "ASY"),
            "unp_demographic_parity_difference": -0.26404969440876985,
            "most_unprivileged": ("F", "TA"),
            "is_biased": False,
        },
    },
]


@pytest.mark.parametrize(
    "data, label, sensitive, real_target, predicted_target, threshold, expected_result",
    [
        (dataset2, "dpd", [], [], [], 0.3, expected_result_2),
        (
            df,
            "demographic_parity_difference",
            ["Sex", "ChestPainType"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            0.1,
            expected_result_3,
        ),
        (
            dataset6,
            "demographic_parity_difference",
            [],
            [],
            [],
            0.45,
            expected_result_6,
        ),
    ],
)
def test_demographic_parity_difference(
    data: Dataset | pd.DataFrame,
    label: str,
    sensitive: Sequence[str],
    real_target: Sequence[str],
    predicted_target: Sequence[str],
    threshold: float,
    expected_result: Sequence[dict[str, dict]],
):
    if isinstance(data, Dataset):
        dpd = DemographicParityDifference(data, label)
    else:
        dpd = DemographicParityDifference(
            data, label, sensitive, real_target, predicted_target
        )
    result = dpd.summary
    assert result == expected_result[0]
    md = dpd.mean_differences(threshold)
    assert md == expected_result[1]


expected_result_2 = [
    {
        "HeartDisease": {
            "dpr": 0.4100957078534742,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {
        "HeartDisease": {
            "pr_dpr": 0.4100957078534742,
            "most_privileged": ("F",),
            "unp_dpr": 2.4384551724137933,
            "most_unprivileged": ("M",),
            "is_biased": False,
        }
    },
]


expected_result_3 = [
    {
        "HeartDisease": {
            "demographic_parity_ratio": 0.08045325779036827,
            "privileged": ("M", "ASY"),
            "unprivileged": ("F", "ATA"),
        }
    },
    {
        "HeartDisease": {
            "pr_demographic_parity_ratio": 0.30145207723707795,
            "most_privileged": ("F", "ATA"),
            "unp_demographic_parity_ratio": 5.3797187266238415,
            "most_unprivileged": ("M", "ASY"),
            "is_biased": True,
        }
    },
]


expected_result_6 = [
    {
        "HeartDisease": {
            "demographic_parity_ratio": 0.08045325779036827,
            "privileged": ("M", "ASY"),
            "unprivileged": ("F", "ATA"),
        },
        "ExerciseAngina": {
            "demographic_parity_ratio": 0.0,
            "privileged": ("F", "TA"),
            "unprivileged": ("M", "ASY"),
        },
    },
    {
        "HeartDisease": {
            "pr_demographic_parity_ratio": 0.341659585454887,
            "most_privileged": ("M", "ASY"),
            "unp_demographic_parity_ratio": 4.489065560514,
            "most_unprivileged": ("F", "ATA"),
            "is_biased": True,
        },
        "ExerciseAngina": {
            "pr_demographic_parity_ratio": 0.28322304584791763,
            "most_privileged": ("M", "ASY"),
            "unp_demographic_parity_ratio": np.inf,
            "most_unprivileged": ("F", "TA"),
            "is_biased": True,
        },
    },
]


@pytest.mark.parametrize(
    "data, label, zero_division, sensitive, real_target, predicted_target, threshold, expected_result",
    [
        (dataset1, "dpr", None, [], [], [], 1.2, pytest.raises(ValueError)),
        (
            df,
            "dpr",
            None,
            ["Sex"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            0.4,
            expected_result_2,
        ),
        (
            dataset3,
            "demographic_parity_ratio",
            "Error",
            [],
            [],
            [],
            0.8,
            expected_result_3,
        ),
        (
            dataset6,
            "demographic_parity_ratio",
            np.nan,
            [],
            [],
            [],
            0.3,
            expected_result_6,
        ),
    ],
)
def test_demographic_parity_ratio(
    data: Dataset,
    label: str,
    zero_division: float | str | None,
    sensitive: Sequence[str],
    real_target: Sequence[str],
    predicted_target: Sequence[str],
    threshold: float,
    expected_result: Sequence[dict[str, dict]],
):
    if isinstance(expected_result, Sequence):
        if isinstance(data, Dataset):
            dpr = DemographicParityRatio(data, label, zero_division)
        else:
            dpr = DemographicParityRatio(
                data, label, zero_division, sensitive, real_target, predicted_target
            )
        result = dpr.summary
        assert result == expected_result[0]
        mr = dpr.mean_ratios(threshold)
        assert mr == expected_result[1]
    else:
        with expected_result:
            dpr = DemographicParityRatio(data, label, zero_division)
            dpr.mean_ratios(threshold)


expected_result_2 = [
    {
        "HeartDiseasePred": {
            "did": 0.3619581918885117,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {
        "HeartDiseasePred": {
            "pr_did": 0.3619581918885117,
            "most_privileged": ("M",),
            "unp_did": -0.3619581918885117,
            "most_unprivileged": ("F",),
            "is_biased": True,
        }
    },
]


expected_result_3 = [
    {
        "HeartDiseasePred": {
            "disparate_impact_difference": 0.7437771281778722,
            "privileged": ("M", "ASY"),
            "unprivileged": ("F", "NAP"),
        }
    },
    {
        "HeartDiseasePred": {
            "pr_disparate_impact_difference": 0.5350647912366394,
            "most_privileged": ("M", "ASY"),
            "unp_disparate_impact_difference": -0.31496621239521455,
            "most_unprivileged": ("F", "NAP"),
            "is_biased": True,
        }
    },
]


expected_result_6 = [
    {
        "HeartDiseasePred": {
            "disparate_impact_difference": 0.7437771281778722,
            "privileged": ("M", "ASY"),
            "unprivileged": ("F", "NAP"),
        },
        "ExerciseAngina": {
            "disparate_impact_difference": 0.6197183098591549,
            "privileged": ("M", "ASY"),
            "unprivileged": ("F", "TA"),
        },
    },
    {
        "HeartDiseasePred": {
            "pr_disparate_impact_difference": 0.5350647912366394,
            "most_privileged": ("M", "ASY"),
            "unp_disparate_impact_difference": -0.31496621239521455,
            "most_unprivileged": ("F", "NAP"),
            "is_biased": True,
        },
        "ExerciseAngina": {
            "pr_disparate_impact_difference": 0.44419980257312147,
            "most_privileged": ("M", "ASY"),
            "unp_disparate_impact_difference": -0.26404969440876985,
            "most_unprivileged": ("F", "TA"),
            "is_biased": False,
        },
    },
]


@pytest.mark.parametrize(
    "data, label, sensitive, real_target, predicted_target, positive_target, threshold, expected_result",
    [
        (dataset2, "did", [], [], [], [], 0.3, expected_result_2),
        (
            dataset3,
            "disparate_impact_difference",
            [],
            [],
            [],
            [],
            0.2,
            expected_result_3,
        ),
        (
            df,
            "disparate_impact_difference",
            ["Sex", "ChestPainType"],
            ["HeartDisease", "ExerciseAngina"],
            ["HeartDiseasePred", "ExerciseAngina"],
            [1, "Y"],
            0.45,
            expected_result_6,
        ),
    ],
)
def test_disparate_impact_difference(
    data: Dataset,
    label: str,
    sensitive: Sequence[str],
    real_target: Sequence[str],
    predicted_target: Sequence[str],
    positive_target: Sequence[int | float | str | bool] | None,
    threshold: float,
    expected_result: Sequence[dict[str, dict]],
):
    if isinstance(expected_result, Sequence):
        if isinstance(data, Dataset):
            did = DisparateImpactDifference(data, label)
        else:
            did = DisparateImpactDifference(
                data, label, sensitive, real_target, predicted_target, positive_target
            )

        result = did.summary
        assert result == expected_result[0]
        md = did.mean_differences(threshold)
        assert md == expected_result[1]
    else:
        with expected_result:
            DisparateImpactDifference(data, label)


expected_result_2 = [
    {
        "HeartDiseasePred": {
            "dir": 0.4219830636141608,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {
        "HeartDiseasePred": {
            "pr_dir": 0.4219830636141608,
            "most_privileged": ("F",),
            "unp_dir": 2.369763353617309,
            "most_unprivileged": ("M",),
            "is_biased": False,
        }
    },
]


expected_result_3 = [
    {
        "HeartDiseasePred": {
            "disparate_impact_ratio": 0.09212304698059144,
            "privileged": ("M", "ASY"),
            "unprivileged": ("F", "NAP"),
        }
    },
    {
        "HeartDiseasePred": {
            "pr_disparate_impact_ratio": 0.3746136017304162,
            "most_privileged": ("F", "NAP"),
            "unp_disparate_impact_ratio": 5.474312084041965,
            "most_unprivileged": ("M", "ASY"),
            "is_biased": True,
        }
    },
]


expected_result_6 = [
    {
        "HeartDiseasePred": {
            "disparate_impact_ratio": 0.09212304698059144,
            "privileged": ("M", "ASY"),
            "unprivileged": ("F", "NAP"),
        },
        "ExerciseAngina": {
            "disparate_impact_ratio": 0.0,
            "privileged": ("F", "TA"),
            "unprivileged": ("M", "ASY"),
        },
    },
    {
        "HeartDiseasePred": {
            "pr_disparate_impact_ratio": 0.3468836645650188,
            "most_privileged": ("M", "ASY"),
            "unp_disparate_impact_ratio": 4.113253804597716,
            "most_unprivileged": ("F", "NAP"),
            "is_biased": True,
        },
        "ExerciseAngina": {
            "pr_disparate_impact_ratio": 0.28322304584791763,
            "most_privileged": ("M", "ASY"),
            "unp_disparate_impact_ratio": np.inf,
            "most_unprivileged": ("F", "TA"),
            "is_biased": True,
        },
    },
]


@pytest.mark.parametrize(
    "data, label, zero_division, sensitive, real_target, predicted_target, threshold, expected_result",
    [
        (dataset1, "dir", None, [], [], [], 1.2, pytest.raises(ValueError)),
        (
            df,
            "dir",
            None,
            ["Sex"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            0.4,
            expected_result_2,
        ),
        (
            dataset3,
            "disparate_impact_ratio",
            "Error",
            [],
            [],
            [],
            0.8,
            expected_result_3,
        ),
        (
            dataset6,
            "disparate_impact_ratio",
            np.nan,
            [],
            [],
            [],
            0.3,
            expected_result_6,
        ),
    ],
)
def test_disparate_impact_ratio(
    data: Dataset,
    label: str,
    zero_division: float | str | None,
    sensitive: Sequence[str],
    real_target: Sequence[str],
    predicted_target: Sequence[str],
    threshold: float,
    expected_result: Sequence[dict[str, dict]],
):
    if isinstance(expected_result, Sequence):
        if isinstance(data, Dataset):
            dira = DisparateImpactRatio(data, label, zero_division)
        else:
            dira = DisparateImpactRatio(
                data, label, zero_division, sensitive, real_target, predicted_target
            )

        result = dira.summary
        assert result == expected_result[0]
        mr = dira.mean_ratios(threshold)
        assert mr == expected_result[1]
    else:
        with expected_result:
            dira = DisparateImpactRatio(data, label, zero_division)
            dira.mean_ratios(threshold)


expected_result_2 = [
    {
        "HeartDisease": {
            "eod": 0.03816593886462882,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {
        "HeartDisease": {
            "pr_eod": 0.03816593886462882,
            "most_privileged": ("M",),
            "unp_eod": -0.03816593886462882,
            "most_unprivileged": ("F",),
            "is_biased": False,
        }
    },
]


expected_result_3 = [
    {
        "HeartDisease": {
            "equal_opportunity_difference": 0.33333333333333337,
            "privileged": ("F", "ATA"),
            "unprivileged": ("F", "NAP"),
        }
    },
    {
        "HeartDisease": {
            "pr_equal_opportunity_difference": 0.08516927384528401,
            "most_privileged": ("F", "ATA"),
            "unp_equal_opportunity_difference": -0.29578310710709704,
            "most_unprivileged": ("F", "NAP"),
            "is_biased": True,
        }
    },
]


expected_result_6 = [
    {
        "HeartDisease": {
            "equal_opportunity_difference": 0.33333333333333337,
            "privileged": ("F", "ATA"),
            "unprivileged": ("F", "NAP"),
        },
        "ExerciseAngina": {
            "equal_opportunity_difference": 0.0,
            "privileged": None,
            "unprivileged": None,
        },
    },
    {
        "HeartDisease": {
            "pr_equal_opportunity_difference": 0.08516927384528401,
            "most_privileged": ("F", "ATA"),
            "unp_equal_opportunity_difference": -0.29578310710709704,
            "most_unprivileged": ("F", "NAP"),
            "is_biased": True,
        },
        "ExerciseAngina": {
            "pr_equal_opportunity_difference": 0.0,
            "most_privileged": np.nan,
            "unp_equal_opportunity_difference": 0.0,
            "most_unprivileged": np.nan,
            "is_biased": False,
        },
    },
]


@pytest.mark.parametrize(
    "data, label, sensitive, real_target, predicted_target, threshold, expected_result",
    [
        (dataset4, "eod", [], [], [], 1.2, pytest.raises(ValueError)),
        (dataset2, "eod", [], [], [], 0.1, expected_result_2),
        (
            df,
            "equal_opportunity_difference",
            ["Sex", "ChestPainType"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            0.2,
            expected_result_3,
        ),
        (dataset6, "equal_opportunity_difference", [], [], [], 0.2, expected_result_6),
    ],
)
def test_equal_opportunity_difference(
    data: Dataset | pd.DataFrame,
    label: str,
    sensitive: Sequence[str],
    real_target: Sequence[str],
    predicted_target: Sequence[str],
    threshold: float,
    expected_result: Sequence[dict[str, dict]] | RaisesContext,
):
    if isinstance(expected_result, Sequence):
        if isinstance(data, Dataset):
            eod = EqualOpportunityDifference(data, label)
        else:
            eod = EqualOpportunityDifference(
                data, label, sensitive, real_target, predicted_target
            )
        result = eod.summary
        assert result == expected_result[0]
        md = eod.mean_differences(threshold)
        assert md == expected_result[1]
    else:
        with expected_result:
            eod = EqualOpportunityDifference(data, label)
            eod.mean_differences(threshold)


expected_result_2 = [
    {
        "sensitive": np.array(["M"], dtype=object),
        "accuracy": [0.9779310344827586],
        "balanced accuracy": [0.9778470143761346],
        "precision": [0.986784140969163],
        "recall": [0.9781659388646288],
        "f1-score": [0.9824561403508771],
    },
    {
        "sensitive": np.array(["F"], dtype=object),
        "accuracy": [0.9637305699481865],
        "balanced accuracy": [0.956013986013986],
        "precision": [0.9215686274509803],
        "recall": [0.94],
        "f1-score": [0.9306930693069307],
    },
]


expected_result_3 = [
    {
        "sensitive": np.array(["M", "ASY"], dtype=object),
        "acc": [0.9859154929577465],
        "balanced_acc": [0.9860685319570026],
    },
    {
        "sensitive": np.array(["M", "NAP"], dtype=object),
        "acc": [0.9866666666666667],
        "balanced_acc": [0.9864718614718615],
    },
    {
        "sensitive": np.array(["M", "ATA"], dtype=object),
        "acc": [0.9823008849557522],
        "balanced_acc": [0.9696236559139785],
    },
    {
        "sensitive": np.array(["F", "ASY"], dtype=object),
        "acc": [0.9428571428571428],
        "balanced_acc": [0.9387923904052936],
    },
    {
        "sensitive": np.array(["F", "ATA"], dtype=object),
        "acc": [0.9833333333333333],
        "balanced_acc": [0.9910714285714286],
    },
    {
        "sensitive": np.array(["F", "NAP"], dtype=object),
        "acc": [0.9622641509433962],
        "balanced_acc": [0.8333333333333333],
    },
    {
        "sensitive": np.array(["M", "TA"], dtype=object),
        "acc": [0.8333333333333334],
        "balanced_acc": [0.8328173374613003],
    },
    {
        "sensitive": np.array(["F", "TA"], dtype=object),
        "acc": [1.0],
        "balanced_acc": [1.0],
    },
]


expected_result_6 = [
    {
        "sensitive": np.array(["M", "ASY"], dtype=object),
        "precision_score": [0.997134670487106, 1.0],
        "recall_score": [0.9858356940509915, 1.0],
        "f1_score": [0.9914529914529915, 1.0],
    },
    {
        "sensitive": np.array(["M", "NAP"], dtype=object),
        "precision_score": [0.9848484848484849, 1.0],
        "recall_score": [0.9848484848484849, 1.0],
        "f1_score": [0.9848484848484849, 1.0],
    },
    {
        "sensitive": np.array(["M", "ATA"], dtype=object),
        "precision_score": [0.95, 1.0],
        "recall_score": [0.95, 1.0],
        "f1_score": [0.95, 1.0],
    },
    {
        "sensitive": np.array(["F", "ASY"], dtype=object),
        "precision_score": [0.926829268292683, 1.0],
        "recall_score": [0.9743589743589743, 1.0],
        "f1_score": [0.95, 1.0],
    },
    {
        "sensitive": np.array(["F", "ATA"], dtype=object),
        "precision_score": [0.8, 1.0],
        "recall_score": [1.0, 1.0],
        "f1_score": [0.8888888888888888, 1.0],
    },
    {
        "sensitive": np.array(["F", "NAP"], dtype=object),
        "precision_score": [1.0, 1.0],
        "recall_score": [0.6666666666666666, 1.0],
        "f1_score": [0.8, 1.0],
    },
    {
        "sensitive": np.array(["M", "TA"], dtype=object),
        "precision_score": [0.8421052631578947, 1.0],
        "recall_score": [0.8421052631578947, 1.0],
        "f1_score": [0.8421052631578947, 1.0],
    },
    {
        "sensitive": np.array(["F", "TA"], dtype=object),
        "precision_score": [1.0, 0.0],
        "recall_score": [1.0, 0.0],
        "f1_score": [1.0, 0.0],
    },
]


@pytest.mark.parametrize(
    "data, metrics, sensitive, real_target, predicted_target, positive_target, expected_result",
    [
        (dataset1, None, [], [], [], [], pytest.raises(ValueError)),
        (
            df,
            None,
            ["Sex"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            [],
            expected_result_2,
        ),
        (
            dataset3,
            {
                "acc": accuracy_score,
                "balanced_acc": balanced_accuracy_score,
            },
            [],
            [],
            [],
            [],
            expected_result_3,
        ),
        (
            dataset6,
            [precision_score, recall_score, f1_score],
            [],
            [],
            [],
            [],
            expected_result_6,
        ),
    ],
)
def test_performancemetrics(
    data: Dataset,
    metrics: dict[str, Callable] | None,
    sensitive: Sequence[str],
    real_target: Sequence[str],
    predicted_target: Sequence[str],
    positive_target: Sequence[int | float | str | bool] | None,
    expected_result: Sequence[dict[str, Sequence]] | RaisesContext,
):
    if isinstance(expected_result, Sequence):
        if isinstance(data, Dataset):
            pm = PerformanceMetric(data, metrics)
        else:
            pm = PerformanceMetric(
                data, metrics, sensitive, real_target, predicted_target, positive_target
            )
        result = pm()
        assert result[0] == pm.data.real_target
        for i, res in enumerate(result[1]):
            for key in res.keys():
                if isinstance(res[key][0], object):
                    try:
                        assert (res[key] == expected_result[i][key]).all()
                    except AttributeError:
                        assert res[key] == expected_result[i][key]
                else:
                    assert (
                        np.isclose(res[key], expected_result[i][key], atol=0.0000002)
                    ).all()

    else:
        with expected_result:
            cf = ConfusionMatrix(data, metrics)
            cf()
