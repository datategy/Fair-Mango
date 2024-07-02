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
from fair_mango.metrics.base import encode_target, is_binary
from fair_mango.metrics.metrics import (
    ConfusionMatrix,
    DemographicParityDifference,
    DemographicParityRatio,
    DisparateImpactDifference,
    DisparateImpactRatio,
    EqualisedOddsDifference,
    EqualisedOddsRatio,
    EqualOpportunityDifference,
    EqualOpportunityRatio,
    FalsePositiveRateDifference,
    FalsePositiveRateRatio,
    PerformanceMetric,
    SelectionRate,
    false_negative_rate,
    false_positive_rate,
    is_binary,
    super_set,
    true_negative_rate,
)

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
        "false_negative_rate": [0.0, np.nan],
    },
]


@pytest.mark.parametrize(
    "data, metrics, expected_result",
    [
        (dataset1, None, pytest.raises(ValueError)),
        (dataset2, {"sensitive": false_positive_rate}, pytest.raises(KeyError)),
        (dataset2, None, expected_result_2),
        (dataset3, {"fpr": false_positive_rate}, expected_result_3),
        (
            dataset6,
            [true_negative_rate, false_negative_rate],  # type: ignore[list-item]
            expected_result_6,
        ),
    ],
)
def test_confusionmatrix(
    data: Dataset,
    metrics: dict[str, Callable] | Sequence[Callable] | None,
    expected_result: Sequence[dict[str, Sequence]] | RaisesContext,
):
    if isinstance(expected_result, Sequence):
        cf = ConfusionMatrix(data, metrics)
        result = cf()
        assert result[0] == data.real_target
        for i, res in enumerate(result[1]):
            for key in res.keys():
                if key == "sensitive":
                    assert (res[key] == expected_result[i][key]).all()
                else:
                    for val, expected_val in zip(res[key], expected_result[i][key]):
                        if np.isnan(val) and np.isnan(expected_val):
                            assert True
                        else:
                            assert (np.isclose(val, expected_val, atol=0.0000002)).all()

    else:
        with expected_result:
            cf = ConfusionMatrix(data, metrics)
            cf()


expected_result_2 = [
    {
        "HeartDisease": {
            "dpd": 0.3726567804180811,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {"HeartDisease": {("M",): 0.3726567804180811, ("F",): -0.3726567804180811}},
    {"HeartDisease": True},
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
            ("F", "ATA"): -0.32529873764554856,
            ("F", "TA"): -0.28720349955031044,
            ("F", "NAP"): -0.27210915992766893,
            ("M", "ATA"): -0.19921361333033571,
            ("M", "NAP"): 0.10136792902111814,
            ("M", "TA"): 0.20168538933857846,
            ("F", "ASY"): 0.23524548004152632,
            ("M", "ASY"): 0.5455262120526406,
        }
    },
    {"HeartDisease": True},
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
            ("M", "ASY"): 0.5455262120526406,
            ("F", "ASY"): 0.23524548004152632,
            ("M", "TA"): 0.20168538933857846,
            ("M", "NAP"): 0.10136792902111814,
            ("M", "ATA"): -0.19921361333033571,
            ("F", "NAP"): -0.27210915992766893,
            ("F", "TA"): -0.28720349955031044,
            ("F", "ATA"): -0.32529873764554856,
        },
        "ExerciseAngina": {
            ("M", "ASY"): 0.44419980257312147,
            ("F", "ASY"): 0.27472581579531175,
            ("M", "NAP"): 0.08642649606742059,
            ("M", "TA"): -0.07357350393257941,
            ("M", "ATA"): -0.14268433410535647,
            ("F", "NAP"): -0.1562329828184734,
            ("F", "ATA"): -0.16881159917067465,
            ("F", "TA"): -0.26404969440876985,
        },
    },
    {"HeartDisease": True, "ExerciseAngina": False},
]


@pytest.mark.parametrize(
    "data, label, sensitive, real_target, predicted_target, threshold, expected_result",
    [
        (dataset2, "dpd", None, None, None, 0.3, expected_result_2),
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
            None,
            None,
            None,
            0.45,
            expected_result_6,
        ),
    ],
)
def test_demographic_parity_difference(
    data: Dataset | pd.DataFrame,
    label: str,
    sensitive: Sequence[str] | None,
    real_target: Sequence[str] | None,
    predicted_target: Sequence[str] | None,
    threshold: float,
    expected_result: Sequence[dict[str, dict]],
):
    if isinstance(data, Dataset):
        dpd = DemographicParityDifference(data, label)
    else:
        dpd = DemographicParityDifference(
            data, label, sensitive, real_target, predicted_target
        )
    result = dpd.summary()
    assert result == expected_result[0]
    ranking = dpd.rank()
    assert ranking == expected_result[1]
    is_biased = dpd.is_biased(threshold)
    assert is_biased == expected_result[2]


expected_result_1 = [
    {
        "HeartDisease": {
            "dpr": 0.4100957078534742,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {"HeartDisease": {("M",): 0.4100957078534742, ("F",): 2.4384551724137933}},
    pytest.raises(ValueError),
]


expected_result_2 = [
    {
        "HeartDisease": {
            "dpr": 0.4100957078534742,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {"HeartDisease": {("M",): 0.4100957078534742, ("F",): 2.4384551724137933}},
    {"HeartDisease": False},
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
            ("M", "ASY"): 0.341659585454887,
            ("F", "ASY"): 0.5777645230023888,
            ("M", "TA"): 0.6178592623058512,
            ("M", "NAP"): 0.7696183431338224,
            ("M", "ATA"): 2.125556915316397,
            ("F", "NAP"): 3.403630912694409,
            ("F", "TA"): 3.8720349955031037,
            ("F", "ATA"): 5.879481064683228,
        }
    },
    {"HeartDisease": True},
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
            ("F", "ATA"): 5.879481064683228,
            ("F", "TA"): 3.8720349955031037,
            ("F", "NAP"): 3.403630912694409,
            ("M", "ATA"): 2.125556915316397,
            ("M", "NAP"): 0.7696183431338224,
            ("M", "TA"): 0.6178592623058512,
            ("F", "ASY"): 0.5777645230023888,
            ("M", "ASY"): 0.341659585454887,
        },
        "ExerciseAngina": {
            ("F", "TA"): np.inf,
            ("F", "ATA"): 3.0257391900480957,
            ("F", "NAP"): 2.6560696178758176,
            ("M", "ATA"): 2.34361081282544,
            ("M", "TA"): 1.4414410235954764,
            ("M", "NAP"): 0.7181744693453675,
            ("F", "ASY"): 0.4172482695250963,
            ("M", "ASY"): 0.28322304584791763,
        },
    },
    {"HeartDisease": True, "ExerciseAngina": True},
]


@pytest.mark.parametrize(
    "data, label, sensitive, real_target, predicted_target, threshold, expected_result",
    [
        (dataset1, "dpr", None, None, None, 1.2, expected_result_1),
        (
            df,
            "dpr",
            ["Sex"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            0.4,
            expected_result_2,
        ),
        (
            dataset3,
            "demographic_parity_ratio",
            None,
            None,
            None,
            0.8,
            expected_result_3,
        ),
        (
            dataset6,
            "demographic_parity_ratio",
            None,
            None,
            None,
            0.3,
            expected_result_6,
        ),
    ],
)
def test_demographic_parity_ratio(
    data: Dataset,
    label: str,
    sensitive: Sequence[str] | None,
    real_target: Sequence[str] | None,
    predicted_target: Sequence[str] | None,
    threshold: float,
    expected_result: Sequence[dict[str, dict]],
):
    if isinstance(data, Dataset):
        dpr = DemographicParityRatio(data, label)
    else:
        dpr = DemographicParityRatio(
            data, label, sensitive, real_target, predicted_target
        )
    result = dpr.summary()
    assert result == expected_result[0]
    ranking = dpr.rank()
    assert ranking == expected_result[1]
    if isinstance(expected_result[2], dict):
        is_biased = dpr.is_biased(threshold)
        assert is_biased == expected_result[2]
    else:
        with expected_result[2]:
            dpr.is_biased(threshold)


expected_result_2 = [
    {
        "HeartDiseasePred": {
            "did": 0.3619581918885117,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {"HeartDisease": {("F",): -0.3619581918885117, ("M",): 0.3619581918885117}},
    {"HeartDisease": True},
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
        "HeartDisease": {
            ("M", "ASY"): 0.5350647912366394,
            ("F", "ASY"): 0.26816817343458915,
            ("M", "TA"): 0.2019550215071515,
            ("M", "NAP"): 0.10163756118969114,
            ("M", "ATA"): -0.19894398116176273,
            ("F", "TA"): -0.28693386738173743,
            ("F", "ATA"): -0.30598148642935646,
            ("F", "NAP"): -0.31496621239521455,
        }
    },
    pytest.raises(ValueError),
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
        "HeartDisease": {
            ("F", "NAP"): -0.31496621239521455,
            ("F", "ATA"): -0.30598148642935646,
            ("F", "TA"): -0.28693386738173743,
            ("M", "ATA"): -0.19894398116176273,
            ("M", "NAP"): 0.10163756118969114,
            ("M", "TA"): 0.2019550215071515,
            ("F", "ASY"): 0.26816817343458915,
            ("M", "ASY"): 0.5350647912366394,
        },
        "ExerciseAngina": {
            ("F", "TA"): -0.26404969440876985,
            ("F", "ATA"): -0.16881159917067465,
            ("F", "NAP"): -0.1562329828184734,
            ("M", "ATA"): -0.14268433410535647,
            ("M", "TA"): -0.07357350393257941,
            ("M", "NAP"): 0.08642649606742059,
            ("F", "ASY"): 0.27472581579531175,
            ("M", "ASY"): 0.44419980257312147,
        },
    },
    {"HeartDisease": True, "ExerciseAngina": False},
]


@pytest.mark.parametrize(
    "data, label, sensitive, real_target, predicted_target, positive_target, threshold, expected_result",
    [
        (dataset2, "did", None, None, None, None, 0.3, expected_result_2),
        (
            dataset3,
            "disparate_impact_difference",
            None,
            None,
            None,
            None,
            -0.2,
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
    sensitive: Sequence[str] | None,
    real_target: Sequence[str] | None,
    predicted_target: Sequence[str] | None,
    positive_target: Sequence[int | float | str | bool] | None,
    threshold: float,
    expected_result: Sequence[dict[str, dict]],
):
    if isinstance(data, Dataset):
        did = DisparateImpactDifference(data, label)
    else:
        did = DisparateImpactDifference(
            data, label, sensitive, real_target, predicted_target, positive_target
        )

    result = did.summary()
    assert result == expected_result[0]
    ranking = did.rank()
    assert ranking == expected_result[1]
    if isinstance(expected_result[2], dict):
        is_biased = did.is_biased(threshold)
        assert is_biased == expected_result[2]
    else:
        with expected_result[2]:
            did.is_biased(threshold)


expected_result_2 = [
    {
        "HeartDiseasePred": {
            "dir": 0.4219830636141608,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {"HeartDisease": {("M",): 0.4219830636141608, ("F",): 2.369763353617309}},
    {"HeartDisease": False},
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
        "HeartDisease": {
            ("M", "ASY"): 0.3468836645650188,
            ("F", "ASY"): 0.5421518990141162,
            ("M", "TA"): 0.6173483803022393,
            ("M", "NAP"): 0.7690055427507021,
            ("M", "ATA"): 2.1240334935639593,
            ("F", "TA"): 3.869338673817374,
            ("F", "ATA"): 4.671777837152278,
            ("F", "NAP"): 5.173302314236593,
        }
    },
    {"HeartDisease": True},
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
        "HeartDisease": {
            ("F", "NAP"): 5.173302314236593,
            ("F", "ATA"): 4.671777837152278,
            ("F", "TA"): 3.869338673817374,
            ("M", "ATA"): 2.1240334935639593,
            ("M", "NAP"): 0.7690055427507021,
            ("M", "TA"): 0.6173483803022393,
            ("F", "ASY"): 0.5421518990141162,
            ("M", "ASY"): 0.3468836645650188,
        },
        "ExerciseAngina": {
            ("F", "TA"): np.inf,
            ("F", "ATA"): 3.0257391900480957,
            ("F", "NAP"): 2.6560696178758176,
            ("M", "ATA"): 2.34361081282544,
            ("M", "TA"): 1.4414410235954764,
            ("M", "NAP"): 0.7181744693453675,
            ("F", "ASY"): 0.4172482695250963,
            ("M", "ASY"): 0.28322304584791763,
        },
    },
    {"HeartDisease": True, "ExerciseAngina": True},
]


@pytest.mark.parametrize(
    "data, label, sensitive, real_target, predicted_target, threshold, expected_result",
    [
        (
            dataset1,
            "dir",
            None,
            None,
            None,
            1.2,
            pytest.raises(ValueError),
        ),
        (
            df,
            "dir",
            ["Sex"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            0.4,
            expected_result_2,
        ),
        (
            dataset3,
            "disparate_impact_ratio",
            None,
            None,
            None,
            0.8,
            expected_result_3,
        ),
        (
            dataset6,
            "disparate_impact_ratio",
            None,
            None,
            None,
            0.3,
            expected_result_6,
        ),
    ],
)
def test_disparate_impact_ratio(
    data: Dataset,
    label: str,
    sensitive: Sequence[str] | None,
    real_target: Sequence[str] | None,
    predicted_target: Sequence[str] | None,
    threshold: float,
    expected_result: Sequence[dict[str, dict]] | RaisesContext,
):
    if isinstance(expected_result, Sequence):
        if isinstance(data, Dataset):
            dira = DisparateImpactRatio(data, label)
        else:
            dira = DisparateImpactRatio(
                data, label, sensitive, real_target, predicted_target
            )
        result = dira.summary()
        assert result == expected_result[0]
        ranking = dira.rank()
        assert ranking == expected_result[1]
        is_biased = dira.is_biased(threshold)
        assert is_biased == expected_result[2]
    else:
        with expected_result:
            if isinstance(data, Dataset):
                dira = DisparateImpactRatio(data, label)
            else:
                dira = DisparateImpactRatio(
                    data, label, sensitive, real_target, predicted_target
                )
            dira.summary()


expected_result_2 = [
    {
        "HeartDisease": {
            "eod": 0.03816593886462882,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {"HeartDisease": {("M",): 0.03816593886462882, ("F",): -0.03816593886462882}},
    {"HeartDisease": False},
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
            ("F", "ATA"): 0.08516927384528401,
            ("F", "TA"): 0.08516927384528401,
            ("M", "ASY"): 0.0689814956178457,
            ("M", "NAP"): 0.0678532565292667,
            ("F", "ASY"): 0.05586524454125468,
            ("M", "ATA"): 0.028026416702426813,
            ("M", "TA"): -0.09528185397426492,
            ("F", "NAP"): -0.29578310710709704,
        }
    },
    {"HeartDisease": True},
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
            ("F", "ATA"): 0.08516927384528401,
            ("F", "TA"): 0.08516927384528401,
            ("M", "ASY"): 0.0689814956178457,
            ("M", "NAP"): 0.0678532565292667,
            ("F", "ASY"): 0.05586524454125468,
            ("M", "ATA"): 0.028026416702426813,
            ("M", "TA"): -0.09528185397426492,
            ("F", "NAP"): -0.29578310710709704,
        },
        "ExerciseAngina": {
            ("M", "ASY"): np.nan,
            ("M", "NAP"): np.nan,
            ("M", "ATA"): np.nan,
            ("F", "ASY"): np.nan,
            ("F", "ATA"): np.nan,
            ("F", "NAP"): np.nan,
            ("M", "TA"): np.nan,
            ("F", "TA"): np.nan,
        },
    },
    {"HeartDisease": True, "ExerciseAngina": False},
]


@pytest.mark.parametrize(
    "data, label, sensitive, real_target, predicted_target, threshold, expected_result",
    [
        (dataset2, "eod", None, None, None, 0.1, expected_result_2),
        (
            df,
            "equal_opportunity_difference",
            ["Sex", "ChestPainType"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            0.2,
            expected_result_3,
        ),
        (
            dataset6,
            "equal_opportunity_difference",
            None,
            None,
            None,
            0.2,
            expected_result_6,
        ),
    ],
)
def test_equal_opportunity_difference(
    data: Dataset | pd.DataFrame,
    label: str,
    sensitive: Sequence[str] | None,
    real_target: Sequence[str] | None,
    predicted_target: Sequence[str] | None,
    threshold: float,
    expected_result: Sequence[dict[str, dict]],
):
    if isinstance(data, Dataset):
        eod = EqualOpportunityDifference(data, label)
    else:
        eod = EqualOpportunityDifference(
            data, label, sensitive, real_target, predicted_target
        )
    result = eod.summary()
    assert result == expected_result[0]
    rankings = eod.rank()
    for (target, ranking), (expected_target, expected_ranking) in zip(
        rankings.items(), expected_result[1].items()
    ):
        assert ranking.keys() == expected_ranking.keys()
        for val, expected_val in zip(ranking.values(), expected_ranking.values()):
            if not np.isnan(val) and not np.isnan(expected_val):
                assert np.isclose(val, expected_val, atol=0.000002)
    is_biased = eod.is_biased(threshold)
    assert is_biased == expected_result[2]


expected_result_2 = [
    {
        "HeartDisease": {
            "eor": 0.9609821428571428,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {"HeartDisease": {("M",): 0.9609821428571428, ("F",): 1.0406020626219457}},
    {"HeartDisease": False},
]


expected_result_3 = [
    {
        "HeartDisease": {
            "equal_opportunity_ratio": 0.6666666666666666,
            "privileged": ("F", "ATA"),
            "unprivileged": ("F", "NAP"),
        }
    },
    {
        "HeartDisease": {
            ("F", "ATA"): 0.9148307261547161,
            ("F", "TA"): 0.9148307261547161,
            ("M", "ASY"): 0.9300273909393691,
            ("M", "NAP"): 0.9311028472164368,
            ("F", "ASY"): 0.9426646174445017,
            ("M", "ATA"): 0.9704985087342874,
            ("M", "TA"): 1.1131472015944397,
            ("F", "NAP"): 1.4436746606606454,
        }
    },
    {"HeartDisease": True},
]


expected_result_6 = [
    {
        "HeartDisease": {
            "equal_opportunity_ratio": 0.6666666666666666,
            "privileged": ("F", "ATA"),
            "unprivileged": ("F", "NAP"),
        },
        "ExerciseAngina": {
            "equal_opportunity_ratio": 1.0,
            "privileged": None,
            "unprivileged": None,
        },
    },
    {
        "HeartDisease": {
            ("F", "ATA"): 0.9148307261547161,
            ("F", "TA"): 0.9148307261547161,
            ("M", "ASY"): 0.9300273909393691,
            ("M", "NAP"): 0.9311028472164368,
            ("F", "ASY"): 0.9426646174445017,
            ("M", "ATA"): 0.9704985087342874,
            ("M", "TA"): 1.1131472015944397,
            ("F", "NAP"): 1.4436746606606454,
        },
        "ExerciseAngina": {
            ("M", "ASY"): np.nan,
            ("M", "NAP"): np.nan,
            ("M", "ATA"): np.nan,
            ("F", "ASY"): np.nan,
            ("F", "ATA"): np.nan,
            ("F", "NAP"): np.nan,
            ("M", "TA"): np.nan,
            ("F", "TA"): np.nan,
        },
    },
    {"HeartDisease": True, "ExerciseAngina": False},
]


@pytest.mark.parametrize(
    "data, label, sensitive, real_target, predicted_target, threshold, expected_result",
    [
        (
            dataset1,
            "eor",
            None,
            None,
            None,
            1.2,
            pytest.raises(ValueError),
        ),
        (
            df,
            "eor",
            ["Sex"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            0.4,
            expected_result_2,
        ),
        (
            dataset3,
            "equal_opportunity_ratio",
            None,
            None,
            None,
            0.8,
            expected_result_3,
        ),
        (
            dataset6,
            "equal_opportunity_ratio",
            None,
            None,
            None,
            0.9,
            expected_result_6,
        ),
    ],
)
def test_equal_opportuinity_ratio(
    data: Dataset,
    label: str,
    sensitive: Sequence[str] | None,
    real_target: Sequence[str] | None,
    predicted_target: Sequence[str] | None,
    threshold: float,
    expected_result: Sequence[dict[str, dict]] | RaisesContext,
):
    if isinstance(expected_result, Sequence):
        if isinstance(data, Dataset):
            eor = EqualOpportunityRatio(data, label)
        else:
            eor = EqualOpportunityRatio(
                data, label, sensitive, real_target, predicted_target
            )
        result = eor.summary()
        assert result == expected_result[0]
        rankings = eor.rank()
        for (target, ranking), (expected_target, expected_ranking) in zip(
            rankings.items(), expected_result[1].items()
        ):
            assert ranking.keys() == expected_ranking.keys()
            for val, expected_val in zip(ranking.values(), expected_ranking.values()):
                if not np.isnan(val) and not np.isnan(expected_val):
                    assert np.isclose(val, expected_val, atol=0.000002)
        is_biased = eor.is_biased(threshold)
        assert is_biased == expected_result[2]
    else:
        with expected_result:
            if isinstance(data, Dataset):
                eor = EqualOpportunityRatio(data, label)
            else:
                eor = EqualOpportunityRatio(
                    data, label, sensitive, real_target, predicted_target
                )
            eor.summary()


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
        (dataset1, None, None, None, None, None, pytest.raises(ValueError)),
        (
            dataset2,
            {"sensitive": f1_score},
            None,
            None,
            None,
            None,
            pytest.raises(KeyError),
        ),
        (
            df,
            None,
            ["Sex"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            None,
            expected_result_2,
        ),
        (
            dataset3,
            {
                "acc": accuracy_score,
                "balanced_acc": balanced_accuracy_score,
            },
            None,
            None,
            None,
            None,
            expected_result_3,
        ),
        (
            dataset6,
            [precision_score, recall_score, f1_score],
            None,
            None,
            None,
            None,
            expected_result_6,
        ),
    ],
)
def test_performancemetrics(
    data: Dataset,
    metrics: dict[str, Callable] | None,
    sensitive: Sequence[str] | None,
    real_target: Sequence[str] | None,
    predicted_target: Sequence[str] | None,
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
            pm = PerformanceMetric(
                data, metrics, sensitive, real_target, predicted_target, positive_target
            )
            pm()


expected_result_2 = [
    {
        "HeartDisease": {
            "equalised_odds_difference": 0.03816593886462882,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {"HeartDisease": {("M",): 0.03816593886462882, ("F",): -0.03816593886462882}},
    {"HeartDisease": False},
]


expected_result_3 = [
    {
        "HeartDisease": {
            "equalised_odds_difference": 0.33333333333333337,
            "privileged": ("F", "ATA"),
            "unprivileged": ("F", "NAP"),
        }
    },
    {
        "HeartDisease": {
            ("F", "TA"): 0.10053586843924016,
            ("F", "ATA"): 0.09033178680658709,
            ("M", "NAP"): 0.08199377127623639,
            ("M", "ASY"): 0.08153282325925479,
            ("M", "ATA"): 0.05205550855335028,
            ("F", "ASY"): 0.0014697534603408588,
            ("M", "TA"): -0.16240914536313006,
            ("F", "NAP"): -0.24551036643187954,
        }
    },
    {"HeartDisease": True},
]


expected_result_6 = [
    {
        "HeartDisease": {
            "equalised_odds_difference": 0.33333333333333337,
            "privileged": ("F", "ATA"),
            "unprivileged": ("F", "NAP"),
        },
        "ExerciseAngina": {
            "equalised_odds_difference": 0.0,
            "privileged": None,
            "unprivileged": None,
        },
    },
    {
        "HeartDisease": {
            ("F", "TA"): 0.10053586843924016,
            ("F", "ATA"): 0.09033178680658709,
            ("M", "NAP"): 0.08199377127623639,
            ("M", "ASY"): 0.08153282325925479,
            ("M", "ATA"): 0.05205550855335028,
            ("F", "ASY"): 0.0014697534603408588,
            ("M", "TA"): -0.16240914536313006,
            ("F", "NAP"): -0.24551036643187954,
        },
        "ExerciseAngina": {
            ("M", "ASY"): 0.0,
            ("M", "NAP"): 0.0,
            ("M", "ATA"): 0.0,
            ("F", "ASY"): 0.0,
            ("F", "ATA"): 0.0,
            ("F", "NAP"): 0.0,
            ("M", "TA"): 0.0,
            ("F", "TA"): 0.0,
        },
    },
    {"HeartDisease": True, "ExerciseAngina": False},
]


@pytest.mark.parametrize(
    "data, sensitive, real_target, predicted_target, threshold, expected_result",
    [
        (dataset2, None, None, None, 0.1, expected_result_2),
        (
            df,
            ["Sex", "ChestPainType"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            0.2,
            expected_result_3,
        ),
        (
            dataset6,
            None,
            None,
            None,
            0.2,
            expected_result_6,
        ),
    ],
)
def test_equalised_odds_difference(
    data: Dataset | pd.DataFrame,
    sensitive: Sequence[str] | None,
    real_target: Sequence[str] | None,
    predicted_target: Sequence[str] | None,
    threshold: float,
    expected_result: Sequence[dict[str, dict]],
):
    if isinstance(data, Dataset):
        eod = EqualisedOddsDifference(data)
    else:
        eod = EqualisedOddsDifference(data, sensitive, real_target, predicted_target)
    result = eod.summary()
    assert result == expected_result[0]
    rankings = eod.rank()
    for (target, ranking), (expected_target, expected_ranking) in zip(
        rankings.items(), expected_result[1].items()
    ):
        assert ranking.keys() == expected_ranking.keys()
        for val, expected_val in zip(ranking.values(), expected_ranking.values()):
            if not np.isnan(val) and not np.isnan(expected_val):
                assert np.isclose(val, expected_val, atol=0.000002)
    is_biased = eod.is_biased(threshold)
    assert is_biased == expected_result[2]


expected_result_2 = [
    {
        "HeartDisease": {
            "equalised_odds_ratio": 0.8033707865168539,
            "privileged": ("M",),
            "unprivileged": ("F",),
        }
    },
    {"HeartDisease": {("M",): 0.8033707865168539, ("F",): 1.2447552447552448}},
    {"HeartDisease": False},
]


expected_result_3 = [
    {
        "HeartDisease": {
            "equalised_odds_ratio": 0.0,
            "privileged": ("F", "NAP"),
            "unprivileged": ("M", "ASY"),
        }
    },
    {
        "HeartDisease": {
            ("M", "ASY"): np.inf,
            ("M", "NAP"): np.inf,
            ("M", "ATA"): np.inf,
            ("F", "ASY"): np.inf,
            ("F", "ATA"): np.inf,
            ("F", "NAP"): np.nan,
            ("M", "TA"): np.inf,
            ("F", "TA"): np.nan,
        }
    },
    {"HeartDisease": False},
]


@pytest.mark.parametrize(
    "data, sensitive, real_target, predicted_target, threshold, expected_result",
    [
        (
            dataset1,
            None,
            None,
            None,
            1.2,
            pytest.raises(ValueError),
        ),
        (
            df,
            ["Sex"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            0.4,
            expected_result_2,
        ),
        (
            dataset3,
            None,
            None,
            None,
            0.8,
            expected_result_3,
        ),
    ],
)
def test_equalised_odds_ratio(
    data: Dataset,
    sensitive: Sequence[str] | None,
    real_target: Sequence[str] | None,
    predicted_target: Sequence[str] | None,
    threshold: float,
    expected_result: Sequence[dict[str, dict]] | RaisesContext,
):
    if isinstance(expected_result, Sequence):
        if isinstance(data, Dataset):
            eor = EqualisedOddsRatio(data)
        else:
            eor = EqualisedOddsRatio(data, sensitive, real_target, predicted_target)
        result = eor.summary()
        assert result == expected_result[0]
        rankings = eor.rank()
        for (target, ranking), (expected_target, expected_ranking) in zip(
            rankings.items(), expected_result[1].items()
        ):
            assert ranking.keys() == expected_ranking.keys()
            for val, expected_val in zip(ranking.values(), expected_ranking.values()):
                if not np.isnan(val) and not np.isnan(expected_val):
                    assert np.isclose(val, expected_val, atol=0.000002)
        is_biased = eor.is_biased(threshold)
        assert is_biased == expected_result[2]
    else:
        with expected_result:
            if isinstance(data, Dataset):
                eor = EqualisedOddsRatio(data)
            else:
                eor = EqualisedOddsRatio(data, sensitive, real_target, predicted_target)
            eor.summary()


expected_result_1 = [
    {
        "sensitive": ("Sex",),
        "result": {
            "HeartDisease": {("M",): 0.3726567804180811, ("F",): -0.3726567804180811}
        },
    }
]
expected_result_2 = [
    {
        "sensitive": ("Sex",),
        "result": {
            "HeartDisease": {("M",): 0.4219830636141608, ("F",): 2.369763353617309}
        },
    },
    {
        "sensitive": ("ChestPainType",),
        "result": {
            "HeartDisease": {
                ("ASY",): 0.3917632113245289,
                ("TA",): 0.9779803774692927,
                ("NAP",): 1.3200622150699777,
                ("ATA",): 3.612010526994567,
            }
        },
    },
    {
        "sensitive": ("Sex", "ChestPainType"),
        "result": {
            "HeartDisease": {
                ("M", "ASY"): 0.3468836645650188,
                ("F", "ASY"): 0.5421518990141162,
                ("M", "TA"): 0.6173483803022393,
                ("M", "NAP"): 0.7690055427507021,
                ("M", "ATA"): 2.1240334935639593,
                ("F", "TA"): 3.869338673817374,
                ("F", "ATA"): 4.671777837152278,
                ("F", "NAP"): 5.173302314236593,
            }
        },
    },
]


expected_result_3 = [
    {
        "sensitive": ("Sex",),
        "result": {
            "HeartDisease": {("M",): 0.03816593886462882, ("F",): -0.03816593886462882}
        },
    },
    {
        "sensitive": ("ChestPainType",),
        "result": {
            "HeartDisease": {
                ("NAP",): 0.048316838338099695,
                ("ASY",): 0.04340882369780954,
                ("ATA",): 0.025394519369986518,
                ("TA",): -0.11712018140589575,
            }
        },
    },
    {
        "sensitive": ("Sex", "ChestPainType"),
        "result": {
            "HeartDisease": {
                ("F", "TA"): 0.10053586843924016,
                ("F", "ATA"): 0.09033178680658709,
                ("M", "NAP"): 0.08199377127623639,
                ("M", "ASY"): 0.08153282325925479,
                ("M", "ATA"): 0.05205550855335028,
                ("F", "ASY"): 0.0014697534603408588,
                ("M", "TA"): -0.16240914536313006,
                ("F", "NAP"): -0.24551036643187954,
            }
        },
    },
]


@pytest.mark.parametrize(
    "metric, data, sensitive, real_target, predicted_target, expected_result",
    [
        (
            DemographicParityDifference,
            dataset1,
            [],
            None,
            None,
            expected_result_1,
        ),
        (
            DisparateImpactRatio,
            df,
            ["Sex", "ChestPainType"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            expected_result_2,
        ),
        (
            EqualisedOddsDifference,
            dataset3,
            [],
            None,
            None,
            expected_result_3,
        ),
    ],
)
def test_super_set(
    metric: (
        type[DemographicParityDifference]
        | type[DemographicParityRatio]
        | type[DisparateImpactDifference]
        | type[DisparateImpactRatio]
        | type[EqualOpportunityDifference]
        | type[EqualOpportunityRatio]
        | type[EqualisedOddsDifference]
        | type[EqualisedOddsRatio]
        | type[FalsePositiveRateDifference]
        | type[FalsePositiveRateRatio]
    ),
    data: pd.DataFrame | Dataset,
    sensitive: Sequence[str],
    real_target: Sequence[str] | None,
    predicted_target: Sequence[str] | None,
    expected_result: Sequence[dict[str, dict]] | RaisesContext,
):
    result = super_set(
        metric,
        data,
        sensitive,
        real_target,
        predicted_target,
    )
    assert result == expected_result
