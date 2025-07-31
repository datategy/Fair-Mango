from collections.abc import Callable, Sequence
from typing import Any, Literal
from contextlib import AbstractContextManager

import numpy as np
import pandas as pd
import pytest
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
    EqualisedOddsDifference,
    EqualisedOddsRatio,
    EqualOpportunityDifference,
    EqualOpportunityRatio,
    PerformanceMetric,
    SelectionRate,
    false_negative_rate,
    false_positive_rate,
    true_negative_rate,
)

df = pd.read_csv("tests/data/heart_data.csv")

dataset1 = Dataset(df, "Sex", "HeartDisease")

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
    "data, use_y_true, expected_groups, expected_result",
    [
        (dataset1, True, [["M"], ["F"]], [0.63172414, 0.25906736]),
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
            [["M"], ["F"]],
            [0.45241379, 0.22279793],
        ),
        (
            dataset5,
            False,
            [["M"], ["F"]],
            [0.45241379, 0.22279793],
        ),
    ],
)
def test_selectionrate(
    data: Dataset,
    use_y_true: bool,
    expected_groups: Sequence[str],
    expected_result: Sequence[float] | AbstractContextManager,
):
    if not isinstance(expected_result, AbstractContextManager):
        sr = SelectionRate(data, use_y_true)
        result = sr()
        for i, res in enumerate(result):
            assert res.sensitive_group == expected_groups[i]
            assert np.isclose(res.data, expected_result[i])
    else:
        with expected_result:
            sr = SelectionRate(data, use_y_true)
            sr()


confusionmatrix_expected_result_2 = [
    {
        "sensitive_group": np.array(["M"]),
        "false_negative_rate": [0.021834061135371178],
        "false_positive_rate": [0.02247191011235955],
        "true_negative_rate": [0.9775280898876404],
        "true_positive_rate": [0.9781659388646288],
    },
    {
        "sensitive_group": np.array(["F"]),
        "false_negative_rate": [0.06],
        "false_positive_rate": [0.027972027972027972],
        "true_negative_rate": [0.972027972027972],
        "true_positive_rate": [0.94],
    },
]


confusionmatrix_expected_result_3 = [
    {"sensitive_group": np.array(["M", "ASY"]), "fpr": [0.0136986301369863]},
    {"sensitive_group": np.array(["M", "NAP"]), "fpr": [0.011904761904761904]},
    {"sensitive_group": np.array(["M", "ATA"]), "fpr": [0.010752688172043012]},
    {"sensitive_group": np.array(["F", "ASY"]), "fpr": [0.0967741935483871]},
    {"sensitive_group": np.array(["F", "ATA"]), "fpr": [0.017857142857142856]},
    {"sensitive_group": np.array(["F", "NAP"]), "fpr": [0.0]},
    {"sensitive_group": np.array(["M", "TA"]), "fpr": [0.17647058823529413]},
    {"sensitive_group": np.array(["F", "TA"]), "fpr": [0.0]},
]


confusionmatrix_expected_result_6 = [
    {
        "sensitive_group": np.array(["M", "ASY"]),
        "true_negative_rate": [0.9863013698630136],
        "false_negative_rate": [0.014164305949008499],
    },
    {
        "sensitive_group": np.array(["M", "NAP"]),
        "true_negative_rate": [0.9880952380952381],
        "false_negative_rate": [0.015151515151515152],
    },
    {
        "sensitive_group": np.array(["M", "ATA"]),
        "true_negative_rate": [0.989247311827957],
        "false_negative_rate": [0.05],
    },
    {
        "sensitive_group": np.array(["F", "ASY"]),
        "true_negative_rate": [0.9032258064516129],
        "false_negative_rate": [0.02564102564102564],
    },
    {
        "sensitive_group": np.array(["F", "ATA"]),
        "true_negative_rate": [0.9821428571428571],
        "false_negative_rate": [
            0.0,
        ],
    },
    {
        "sensitive_group": np.array(["F", "NAP"]),
        "true_negative_rate": [1.0],
        "false_negative_rate": [0.3333333333333333],
    },
    {
        "sensitive_group": np.array(["M", "TA"]),
        "true_negative_rate": [0.8235294117647058],
        "false_negative_rate": [0.15789473684210525],
    },
    {
        "sensitive_group": np.array(["F", "TA"]),
        "true_negative_rate": [1.0],
        "false_negative_rate": [0.0],
    },
]
ConfusionMatrixMetric = Callable[[int, int], float]
confusion_matrix_metrics: list[ConfusionMatrixMetric] = [
    true_negative_rate,
    false_negative_rate,
]


@pytest.mark.parametrize(
    "data, metrics, expected_result",
    [
        (dataset1, None, pytest.raises(ValueError)),
        (dataset2, {"sensitive_group": false_positive_rate}, pytest.raises(KeyError)),
        (dataset2, None, confusionmatrix_expected_result_2),
        (dataset3, {"fpr": false_positive_rate}, confusionmatrix_expected_result_3),
        (
            dataset6,
            confusion_matrix_metrics,
            confusionmatrix_expected_result_6,
        ),
    ],
)
def test_confusionmatrix(
    data: Dataset,
    metrics: dict[str, Callable[..., Any]] | Sequence[ConfusionMatrixMetric] | None,
    expected_result: Sequence[dict[str, Sequence]] | AbstractContextManager,
):
    if not isinstance(expected_result, AbstractContextManager):
        cf = ConfusionMatrix(data, metrics)
        result = cf()
        for i, res in enumerate(result):
            assert np.array_equal(
                res.sensitive_group, expected_result[i]["sensitive_group"]
            )
            for key in expected_result[i].keys():
                if key != "sensitive_group":
                    res_values = getattr(res, key)
                    for val, expected_val in zip(
                        res_values, expected_result[i][key], strict=True
                    ):
                        if np.isnan(val) and np.isnan(expected_val):
                            continue
                        else:
                            assert np.all(np.isclose(val, expected_val))

    else:
        with expected_result:
            cf = ConfusionMatrix(data, metrics)
            cf()


dpd_expected_result_2 = [
    {
        "demographic_parity_difference": 0.3726567804180811,
        "privileged_sensitive_group": ["M"],
        "unprivileged_sensitive_group": ["F"],
    },
    [
        {"sensitive_group": ["M"], "score": 0.3726567804180811},
        {"sensitive_group": ["F"], "score": -0.3726567804180811},
    ],
    True,
]


dpd_expected_result_3 = [
    {
        "demographic_parity_difference": 0.7619718309859155,
        "privileged_sensitive_group": ["M", "ASY"],
        "unprivileged_sensitive_group": ["F", "ATA"],
    },
    [
        {"sensitive_group": ["M", "ASY"], "score": 0.3886384976525821},
        {"sensitive_group": ["F", "ASY"], "score": -0.27149564050972497},
        {"sensitive_group": ["M", "TA"], "score": -0.30086071987480434},
        {"sensitive_group": ["M", "NAP"], "score": -0.3886384976525821},
        {"sensitive_group": ["M", "ATA"], "score": -0.6516473472101043},
        {"sensitive_group": ["F", "NAP"], "score": -0.7154309504827708},
        {"sensitive_group": ["F", "TA"], "score": -0.7286384976525822},
        {"sensitive_group": ["F", "ATA"], "score": -0.7619718309859155},
    ],
    True,
]


dpd_expected_result_6 = [
    {
        "demographic_parity_difference": 0.7619718309859155,
        "privileged_sensitive_group": ["M", "ASY"],
        "unprivileged_sensitive_group": ["F", "ATA"],
    },
    [
        {"sensitive_group": ["M", "ASY"], "score": 0.3886384976525821},
        {"sensitive_group": ["F", "ASY"], "score": -0.27149564050972497},
        {"sensitive_group": ["M", "TA"], "score": -0.30086071987480434},
        {"sensitive_group": ["M", "NAP"], "score": -0.3886384976525821},
        {"sensitive_group": ["M", "ATA"], "score": -0.6516473472101043},
        {"sensitive_group": ["F", "NAP"], "score": -0.7154309504827708},
        {"sensitive_group": ["F", "TA"], "score": -0.7286384976525822},
        {"sensitive_group": ["F", "ATA"], "score": -0.7619718309859155},
    ],
    True,
]


@pytest.mark.parametrize(
    "data, label, threshold, expected_result",
    [
        (dataset2, "demographic_parity_difference", 0.3, dpd_expected_result_2),
        (
            Dataset(
                df,
                ["Sex", "ChestPainType"],
                "HeartDisease",
                "HeartDiseasePred",
            ),
            "demographic_parity_difference",
            0.1,
            dpd_expected_result_3,
        ),
        (
            dataset6,
            "demographic_parity_difference",
            0.45,
            dpd_expected_result_6,
        ),
    ],
)
def test_demographic_parity_difference(
    data: Dataset,
    label: Literal["demographic_parity_difference"],
    threshold: float,
    expected_result: tuple[dict, list[dict], bool],
):
    dpd = DemographicParityDifference(data, label)

    is_biased = dpd.is_biased(threshold)
    assert is_biased == expected_result[2]


dpr_expected_result_1 = [
    {
        "demographic_parity_ratio": 0.4100957078534742,
        "privileged_sensitive_group": ["M"],
        "unprivileged_sensitive_group": ["F"],
    },
    [
        {"sensitive_group": ["F"], "score": 0.4100957078534742},
        {"sensitive_group": ["M"], "score": 2.4384551724137933},
    ],
    pytest.raises(ValueError),
]


dpr_expected_result_2 = [
    {
        "demographic_parity_ratio": 0.4100957078534742,
        "privileged_sensitive_group": ["M"],
        "unprivileged_sensitive_group": ["F"],
    },
    [
        {"sensitive_group": ["F"], "score": 0.4100957078534742},
        {"sensitive_group": ["M"], "score": 2.4384551724137933},
    ],
    False,
]


dpr_expected_result_3 = [
    {
        "demographic_parity_ratio": 0.08045325779036827,
        "privileged_sensitive_group": ["M", "ASY"],
        "unprivileged_sensitive_group": ["F", "ATA"],
    },
    [
        {"sensitive_group": ["F", "ATA"], "score": 0.08045325779036827},
        {"sensitive_group": ["F", "TA"], "score": 0.12067988668555243},
        {"sensitive_group": ["F", "NAP"], "score": 0.1366187396440216},
        {"sensitive_group": ["M", "ATA"], "score": 0.21359271979743788},
        {"sensitive_group": ["M", "NAP"], "score": 0.5309915014164307},
        {"sensitive_group": ["M", "TA"], "score": 0.6369216241737489},
        {"sensitive_group": ["F", "ASY"], "score": 0.6723593686766491},
        {"sensitive_group": ["M", "ASY"], "score": 1.8832693128467775},
    ],
    True,
]


dpr_expected_result_6 = [
    {
        "demographic_parity_ratio": 0.08045325779036827,
        "privileged_sensitive_group": ["M", "ASY"],
        "unprivileged_sensitive_group": ["F", "ATA"],
    },
    [
        {"sensitive_group": ["F", "ATA"], "score": 0.08045325779036827},
        {"sensitive_group": ["F", "TA"], "score": 0.12067988668555243},
        {"sensitive_group": ["F", "NAP"], "score": 0.1366187396440216},
        {"sensitive_group": ["M", "ATA"], "score": 0.21359271979743788},
        {"sensitive_group": ["M", "NAP"], "score": 0.5309915014164307},
        {"sensitive_group": ["M", "TA"], "score": 0.6369216241737489},
        {"sensitive_group": ["F", "ASY"], "score": 0.6723593686766491},
        {"sensitive_group": ["M", "ASY"], "score": 1.8832693128467775},
    ],
    True,
]


@pytest.mark.parametrize(
    "data, label, threshold, expected_result",
    [
        (dataset1, "demographic_parity_ratio", 1.2, dpr_expected_result_1),
        (
            Dataset(df, "Sex", "HeartDisease", "HeartDiseasePred"),
            "demographic_parity_ratio",
            0.4,
            dpr_expected_result_2,
        ),
        (
            dataset3,
            "demographic_parity_ratio",
            0.8,
            dpr_expected_result_3,
        ),
        (
            dataset6,
            "demographic_parity_ratio",
            0.3,
            dpr_expected_result_6,
        ),
    ],
)
def test_demographic_parity_ratio(
    data: Dataset,
    label: Literal["demographic_parity_ratio"],
    threshold: float,
    expected_result: Sequence[dict[str, dict]],
):
    dpr = DemographicParityRatio(data, label)
    if isinstance(expected_result[2], bool):
        is_biased = dpr.is_biased(threshold)
        assert is_biased == expected_result[2]
    else:
        with pytest.raises(ValueError):
            dpr.is_biased(threshold)


did_expected_result_2 = [
    {
        "disparate_impact_difference": 0.3619581918885117,
        "privileged_sensitive_group": ["M"],
        "unprivileged_sensitive_group": ["F"],
    },
    [
        {"sensitive_group": ["M"], "score": 0.3619581918885117},
        {"sensitive_group": ["F"], "score": -0.3619581918885117},
    ],
    True,
]


did_expected_result_3 = [
    {
        "disparate_impact_difference": 0.7437771281778722,
        "privileged_sensitive_group": ["M", "ASY"],
        "unprivileged_sensitive_group": ["F", "NAP"],
    },
    {
        "HeartDisease": [
            {"sensitive_group": ["M", "ASY"], "score": 0.5350647912366394},
            {"sensitive_group": ["F", "ASY"], "score": 0.26816817343458915},
            {"sensitive_group": ["M", "TA"], "score": 0.2019550215071515},
            {"sensitive_group": ["M", "NAP"], "score": 0.10163756118969114},
            {"sensitive_group": ["M", "ATA"], "score": -0.19894398116176273},
            {"sensitive_group": ["F", "TA"], "score": -0.28693386738173743},
            {"sensitive_group": ["F", "ATA"], "score": -0.30598148642935646},
            {"sensitive_group": ["F", "NAP"], "score": -0.31496621239521455},
        ]
    },
    pytest.raises(ValueError),
]


did_expected_result_6 = [
    {
        "disparate_impact_difference": 0.7437771281778722,
        "privileged_sensitive_group": ["M", "ASY"],
        "unprivileged_sensitive_group": ["F", "NAP"],
    },
    [
        {"sensitive_group": ["M", "ASY"], "score": 0.37924882629107975},
        {"sensitive_group": ["F", "ASY"], "score": -0.23353454057679401},
        {"sensitive_group": ["M", "TA"], "score": -0.29147104851330197},
        {"sensitive_group": ["M", "NAP"], "score": -0.37924882629107975},
        {"sensitive_group": ["M", "ATA"], "score": -0.6422576758486018},
        {"sensitive_group": ["F", "TA"], "score": -0.7192488262910798},
        {"sensitive_group": ["F", "ATA"], "score": -0.7359154929577464},
        {"sensitive_group": ["F", "NAP"], "score": -0.7437771281778722},
    ],
    True,
]


@pytest.mark.parametrize(
    "data, label, threshold, expected_result",
    [
        (dataset2, "disparate_impact_difference", 0.3, did_expected_result_2),
        (
            dataset3,
            "disparate_impact_difference",
            -0.2,
            pytest.raises(ValueError),
        ),
        (
            dataset6,
            "disparate_impact_difference",
            0.45,
            did_expected_result_6,
        ),
    ],
)
def test_disparate_impact_difference(
    data: Dataset,
    label: Literal["disparate_impact_difference"],
    threshold: float,
    expected_result: Sequence[dict[str, dict]] | AbstractContextManager,
):
    if not isinstance(expected_result, AbstractContextManager):
        did = DisparateImpactDifference(data, label)

        if isinstance(expected_result[2], bool):
            is_biased = did.is_biased(threshold)
            assert is_biased == expected_result[2]
        else:
            with pytest.raises(ValueError):
                did.is_biased(threshold)
    else:
        with expected_result:
            did = DisparateImpactDifference(data, label)
            did.is_biased(threshold)


dir_expected_result_2 = [
    {
        "disparate_impact_ratio": 0.4219830636141608,
        "privileged_sensitive_group": ["M"],
        "unprivileged_sensitive_group": ["F"],
    },
    [
        {"sensitive_group": ["F"], "score": 0.4219830636141608},
        {"sensitive_group": ["M"], "score": 2.369763353617309},
    ],
    False,
]


dir_expected_result_3 = [
    {
        "disparate_impact_ratio": 0.09212304698059144,
        "privileged_sensitive_group": ["M", "ASY"],
        "unprivileged_sensitive_group": ["F", "NAP"],
    },
    [
        {"sensitive_group": ["F", "NAP"], "score": 0.09212304698059144},
        {"sensitive_group": ["F", "ATA"], "score": 0.10171919770773638},
        {"sensitive_group": ["F", "TA"], "score": 0.12206303724928369},
        {"sensitive_group": ["M", "ATA"], "score": 0.21604077389253748},
        {"sensitive_group": ["M", "NAP"], "score": 0.5370773638968481},
        {"sensitive_group": ["M", "TA"], "score": 0.6442215854823304},
        {"sensitive_group": ["F", "ASY"], "score": 0.7149406467458044},
        {"sensitive_group": ["M", "ASY"], "score": 1.861929150661545},
    ],
    True,
]


dir_expected_result_6 = [
    {
        "disparate_impact_ratio": 0.09212304698059144,
        "privileged_sensitive_group": ["M", "ASY"],
        "unprivileged_sensitive_group": ["F", "NAP"],
    },
    [
        {"sensitive_group": ["F", "NAP"], "score": 0.09212304698059144},
        {"sensitive_group": ["F", "ATA"], "score": 0.10171919770773638},
        {"sensitive_group": ["F", "TA"], "score": 0.12206303724928369},
        {"sensitive_group": ["M", "ATA"], "score": 0.21604077389253748},
        {"sensitive_group": ["M", "NAP"], "score": 0.5370773638968481},
        {"sensitive_group": ["M", "TA"], "score": 0.6442215854823304},
        {"sensitive_group": ["F", "ASY"], "score": 0.7149406467458044},
        {"sensitive_group": ["M", "ASY"], "score": 1.861929150661545},
    ],
    True,
]


@pytest.mark.parametrize(
    "data, label, threshold, expected_result",
    [
        (
            dataset1,
            "disparate_impact_ratio",
            1.2,
            pytest.raises(ValueError),
        ),
        (
            Dataset(df, ["Sex"], "HeartDisease", "HeartDiseasePred"),
            "disparate_impact_ratio",
            0.4,
            dir_expected_result_2,
        ),
        (
            dataset3,
            "disparate_impact_ratio",
            0.8,
            dir_expected_result_3,
        ),
        (
            dataset6,
            "disparate_impact_ratio",
            0.3,
            dir_expected_result_6,
        ),
    ],
)
def test_disparate_impact_ratio(
    data: Dataset,
    label: Literal["disparate_impact_ratio"],
    threshold: float,
    expected_result: Sequence[dict[str, dict]] | AbstractContextManager,
):
    if not isinstance(expected_result, AbstractContextManager):
        dira = DisparateImpactRatio(data, label)

        is_biased = dira.is_biased(threshold)
        assert is_biased == expected_result[2]
    else:
        with expected_result:
            dira = DisparateImpactRatio(data, label)
            dira.summary()


eod_expected_result_2 = [
    {
        "equal_opportunity_difference": 0.03816593886462882,
        "privileged_sensitive_group": ["M"],
        "unprivileged_sensitive_group": ["F"],
    },
    [
        {"sensitive_group": ["M"], "score": 0.03816593886462882},
        {"sensitive_group": ["F"], "score": -0.03816593886462882},
    ],
    False,
]


eod_expected_result_3 = [
    {
        "equal_opportunity_difference": 0.33333333333333337,
        "privileged_sensitive_group": ["F", "ATA"],
        "unprivileged_sensitive_group": ["F", "NAP"],
    },
    [
        {"sensitive_group": ["F", "ATA"], "score": 0.014164305949008527},
        {"sensitive_group": ["F", "TA"], "score": 0.014164305949008527},
        {"sensitive_group": ["M", "ASY"], "score": 0.0009872092025066115},
        {"sensitive_group": ["M", "NAP"], "score": -0.0009872092025066115},
        {"sensitive_group": ["F", "ASY"], "score": -0.011476719692017134},
        {"sensitive_group": ["M", "ATA"], "score": -0.03583569405099152},
        {"sensitive_group": ["M", "TA"], "score": -0.14373043089309678},
        {"sensitive_group": ["F", "NAP"], "score": -0.31916902738432484},
    ],
    True,
]


eod_expected_result_6 = [
    {
        "equal_opportunity_difference": 0.33333333333333337,
        "privileged_sensitive_group": ["F", "ATA"],
        "unprivileged_sensitive_group": ["F", "NAP"],
    },
    [
        {"sensitive_group": ["F", "ATA"], "score": 0.014164305949008527},
        {"sensitive_group": ["F", "TA"], "score": 0.014164305949008527},
        {"sensitive_group": ["M", "ASY"], "score": 0.0009872092025066115},
        {"sensitive_group": ["M", "NAP"], "score": -0.0009872092025066115},
        {"sensitive_group": ["F", "ASY"], "score": -0.011476719692017134},
        {"sensitive_group": ["M", "ATA"], "score": -0.03583569405099152},
        {"sensitive_group": ["M", "TA"], "score": -0.14373043089309678},
        {"sensitive_group": ["F", "NAP"], "score": -0.31916902738432484},
    ],
    True,
]


@pytest.mark.parametrize(
    "data, label, threshold, expected_result",
    [
        (dataset2, "equal_opportunity_difference", 0.1, eod_expected_result_2),
        (
            Dataset(
                df,
                ["Sex", "ChestPainType"],
                "HeartDisease",
                "HeartDiseasePred",
            ),
            "equal_opportunity_difference",
            0.2,
            eod_expected_result_3,
        ),
        (
            dataset6,
            "equal_opportunity_difference",
            0.2,
            eod_expected_result_6,
        ),
    ],
)
def test_equal_opportunity_difference(
    data: Dataset,
    label: Literal["equal_opportunity_difference"],
    threshold: float,
    expected_result: tuple[dict, list[dict], bool],
):
    eod = EqualOpportunityDifference(data, label)
    is_biased = eod.is_biased(threshold)
    assert is_biased == expected_result[2]


eor_expected_result_2 = [
    {
        "equal_opportunity_ratio": 0.9609821428571428,
        "privileged_sensitive_group": ["M"],
        "unprivileged_sensitive_group": ["F"],
    },
    [
        {"sensitive_group": ["F"], "score": 0.9609821428571428},
        {"sensitive_group": ["M"], "score": 1.0406020626219457},
    ],
    False,
]


eor_expected_result_3 = [
    {
        "equal_opportunity_ratio": 0.6666666666666666,
        "privileged_sensitive_group": ["F", "ATA"],
        "unprivileged_sensitive_group": ["F", "NAP"],
    },
    [
        {"sensitive_group": ["F", "NAP"], "score": 0.6762452107279693},
        {"sensitive_group": ["M", "TA"], "score": 0.854204476709014},
        {"sensitive_group": ["M", "ATA"], "score": 0.9636494252873564},
        {"sensitive_group": ["F", "ASY"], "score": 0.9883583849101092},
        {"sensitive_group": ["M", "NAP"], "score": 0.9989986067572275},
        {"sensitive_group": ["M", "ASY"], "score": 1.0010023970363913},
        {"sensitive_group": ["F", "ATA"], "score": 1.014367816091954},
        {"sensitive_group": ["F", "TA"], "score": 1.014367816091954},
    ],
    True,
]


eor_expected_result_6 = [
    {
        "equal_opportunity_ratio": 0.6666666666666666,
        "privileged_sensitive_group": ["F", "ATA"],
        "unprivileged_sensitive_group": ["F", "NAP"],
    },
    [
        {"sensitive_group": ["F", "NAP"], "score": 0.6762452107279693},
        {"sensitive_group": ["M", "TA"], "score": 0.854204476709014},
        {"sensitive_group": ["M", "ATA"], "score": 0.9636494252873564},
        {"sensitive_group": ["F", "ASY"], "score": 0.9883583849101092},
        {"sensitive_group": ["M", "NAP"], "score": 0.9989986067572275},
        {"sensitive_group": ["M", "ASY"], "score": 1.0010023970363913},
        {"sensitive_group": ["F", "ATA"], "score": 1.014367816091954},
        {"sensitive_group": ["F", "TA"], "score": 1.014367816091954},
    ],
    True,
]


@pytest.mark.parametrize(
    "data, label, threshold, expected_result",
    [
        (
            dataset1,
            "equal_opportunity_ratio",
            1.2,
            pytest.raises(ValueError),
        ),
        (
            Dataset(df, ["Sex"], "HeartDisease", "HeartDiseasePred"),
            "equal_opportunity_ratio",
            0.4,
            eor_expected_result_2,
        ),
        (
            dataset3,
            "equal_opportunity_ratio",
            0.8,
            eor_expected_result_3,
        ),
        (
            dataset6,
            "equal_opportunity_ratio",
            0.9,
            eor_expected_result_6,
        ),
    ],
)
def test_equal_opportuinity_ratio(
    data: Dataset,
    label: Literal["equal_opportunity_ratio"],
    threshold: float,
    expected_result: tuple[dict, list[dict], bool] | AbstractContextManager,
):
    if not isinstance(expected_result, AbstractContextManager):
        eor = EqualOpportunityRatio(data, label)        
        is_biased = eor.is_biased(threshold)
        assert is_biased == expected_result[2]
    else:
        with expected_result:
            eor = EqualOpportunityRatio(data, label)
            eor.summary()


performancemetrics_expected_result_2 = [
    {
        "sensitive_group": np.array(["M"], dtype=object),
        "accuracy": [0.9779310344827586],
        "balanced accuracy": [0.9778470143761346],
        "precision": [0.986784140969163],
        "recall": [0.9781659388646288],
        "f1-score": [0.9824561403508771],
    },
    {
        "sensitive_group": np.array(["F"], dtype=object),
        "accuracy": [0.9637305699481865],
        "balanced accuracy": [0.956013986013986],
        "precision": [0.9215686274509803],
        "recall": [0.94],
        "f1-score": [0.9306930693069307],
    },
]


performancemetrics_expected_result_3 = [
    {
        "sensitive_group": np.array(["M", "ASY"], dtype=object),
        "acc": [0.9859154929577465],
        "balanced_acc": [0.9860685319570026],
    },
    {
        "sensitive_group": np.array(["M", "NAP"], dtype=object),
        "acc": [0.9866666666666667],
        "balanced_acc": [0.9864718614718615],
    },
    {
        "sensitive_group": np.array(["M", "ATA"], dtype=object),
        "acc": [0.9823008849557522],
        "balanced_acc": [0.9696236559139785],
    },
    {
        "sensitive_group": np.array(["F", "ASY"], dtype=object),
        "acc": [0.9428571428571428],
        "balanced_acc": [0.9387923904052936],
    },
    {
        "sensitive_group": np.array(["F", "ATA"], dtype=object),
        "acc": [0.9833333333333333],
        "balanced_acc": [0.9910714285714286],
    },
    {
        "sensitive_group": np.array(["F", "NAP"], dtype=object),
        "acc": [0.9622641509433962],
        "balanced_acc": [0.8333333333333333],
    },
    {
        "sensitive_group": np.array(["M", "TA"], dtype=object),
        "acc": [0.8333333333333334],
        "balanced_acc": [0.8328173374613003],
    },
    {
        "sensitive_group": np.array(["F", "TA"], dtype=object),
        "acc": [1.0],
        "balanced_acc": [1.0],
    },
]


performancemetrics_expected_result_6 = [
    {
        "sensitive_group": np.array(["M", "ASY"], dtype=object),
        "precision_score": [0.997134670487106],
        "recall_score": [
            0.9858356940509915,
        ],
        "f1_score": [
            0.9914529914529915,
        ],
    },
    {
        "sensitive_group": np.array(["M", "NAP"], dtype=object),
        "precision_score": [
            0.9848484848484849,
        ],
        "recall_score": [
            0.9848484848484849,
        ],
        "f1_score": [
            0.9848484848484849,
        ],
    },
    {
        "sensitive_group": np.array(["M", "ATA"], dtype=object),
        "precision_score": [
            0.95,
        ],
        "recall_score": [
            0.95,
        ],
        "f1_score": [
            0.95,
        ],
    },
    {
        "sensitive_group": np.array(["F", "ASY"], dtype=object),
        "precision_score": [
            0.926829268292683,
        ],
        "recall_score": [
            0.9743589743589743,
        ],
        "f1_score": [
            0.95,
        ],
    },
    {
        "sensitive_group": np.array(["F", "ATA"], dtype=object),
        "precision_score": [0.8],
        "recall_score": [1.0],
        "f1_score": [0.8888888888888888],
    },
    {
        "sensitive_group": np.array(["F", "NAP"], dtype=object),
        "precision_score": [1.0],
        "recall_score": [0.6666666666666666],
        "f1_score": [0.8],
    },
    {
        "sensitive_group": np.array(["M", "TA"], dtype=object),
        "precision_score": [0.8421052631578947],
        "recall_score": [0.8421052631578947],
        "f1_score": [0.8421052631578947],
    },
    {
        "sensitive_group": np.array(["F", "TA"], dtype=object),
        "precision_score": [1.0],
        "recall_score": [1.0],
        "f1_score": [1.0],
    },
]


@pytest.mark.parametrize(
    "data, metrics, expected_result",
    [
        (dataset1, None, pytest.raises(ValueError)),
        (
            dataset2,
            {"sensitive_group": f1_score},
            pytest.raises(KeyError),
        ),
        (
            Dataset(df, ["Sex"], "HeartDisease", "HeartDiseasePred"),
            None,
            performancemetrics_expected_result_2,
        ),
        (
            dataset3,
            {
                "acc": accuracy_score,
                "balanced_acc": balanced_accuracy_score,
            },
            performancemetrics_expected_result_3,
        ),
        (
            dataset6,
            [precision_score, recall_score, f1_score],
            performancemetrics_expected_result_6,
        ),
    ],
)
def test_performancemetrics(
    data: Dataset,
    metrics: dict[str, Callable[..., Any]] | Sequence[Callable[..., Any]] | None,
    expected_result: Sequence[dict[str, Sequence]] | AbstractContextManager,
):
    if not isinstance(expected_result, AbstractContextManager):
        pm = PerformanceMetric(data, metrics)
        result = pm()
        for i, res in enumerate(result):
            assert np.array_equal(
                res.sensitive_group, expected_result[i]["sensitive_group"]
            )
            for key in expected_result[i].keys():
                if key != "sensitive_group":
                    res_value = getattr(res, key)
                    if isinstance(res_value[0], object):
                        try:
                            assert np.all(res_value == expected_result[i][key])
                        except AttributeError:
                            assert res_value == expected_result[i][key]
                    else:
                        assert np.all(np.isclose(res_value, expected_result[i][key]))

    else:
        with expected_result:
            pm = PerformanceMetric(
                data,
                metrics,
            )
            pm()


eod_expected_result_2 = [
    {
        "equalised_odds_difference": 0.03816593886462882,
        "privileged_sensitive_group": ["M"],
        "unprivileged_sensitive_group": ["F"],
    },
    [
        {"sensitive_group": ["M"], "score": 0.03816593886462882},
        {"sensitive_group": ["F"], "score": -0.03816593886462882},
    ],
    False,
]


eod_expected_result_3 = [
    {
        "equalised_odds_difference": 0.33333333333333337,
        "privileged_sensitive_group": ["F", "ATA"],
        "unprivileged_sensitive_group": ["F", "NAP"],
    },
    [
        {"sensitive_group": ["F", "TA"], "score": 0.10053586843924016},
        {"sensitive_group": ["F", "ATA"], "score": 0.09033178680658709},
        {"sensitive_group": ["M", "NAP"], "score": 0.08199377127623639},
        {"sensitive_group": ["M", "ASY"], "score": 0.08153282325925479},
        {"sensitive_group": ["M", "ATA"], "score": 0.05205550855335028},
        {"sensitive_group": ["F", "ASY"], "score": 0.0014697534603408588},
        {"sensitive_group": ["M", "TA"], "score": -0.16240914536313006},
        {"sensitive_group": ["F", "NAP"], "score": -0.24551036643187954},
    ],
    True,
]


eod_expected_result_6 = [
    {
        "equalised_odds_difference": 0.33333333333333337,
        "privileged_sensitive_group": ["F", "ATA"],
        "unprivileged_sensitive_group": ["F", "NAP"],
    },
    [
        {"sensitive_group": ["F", "TA"], "score": 0.10053586843924016},
        {"sensitive_group": ["F", "ATA"], "score": 0.09033178680658709},
        {"sensitive_group": ["M", "NAP"], "score": 0.08199377127623639},
        {"sensitive_group": ["M", "ASY"], "score": 0.08153282325925479},
        {"sensitive_group": ["M", "ATA"], "score": 0.05205550855335028},
        {"sensitive_group": ["F", "ASY"], "score": 0.0014697534603408588},
        {"sensitive_group": ["M", "TA"], "score": -0.16240914536313006},
        {"sensitive_group": ["F", "NAP"], "score": -0.24551036643187954},
    ],
    True,
]


@pytest.mark.parametrize(
    "data, threshold, expected_result",
    [
        (dataset2, 0.1, eod_expected_result_2),
        (
            Dataset(df, ["Sex", "ChestPainType"], "HeartDisease", "HeartDiseasePred"),
            0.2,
            eod_expected_result_3,
        ),
        (
            dataset6,
            0.2,
            eod_expected_result_6,
        ),
    ],
)
def test_equalised_odds_difference(
    data: Dataset,
    threshold: float,
    expected_result: tuple[dict, list[dict], bool],
):
    eod = EqualisedOddsDifference(data)
    is_biased = eod.is_biased(threshold)
    assert is_biased == expected_result[2]


eor_expected_result_2 = [
    {
        "equalised_odds_ratio": 0.8033707865168539,
        "privileged_sensitive_group": ["M"],
        "unprivileged_sensitive_group": ["F"],
    },
    [
        {"sensitive_group": ["M"], "score": 0.8033707865168539},
        {"sensitive_group": ["F"], "score": 1.2447552447552448},
    ],
    False,
]


eor_expected_result_3 = [
    {
        "equalised_odds_ratio": 0.0,
        "privileged_sensitive_group": ["F", "NAP"],
        "unprivileged_sensitive_group": ["M", "ASY"],
    },
    [
        {"sensitive_group": ["F", "TA"], "score": np.nan},
        {"sensitive_group": ["M", "ASY"], "score": np.nan},
        {"sensitive_group": ["M", "NAP"], "score": np.nan},
        {"sensitive_group": ["M", "ATA"], "score": np.nan},
        {"sensitive_group": ["F", "ASY"], "score": np.nan},
        {"sensitive_group": ["F", "ATA"], "score": np.nan},
        {"sensitive_group": ["F", "NAP"], "score": np.nan},
        {"sensitive_group": ["M", "TA"], "score": np.nan},
    ],
    True,
]

eor_expected_result_6 = [
    {
        "equalised_odds_ratio": np.float64(0.0),
        "privileged_sensitive_group": ["F", "NAP"],
        "unprivileged_sensitive_group": ["M", "ASY"],
    },
    [
        {"sensitive_group": ["F", "TA"], "score": np.nan},
        {"sensitive_group": ["M", "ASY"], "score": np.nan},
        {"sensitive_group": ["M", "NAP"], "score": np.nan},
        {"sensitive_group": ["M", "ATA"], "score": np.nan},
        {"sensitive_group": ["F", "ASY"], "score": np.nan},
        {"sensitive_group": ["F", "ATA"], "score": np.nan},
        {"sensitive_group": ["F", "NAP"], "score": np.nan},
        {"sensitive_group": ["M", "TA"], "score": np.nan},
    ],
    True,
]


@pytest.mark.parametrize(
    "data, threshold, expected_result",
    [
        (
            dataset1,
            1.2,
            pytest.raises(ValueError),
        ),
        (
            Dataset(df, "Sex", "HeartDisease", "HeartDiseasePred"),
            0.4,
            eor_expected_result_2,
        ),
        (
            dataset3,
            0.8,
            eor_expected_result_3,
        ),
        (dataset6, None, eor_expected_result_6),
    ],
)
def test_equalised_odds_ratio(
    data: Dataset,
    threshold: float,
    expected_result: tuple[dict, list[dict], bool] | AbstractContextManager,
):
    if not isinstance(expected_result, AbstractContextManager):
        eor = EqualisedOddsRatio(data)
        if threshold is None:
            is_biased = eor.is_biased()
        else:
            is_biased = eor.is_biased(threshold)
        assert is_biased == expected_result[2]
    else:
        with expected_result:
            eor = EqualisedOddsRatio(data)
            eor.summary()
