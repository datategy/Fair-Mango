from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from _pytest.python_api import RaisesContext

from fair_mango.dataset.dataset import Dataset
from fair_mango.metrics.metrics import (
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
)
from fair_mango.metrics.superset import (
    SupersetFairnessMetrics,
    SupersetPerformanceMetrics,
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


super_set_fairness_metrics_expected_result_1 = [
    {
        "sensitive": ("Sex",),
        "result": {
            "HeartDisease": {("M",): 0.3726567804180811, ("F",): -0.3726567804180811}
        },
    }
]


super_set_fairness_metrics_expected_result_2 = [
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


super_set_fairness_metrics_expected_result_3 = [
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
    "metric, data, sensitive, real_target, predicted_target, expected_results",
    [
        (
            DemographicParityDifference,
            dataset1,
            None,
            None,
            None,
            super_set_fairness_metrics_expected_result_1,
        ),
        (
            DisparateImpactRatio,
            df,
            ["Sex", "ChestPainType"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            super_set_fairness_metrics_expected_result_2,
        ),
        (
            EqualisedOddsDifference,
            dataset3,
            None,
            None,
            None,
            super_set_fairness_metrics_expected_result_3,
        ),
    ],
)
def test_super_set_fairness_metrics(
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
    sensitive: Sequence[str] | None,
    real_target: Sequence[str] | None,
    predicted_target: Sequence[str] | None,
    expected_results: Sequence[dict[str, dict]] | RaisesContext,
):
    super_set_fairness_metrics = SupersetFairnessMetrics(
        metric,
        data,
        sensitive,
        real_target,
        predicted_target,
    )
    results = super_set_fairness_metrics.rank()

    assert results == expected_results


super_set_performance_metrics_expected_result_2 = [
    {
        "sensitive": ("Sex",),
        "result": (
            ["HeartDisease"],
            [
                {
                    "sensitive": np.array(["M"], dtype=object),
                    "selection_rate_in_data": np.array(0.63172414),
                    "selection_rate_in_predictions": np.array(0.6262069),
                    "accuracy": [0.9779310344827586],
                    "balanced accuracy": [np.float64(0.9778470143761346)],
                    "precision": [np.float64(0.986784140969163)],
                    "recall": [np.float64(0.9781659388646288)],
                    "f1-score": [np.float64(0.9824561403508771)],
                    "false_negative_rate": [np.float64(0.021834061135371178)],
                    "false_positive_rate": [np.float64(0.02247191011235955)],
                    "true_negative_rate": [np.float64(0.9775280898876404)],
                    "true_positive_rate": [np.float64(0.9781659388646288)],
                },
                {
                    "sensitive": np.array(["F"], dtype=object),
                    "selection_rate_in_data": np.array(0.25906736),
                    "selection_rate_in_predictions": np.array(0.2642487),
                    "accuracy": [0.9637305699481865],
                    "balanced accuracy": [np.float64(0.956013986013986)],
                    "precision": [np.float64(0.9215686274509803)],
                    "recall": [np.float64(0.94)],
                    "f1-score": [np.float64(0.9306930693069307)],
                    "false_negative_rate": [np.float64(0.06)],
                    "false_positive_rate": [np.float64(0.027972027972027972)],
                    "true_negative_rate": [np.float64(0.972027972027972)],
                    "true_positive_rate": [np.float64(0.94)],
                },
            ],
        ),
    }
]


@pytest.mark.parametrize(
    "data, sensitive, real_target, predicted_target, expected_results",
    [
        (
            dataset1,
            None,
            None,
            None,
            pytest.raises(ValueError),
        ),
        (
            dataset2,
            None,
            None,
            None,
            super_set_performance_metrics_expected_result_2,
        ),
        (
            df,
            ["Sex"],
            ["HeartDisease"],
            ["HeartDiseasePred"],
            super_set_performance_metrics_expected_result_2,
        ),
    ],
)
def test_super_set_performance_metrics(
    data: pd.DataFrame | Dataset,
    sensitive: Sequence[str] | None,
    real_target: Sequence[str] | None,
    predicted_target: Sequence[str] | None,
    expected_results: list[dict] | RaisesContext,
):
    if isinstance(expected_results, list):
        super_set_performance_metrics = SupersetPerformanceMetrics(
            data,
            sensitive,
            real_target,
            predicted_target,
        )
        results = super_set_performance_metrics.evaluate()
        for result, expected_result in zip(results, expected_results):
            for result_values, expected_result_values in zip(
                result["result"][1], expected_result["result"][1]
            ):
                for value, expected_value in zip(result_values, expected_result_values):
                    if isinstance(value, np.ndarray):
                        assert (np.isclose(value, expected_value)).all()
                    elif isinstance(value, float):
                        assert np.isclose(value, expected_value)
                    else:
                        assert value == expected_value
    else:
        with expected_results:
            SupersetPerformanceMetrics(
                data,
                sensitive,
                real_target,
                predicted_target,
            ).evaluate()
