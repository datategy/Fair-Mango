from collections.abc import Sequence
from contextlib import AbstractContextManager

import numpy as np
import pandas as pd
import pytest

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

dataset1 = Dataset(df, ["Sex"], "HeartDisease")

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


super_set_fairness_metrics_expected_result_1 = [
    {
        "sensitive": ("Sex",),
        "result": {
            "HeartDisease": [
                {"sensitive": ["M"], "score": 0.3726567804180811}, 
                {"sensitive": ["F"], "score": -0.3726567804180811}
            ]
        },
    }
]


super_set_fairness_metrics_expected_result_2 = [
    {
        "sensitive": ("Sex",),
        "result": {
            "HeartDisease": [
                {"sensitive": ["M"], "score": 0.4219830636141608}, 
                {"sensitive": ["F"], "score": 2.369763353617309}
            ]
        },
    },
    {
        "sensitive": ("ChestPainType",),
        "result": {
            "HeartDisease": [
                {"sensitive": ["ASY"], "score": 0.3917632113245289},
                {"sensitive": ["TA"], "score": 0.9779803774692927},
                {"sensitive": ["NAP"], "score": 1.3200622150699777},
                {"sensitive": ["ATA"], "score": 3.612010526994567},
            ]
        },
    },
    {
        "sensitive": ("Sex", "ChestPainType"),
        "result": {
            "HeartDisease": [
                {"sensitive": ["M", "ASY"], "score": 0.3468836645650188},
                {"sensitive": ["F", "ASY"], "score": 0.5421518990141162},
                {"sensitive": ["M", "TA"], "score": 0.6173483803022393},
                {"sensitive": ["M", "NAP"], "score": 0.7690055427507021},
                {"sensitive": ["M", "ATA"], "score": 2.1240334935639593},
                {"sensitive": ["F", "TA"], "score": 3.869338673817374},
                {"sensitive": ["F", "ATA"], "score": 4.671777837152278},
                {"sensitive": ["F", "NAP"], "score": 5.173302314236593},
            ]
        },
    },
]


super_set_fairness_metrics_expected_result_3 = [
    {
        "sensitive": ("Sex",),
        "result": {
            "HeartDisease": [
                {"sensitive": ["M"], "score": 0.03816593886462882}, 
                {"sensitive": ["F"], "score": -0.03816593886462882}
            ]
        },
    },
    {
        "sensitive": ("ChestPainType",),
        "result": {
            "HeartDisease": [
                {"sensitive": ["NAP"], "score": 0.048316838338099695},
                {"sensitive": ["ASY"], "score": 0.04340882369780954},
                {"sensitive": ["ATA"], "score": 0.025394519369986518},
                {"sensitive": ["TA"], "score": -0.11712018140589575},
            ]
        },
    },
    {
        "sensitive": ("Sex", "ChestPainType"),
        "result": {
            "HeartDisease": [
                {"sensitive": ["F", "TA"], "score": 0.10053586843924016},
                {"sensitive": ["F", "ATA"], "score": 0.09033178680658709},
                {"sensitive": ["M", "NAP"], "score": 0.08199377127623639},
                {"sensitive": ["M", "ASY"], "score": 0.08153282325925479},
                {"sensitive": ["M", "ATA"], "score": 0.05205550855335028},
                {"sensitive": ["F", "ASY"], "score": 0.0014697534603408588},
                {"sensitive": ["M", "TA"], "score": -0.16240914536313006},
                {"sensitive": ["F", "NAP"], "score": -0.24551036643187954},
            ]
        },
    },
]


@pytest.mark.parametrize(
    "metric, data, expected_results",
    [
        (
            DemographicParityDifference,
            dataset1,
            super_set_fairness_metrics_expected_result_1,
        ),
        (
            DisparateImpactRatio,
            Dataset(df, ["Sex", "ChestPainType"], "HeartDisease", "HeartDiseasePred"),
            super_set_fairness_metrics_expected_result_2,
        ),
        (
            EqualisedOddsDifference,
            dataset3,
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
    data: Dataset,
    expected_results: Sequence[dict[str, dict]] | AbstractContextManager,
):
    super_set_fairness_metrics = SupersetFairnessMetrics(
        metric,
        data,
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
    "data, expected_results",
    [
        (
            dataset1,
            pytest.raises(ValueError),
        ),
        (
            dataset2,
            super_set_performance_metrics_expected_result_2,
        ),
        (
            Dataset(df, ["Sex"], "HeartDisease", "HeartDiseasePred"),
            super_set_performance_metrics_expected_result_2,
        ),
    ],
)
def test_super_set_performance_metrics(
    data: Dataset,
    expected_results: list[dict] | AbstractContextManager,
):
    if isinstance(expected_results, list):
        super_set_performance_metrics = SupersetPerformanceMetrics(
            data,
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
            ).evaluate()
