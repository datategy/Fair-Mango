from contextlib import AbstractContextManager

import numpy as np
import pandas as pd
import pytest

from fair_mango.dataset.dataset import Dataset
from fair_mango.metrics.superset import (
    SupersetFairnessMetrics,
    SupersetPerformanceMetrics,
)
from fair_mango.typing import (
    SupersetFairnessRankingResult,
    SupersetPerformanceMetricsResult,
)

df = pd.read_csv("tests/data/heart_data.csv")

df_multiclass = pd.DataFrame({
    'sensitive': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B'],
    'real_target': ['cat', 'dog', 'cat', 'dog', 'cat', 'bird', 'dog', 'bird'],
    'pred_target': ['cat', 'cat', 'dog', 'dog', 'cat', 'bird', 'dog', 'cat'],
})

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


superset_fairness_expected_result_dataset1 = [
    {
        "sensitive_attributes": ["Sex"],
        "rankings": {
            "demographic_parity_difference": [
                {"sensitive_group": ["M"], "score": 0.3726567804180811},
                {"sensitive_group": ["F"], "score": -0.3726567804180811},
            ]
        },
    }
]

superset_fairness_expected_result_dataset2 = [
    {
        "sensitive_attributes": ["Sex"],
        "rankings": {
            "demographic_parity_difference": [
                {"sensitive_group": ["M"], "score": 0.3726567804180811},
                {"sensitive_group": ["F"], "score": -0.3726567804180811},
            ],
            "demographic_parity_ratio": [
                {"sensitive_group": ["F"], "score": 0.4100957078534742},
                {"sensitive_group": ["M"], "score": 2.4384551724137933},
            ],
            "disparate_impact_difference": [
                {"sensitive_group": ["M"], "score": 0.3619581918885117},
                {"sensitive_group": ["F"], "score": -0.3619581918885117},
            ],
            "disparate_impact_ratio": [
                {"sensitive_group": ["F"], "score": 0.4219830636141608},
                {"sensitive_group": ["M"], "score": 2.369763353617309},
            ],
            "equal_opportunity_difference": [
                {"sensitive_group": ["M"], "score": 0.03816593886462882},
                {"sensitive_group": ["F"], "score": -0.03816593886462882},
            ],
            "equal_opportunity_ratio": [
                {"sensitive_group": ["F"], "score": 0.9609821428571428},
                {"sensitive_group": ["M"], "score": 1.0406020626219457},
            ],
            "equalised_odds_difference": [
                {"sensitive_group": ["M"], "score": 0.03816593886462882},
                {"sensitive_group": ["F"], "score": -0.03816593886462882},
            ],
            "equalised_odds_ratio": [
                {"sensitive_group": ["M"], "score": 0.8033707865168539},
                {"sensitive_group": ["F"], "score": 1.2447552447552448},
            ],
            "false_positive_rate_difference": [
                {"sensitive_group": ["F"], "score": 0.005500117859668422},
                {"sensitive_group": ["M"], "score": -0.005500117859668422},
            ],
            "false_positive_rate_ratio": [
                {"sensitive_group": ["M"], "score": 0.8033707865168539},
                {"sensitive_group": ["F"], "score": 1.2447552447552448},
            ],
        },
    }
]

superset_fairness_expected_result_dataset3 = [
    {
        "sensitive_attributes": ["Sex"],
        "rankings": {
            "demographic_parity_difference": [
                {"sensitive_group": ["M"], "score": 0.3726567804180811},
                {"sensitive_group": ["F"], "score": -0.3726567804180811},
            ]
        },
    },
    {
        "sensitive_attributes": ["ChestPainType"],
        "rankings": {
            "demographic_parity_difference": [
                {"sensitive_group": ["ASY"], "score": 0.48092566886158156},
                {"sensitive_group": ["TA"], "score": 0.006872372928902748},
                {"sensitive_group": ["NAP"], "score": -0.09993136805774676},
                {"sensitive_group": ["ATA"], "score": -0.3878666737327376},
            ]
        },
    },
    {
        "sensitive_attributes": ["Sex", "ChestPainType"],
        "rankings": {
            "demographic_parity_difference": [
                {"sensitive_group": ["M", "ASY"], "score": 0.5455262120526406},
                {"sensitive_group": ["F", "ASY"], "score": 0.23524548004152632},
                {"sensitive_group": ["M", "TA"], "score": 0.20168538933857846},
                {"sensitive_group": ["M", "NAP"], "score": 0.10136792902111814},
                {"sensitive_group": ["M", "ATA"], "score": -0.19921361333033571},
                {"sensitive_group": ["F", "NAP"], "score": -0.27210915992766893},
                {"sensitive_group": ["F", "TA"], "score": -0.28720349955031044},
                {"sensitive_group": ["F", "ATA"], "score": -0.32529873764554856},
            ]
        },
    },
]


@pytest.mark.parametrize(
    "data, expected_results",
    [
        (dataset1, superset_fairness_expected_result_dataset1),
        (dataset2, superset_fairness_expected_result_dataset2),
        (dataset3, superset_fairness_expected_result_dataset3),
    ],
)
def test_super_set_fairness_metrics(
    data: Dataset,
    expected_results: list[dict],
):
    """Test SupersetFairnessMetrics functionality.

    SupersetFairnessMetrics calculates ALL applicable fairness metrics
    across ALL combinations of sensitive attributes automatically.
    """
    super_set_fairness_metrics = SupersetFairnessMetrics(data)
    results = super_set_fairness_metrics.rank()

    assert len(results) == len(expected_results)

    for result, expected_result in zip(results, expected_results):
        assert isinstance(result, SupersetFairnessRankingResult)
        assert isinstance(result.sensitive_attributes, list)
        assert isinstance(result.rankings, dict)

        assert result.sensitive_attributes == expected_result["sensitive_attributes"]

        for metric_name, expected_metric_results in expected_result["rankings"].items():
            if metric_name in result.rankings:
                actual_metric_results = result.rankings[metric_name]

                assert isinstance(actual_metric_results, list)
                assert len(actual_metric_results) == len(expected_metric_results)

                actual_dict = {
                    tuple(item.sensitive_group): item.score
                    for item in actual_metric_results
                }
                expected_dict = {
                    tuple(item["sensitive_group"]): item["score"]
                    for item in expected_metric_results
                }

                for expected_key, expected_score in expected_dict.items():
                    assert expected_key in actual_dict, (
                        f"Missing sensitive group: {expected_key}"
                    )
                    actual_score = actual_dict[expected_key]
                    assert abs(actual_score - expected_score) < 1e-10, (
                        f"Score mismatch for {expected_key}: expected {expected_score}, got {actual_score}"
                    )

def test_super_set_fairness_metrics_multiclass():
    """
    Tests SupersetFairnessMetrics with a multi-class target.
    """
    dataset = Dataset(
        df=df_multiclass,
        sensitive=['sensitive'],
        real_target='real_target',
        predicted_target='pred_target'
    )
    
    super_set_fairness_metrics = SupersetFairnessMetrics(dataset)
    results = super_set_fairness_metrics.rank()
    
    assert isinstance(results, list)
    assert len(results) == 3 
    
    classes_of_interest = {res.positive_outcome for res in results}
    assert classes_of_interest == {'cat', 'dog', 'bird'}

    for result in results:
        assert isinstance(result, SupersetFairnessRankingResult)
        assert result.sensitive_attributes == ('sensitive',)


super_set_performance_metrics_expected_result_2 = [
    {
        "sensitive_attributes": ("Sex",),
        "data": [
            {
                "sensitive_group": np.array(["M"], dtype=object),
                "selection_rate_in_data": np.array(0.63172414),
                "selection_rate_in_predictions": np.array(0.6262069),
                "accuracy": [0.9779310344827586],
                "balanced_accuracy": [np.float64(0.9778470143761346)],
                "precision": [np.float64(0.986784140969163)],
                "recall": [np.float64(0.9781659388646288)],
                "f1_score": [np.float64(0.9824561403508771)],
                "false_negative_rate": [np.float64(0.021834061135371178)],
                "false_positive_rate": [np.float64(0.02247191011235955)],
                "true_negative_rate": [np.float64(0.9775280898876404)],
                "true_positive_rate": [np.float64(0.9781659388646288)],
            },
            {
                "sensitive_group": np.array(["F"], dtype=object),
                "selection_rate_in_data": np.array(0.25906736),
                "selection_rate_in_predictions": np.array(0.2642487),
                "accuracy": [0.9637305699481865],
                "balanced_accuracy": [np.float64(0.956013986013986)],
                "precision": [np.float64(0.9215686274509803)],
                "recall": [np.float64(0.94)],
                "f1_score": [np.float64(0.9306930693069307)],
                "false_negative_rate": [np.float64(0.06)],
                "false_positive_rate": [np.float64(0.027972027972027972)],
                "true_negative_rate": [np.float64(0.972027972027972)],
                "true_positive_rate": [np.float64(0.94)],
            },
        ],
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

        assert len(results) == len(expected_results)

        for result, expected_result in zip(results, expected_results):
            assert isinstance(result, SupersetPerformanceMetricsResult)
            assert isinstance(result.sensitive_attributes, tuple)
            assert isinstance(result.data, list)

            assert result.sensitive_attributes == expected_result["sensitive_attributes"]

            result_list = result.data
            expected_result_list = expected_result["data"]
            assert len(result_list) == len(expected_result_list)

            actual_dict = {tuple(item.sensitive_group): item for item in result_list}
            expected_dict = {
                tuple(item["sensitive_group"]): item for item in expected_result_list
            }

            for expected_key, expected_item in expected_dict.items():
                assert expected_key in actual_dict, (
                    f"Missing sensitive group: {expected_key}"
                )
                result_item = actual_dict[expected_key]

                for key, expected_value in expected_item.items():
                    assert hasattr(result_item, key), (
                        f"Missing metric {key} for group {expected_key}"
                    )
                    actual_value = getattr(result_item, key)

                    if isinstance(expected_value, np.ndarray):
                        if expected_value.dtype.kind in ["U", "S", "O"]:
                            assert np.array_equal(actual_value, expected_value)
                        else:
                            assert np.allclose(actual_value, expected_value)
                    elif isinstance(expected_value, list) and len(expected_value) > 0:
                        if isinstance(expected_value[0], (int, float)):
                            assert np.allclose(actual_value, expected_value)
                        else:
                            assert actual_value == expected_value
                    else:
                        if isinstance(actual_value, np.ndarray) and isinstance(
                            expected_value, np.ndarray
                        ):
                            assert np.array_equal(actual_value, expected_value)
                        else:
                            assert actual_value == expected_value
    else:
        with expected_results:
            SupersetPerformanceMetrics(
                data,
            ).evaluate()

def test_super_set_performance_metrics_multiclass():
    """
    Tests SupersetPerformanceMetrics with a multi-class target.
    """
    dataset = Dataset(
        df=df_multiclass,
        sensitive=['sensitive'],
        real_target='real_target',
        predicted_target='pred_target'
    )
    
    super_set_performance_metrics = SupersetPerformanceMetrics(dataset)
    results = super_set_performance_metrics.evaluate()
    
    assert isinstance(results, list)
    assert len(results) == 3
    
    classes_of_interest = {res.positive_outcome for res in results}
    assert classes_of_interest == {'cat', 'dog', 'bird'}

    for result in results:
        assert isinstance(result, SupersetPerformanceMetricsResult)
        assert result.sensitive_attributes == ('sensitive',)