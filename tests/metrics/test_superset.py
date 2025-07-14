from collections.abc import Sequence
from contextlib import AbstractContextManager

import numpy as np
import pandas as pd
import pytest

from fair_mango.dataset.dataset import Dataset
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


# Expected results for SupersetFairnessMetrics - dataset1 (only demographic_parity_difference)
superset_fairness_expected_result_dataset1 = [
    {
        "sensitive": ["Sex"],
        "rankings": {
            "demographic_parity_difference": [
                {"sensitive": ["M"], "score": 0.3726567804180811},
                {"sensitive": ["F"], "score": -0.3726567804180811}
            ]
        }
    }
]

# Expected results for SupersetFairnessMetrics - dataset2 (all fairness metrics)
superset_fairness_expected_result_dataset2 = [
    {
        "sensitive": ["Sex"],
        "rankings": {
            "demographic_parity_difference": [
                {"sensitive": ["M"], "score": 0.3726567804180811},
                {"sensitive": ["F"], "score": -0.3726567804180811}
            ],
            "demographic_parity_ratio": [
                {"sensitive": ["F"], "score": 0.4100957078534742},
                {"sensitive": ["M"], "score": 2.4384551724137933}
            ],
            "disparate_impact_difference": [
                {"sensitive": ["M"], "score": 0.3619581918885117},
                {"sensitive": ["F"], "score": -0.3619581918885117}
            ],
            "disparate_impact_ratio": [
                {"sensitive": ["F"], "score": 0.4219830636141608},
                {"sensitive": ["M"], "score": 2.369763353617309}
            ],
            "equal_opportunity_difference": [
                {"sensitive": ["M"], "score": 0.03816593886462882},
                {"sensitive": ["F"], "score": -0.03816593886462882}
            ],
            "equal_opportunity_ratio": [
                {"sensitive": ["F"], "score": 0.9609821428571428},
                {"sensitive": ["M"], "score": 1.0406020626219457}
            ],
            "equalised_odds_difference": [
                {"sensitive": ["M"], "score": 0.03816593886462882},
                {"sensitive": ["F"], "score": -0.03816593886462882}
            ],
            "equalised_odds_ratio": [
                {"sensitive": ["M"], "score": 0.8033707865168539},
                {"sensitive": ["F"], "score": 1.2447552447552448}
            ],
            "false_positive_rate_difference": [
                {"sensitive": ["F"], "score": 0.005500117859668422}, 
                {"sensitive": ["M"], "score": -0.005500117859668422} 
            ],
            "false_positive_rate_ratio": [
                {"sensitive": ["M"], "score": 0.8033707865168539},
                {"sensitive": ["F"], "score": 1.2447552447552448}
            ]
        }
    }
]

# Expected results for SupersetFairnessMetrics - dataset3 (multiple sensitive attributes)
superset_fairness_expected_result_dataset3 = [
    {
        "sensitive": ["Sex"],
        "rankings": {
            "demographic_parity_difference": [
                {"sensitive": ["M"], "score": 0.3726567804180811},
                {"sensitive": ["F"], "score": -0.3726567804180811}
            ]
        }
    },
    {
        "sensitive": ["ChestPainType"],
        "rankings": {
            "demographic_parity_difference": [
                {"sensitive": ["ASY"], "score": 0.4356427776894962},
                {"sensitive": ["TA"], "score": -0.3555399719495091},
                {"sensitive": ["NAP"], "score": -0.4356427776894962},
                {"sensitive": ["ATA"], "score": -0.6515942569457394}
            ]
        }
    },
    {
        "sensitive": ["Sex", "ChestPainType"],
        "rankings": {
            "demographic_parity_difference": [
                {"sensitive": ["M", "ASY"], "score": 0.3886384976525821}, 
                {"sensitive": ["F", "ASY"], "score": -0.27149564050972497}, 
                {"sensitive": ["M", "TA"], "score": -0.30086071987480434}, 
                {"sensitive": ["M", "NAP"], "score": -0.3886384976525821}, 
                {"sensitive": ["M", "ATA"], "score": -0.6516473472101043}, 
                {"sensitive": ["F", "NAP"], "score": -0.7154309504827708}, 
                {"sensitive": ["F", "TA"], "score": -0.7286384976525822}, 
                {"sensitive": ["F", "ATA"], "score": -0.7619718309859155},
            ]
        }
    }
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
    
    # Test basic structure
    assert len(results) == len(expected_results)
    
    for result, expected_result in zip(results, expected_results):
        # Test result structure
        assert "sensitive" in result
        assert "rankings" in result
        assert isinstance(result["sensitive"], list)
        assert isinstance(result["rankings"], dict)
        
        # Test sensitive attributes match
        assert result["sensitive"] == expected_result["sensitive"]
        
        # Test that expected metrics are calculated
        for metric_name, expected_metric_results in expected_result["rankings"].items():
            if metric_name in result["rankings"]:
                actual_metric_results = result["rankings"][metric_name]
                
                # Test metric structure
                assert isinstance(actual_metric_results, list)
                assert len(actual_metric_results) == len(expected_metric_results)
                
                # Create dictionaries for easier comparison (order-independent)
                actual_dict = {tuple(item["sensitive"]): item["score"] for item in actual_metric_results}
                expected_dict = {tuple(item["sensitive"]): item["score"] for item in expected_metric_results}
                
                # Test that all expected items are present with correct scores
                for expected_key, expected_score in expected_dict.items():
                    assert expected_key in actual_dict, f"Missing sensitive group: {expected_key}"
                    actual_score = actual_dict[expected_key]
                    assert abs(actual_score - expected_score) < 1e-10, f"Score mismatch for {expected_key}: expected {expected_score}, got {actual_score}"


super_set_performance_metrics_expected_result_2 = [
    {
        "sensitive": ("Sex",),
        "result": [
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
        
        # Test basic structure
        assert len(results) == len(expected_results)
        
        for result, expected_result in zip(results, expected_results):
            # Test dictionary structure
            assert isinstance(result, dict)
            assert "sensitive" in result
            assert "result" in result
            assert isinstance(result["sensitive"], tuple)
            assert isinstance(result["result"], list)
            
            # Test sensitive attributes match
            assert result["sensitive"] == expected_result["sensitive"]
            
            # Test the result list structure
            result_list = result["result"]
            expected_result_list = expected_result["result"]
            assert len(result_list) == len(expected_result_list)
            
            # Create dictionaries for easier comparison (order-independent)
            actual_dict = {tuple(item["sensitive"]): item for item in result_list}
            expected_dict = {tuple(item["sensitive"]): item for item in expected_result_list}
            
            # Test that all expected groups are present with correct metrics
            for expected_key, expected_item in expected_dict.items():
                assert expected_key in actual_dict, f"Missing sensitive group: {expected_key}"
                result_item = actual_dict[expected_key]
                
                # Test each metric in the result
                for key, expected_value in expected_item.items():
                    assert key in result_item, f"Missing metric {key} for group {expected_key}"
                    actual_value = result_item[key]
                    
                    if isinstance(expected_value, np.ndarray):
                        # Handle string arrays differently from numeric arrays
                        if expected_value.dtype.kind in ['U', 'S', 'O']:  # String types
                            assert np.array_equal(actual_value, expected_value)
                        else:
                            assert np.allclose(actual_value, expected_value)
                    elif isinstance(expected_value, list) and len(expected_value) > 0:
                        if isinstance(expected_value[0], (int, float)):
                            assert np.allclose(actual_value, expected_value)
                        else:
                            assert actual_value == expected_value
                    else:
                        if isinstance(actual_value, np.ndarray) and isinstance(expected_value, np.ndarray):
                            assert np.array_equal(actual_value, expected_value)
                        else:
                            assert actual_value == expected_value
    else:
        with expected_results:
            SupersetPerformanceMetrics(
                data,
            ).evaluate()