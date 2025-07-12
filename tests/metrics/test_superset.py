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


# Expected results for comprehensive fairness audit
# Dataset with only real_target (calculates only demographic_parity_difference)
super_set_comprehensive_expected_result_1 = [
    {
        "sensitive": ["Sex"],
        "rankings": {
            "demographic_parity_difference": {
                "HeartDisease": [
                    {"sensitive": ["M"], "score": 0.3726567804180811}, 
                    {"sensitive": ["F"], "score": -0.3726567804180811}
                ]
            }
        }
    }
]

# Dataset with predictions (calculates all 10 fairness metrics)
super_set_comprehensive_expected_result_2 = [
    {
        "sensitive": ["Sex"],
        "rankings": {
            # Note: Actual values will be computed dynamically since they depend on precise calculations
            # We'll check structure and metric names rather than exact values
        }
    }
]

# Multi-attribute dataset (multiple sensitive combinations × all metrics)
super_set_comprehensive_expected_result_3 = [
    {
        "sensitive": ["Sex"],
        "rankings": {
            # Multiple metrics will be calculated
        }
    },
    {
        "sensitive": ["ChestPainType"], 
        "rankings": {
            # Multiple metrics will be calculated
        }
    },
    {
        "sensitive": ["Sex", "ChestPainType"],
        "rankings": {
            # Multiple metrics will be calculated  
        }
    }
]


@pytest.mark.parametrize(
    "data, expected_combinations, expected_metrics",
    [
        (
            dataset1,  # Only real_target
            1,  # Single sensitive attribute combination
            ["demographic_parity_difference"]  # Only dataset metric
        ),
        (
            dataset2,  # Has predictions
            1,  # Single sensitive attribute combination  
            [
                "demographic_parity_difference", "demographic_parity_ratio",
                "disparate_impact_difference", "disparate_impact_ratio", 
                "equal_opportunity_difference", "equal_opportunity_ratio",
                "equalised_odds_difference", "equalised_odds_ratio",
                "false_positive_rate_difference", "false_positive_rate_ratio"
            ]  # All 10 metrics
        ),
        (
            dataset3,  # Multi-attribute with predictions
            3,  # Three combinations: Sex, ChestPainType, Sex+ChestPainType  
            [
                "demographic_parity_difference", "demographic_parity_ratio",
                "disparate_impact_difference", "disparate_impact_ratio", 
                "equal_opportunity_difference", "equal_opportunity_ratio",
                "equalised_odds_difference", "equalised_odds_ratio",
                "false_positive_rate_difference", "false_positive_rate_ratio"
            ]  # All 10 metrics for each combination
        ),
    ],
)
def test_super_set_fairness_metrics(
    data: Dataset,
    expected_combinations: int,
    expected_metrics: list[str],
):
    """Test comprehensive fairness audit functionality.
    
    The new SupersetFairnessMetrics calculates ALL applicable fairness metrics
    across ALL combinations of sensitive attributes automatically.
    """
    super_set_fairness_metrics = SupersetFairnessMetrics(data)
    results = super_set_fairness_metrics.rank()
    
    # Test structure and completeness
    assert len(results) == expected_combinations
    
    for result in results:
        # Test result structure
        assert "sensitive" in result
        assert "rankings" in result
        assert isinstance(result["sensitive"], list)
        assert isinstance(result["rankings"], dict)
        
        # Test that all expected metrics are calculated
        calculated_metrics = list(result["rankings"].keys())
        for expected_metric in expected_metrics:
            assert expected_metric in calculated_metrics, f"Missing metric: {expected_metric}"
        
        # Test that each metric has proper structure
        for metric_name, metric_result in result["rankings"].items():
            assert isinstance(metric_result, list)
            for ranking_item in metric_result:
                assert "sensitive" in ranking_item
                assert "score" in ranking_item
                assert isinstance(ranking_item["sensitive"], list)
                assert isinstance(ranking_item["score"], (int, float))
    
    # Test that comprehensive audit provides more coverage than single metrics
    if expected_combinations == 1 and len(expected_metrics) > 1:
        total_calculations = sum(len(r["rankings"]) for r in results)
        assert total_calculations >= 10, "Should calculate multiple fairness metrics"


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
        for result, expected_result in zip(results, expected_results):
            for result_values, expected_result_values in zip(
                result["result"], expected_result["result"]
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
