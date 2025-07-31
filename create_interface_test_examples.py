#!/usr/bin/env python3
"""
Interface Test Examples Generator

This script creates comprehensive test examples for the Fair-Mango interface,
demonstrating different use cases and expected outputs.
"""

import pandas as pd
from fair_mango.dataset.dataset import Dataset
from fair_mango.metrics.superset import (
    SupersetFairnessMetrics,
    SupersetPerformanceMetrics,
)
import json
from pathlib import Path
from typing import Any
import dataclasses
import numpy as np


def to_dict(obj: Any) -> Any:
    """Convert object to dictionary, handling dataclasses and other types."""
    if dataclasses.is_dataclass(obj):
        result = dataclasses.asdict(obj)
        return _clean_infinite_values(result)
    elif isinstance(obj, dict):
        return _clean_infinite_values(obj)
    elif isinstance(obj, list):
        return [to_dict(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        value = obj.item()
        if np.isinf(value):
            return None  # or "Infinity" as string
        return value
    elif isinstance(obj, float) and np.isinf(obj):
        return None  # or "Infinity" as string
    else:
        return obj


def _clean_infinite_values(obj: Any) -> Any:
    """Recursively clean infinite values from nested structures."""
    if isinstance(obj, dict):
        return {key: _clean_infinite_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_clean_infinite_values(item) for item in obj]
    elif isinstance(obj, float) and np.isinf(obj):
        return None  # or "Infinity" as string
    elif isinstance(obj, (np.integer, np.floating)):
        value = obj.item()
        if np.isinf(value):
            return None  # or "Infinity" as string
        return value
    else:
        return obj





def create_interface_test_scenarios() -> list[dict[str, Any]]:
    """
    Create comprehensive test scenarios for the interface.

    Returns
    -------
    List[Dict[str, Any]]
        List of test scenarios with inputs and expected outputs
    """
    scenarios = []

    # Load all datasets
    datasets_info = {
        "heart_disease": {
            "file": "tests/data/heart_data.csv",
            "sensitive_attrs": ["Sex", "ChestPainType"],
            "targets": {
                "HeartDisease": {"pred": "HeartDiseasePred", "positive": 1},
                "ExerciseAngina": {"pred": "ExerciseAnginaPred", "positive": "Y"},
            },
        },
        "hiring": {
            "file": "tests/data/hiring_data.csv",
            "sensitive_attrs": ["Gender", "Race", "AgeGroup"],
            "targets": {"Hired": {"pred": "HiredPred", "positive": 1}},
        },
        "loan_approval": {
            "file": "tests/data/loan_approval_data.csv",
            "sensitive_attrs": ["Gender", "Race"],
            "targets": {"Approved": {"pred": "ApprovedPred", "positive": 1}},
        },
        "criminal_justice": {
            "file": "tests/data/criminal_justice_data.csv",
            "sensitive_attrs": ["Race", "Gender"],
            "targets": {"Recidivism": {"pred": "RecidivismPred", "positive": 1}},
        },
        "college_admission": {
            "file": "tests/data/college_admission_data.csv",
            "sensitive_attrs": ["Race", "Gender"],
            "targets": {"Admitted": {"pred": "AdmittedPred", "positive": 1}},
        },
    }

    for dataset_name, info in datasets_info.items():
        try:
            df = pd.read_csv(info["file"])
            print(f"\nProcessing dataset: {dataset_name}")

            for target_name, target_info in info["targets"].items():
                # Scenario 1: Single sensitive attribute
                for sensitive_attr in info["sensitive_attrs"]:
                    scenario = create_single_scenario(
                        dataset_name=dataset_name,
                        df=df,
                        sensitive=[sensitive_attr],
                        target=target_name,
                        predicted=target_info["pred"],
                        positive=target_info["positive"],
                        scenario_type="single_sensitive",
                    )
                    scenarios.append(scenario)

                # Scenario 2: Multiple sensitive attributes (first two)
                if len(info["sensitive_attrs"]) >= 2:
                    print(
                        f"Creating multiple_sensitive scenario for {dataset_name} with attributes: {info['sensitive_attrs'][:2]}"
                    )
                    scenario = create_single_scenario(
                        dataset_name=dataset_name,
                        df=df,
                        sensitive=info["sensitive_attrs"][:2],
                        target=target_name,
                        predicted=target_info["pred"],
                        positive=target_info["positive"],
                        scenario_type="multiple_sensitive",
                    )
                    scenarios.append(scenario)

                # Scenario 3: Without predictions (fairness in ground truth)
                scenario = create_single_scenario(
                    dataset_name=dataset_name,
                    df=df,
                    sensitive=[info["sensitive_attrs"][0]],
                    target=target_name,
                    predicted=None,
                    positive=target_info["positive"],
                    scenario_type="no_predictions",
                )
                scenarios.append(scenario)

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue

    return scenarios


def create_single_scenario(
    dataset_name: str,
    df: pd.DataFrame,
    sensitive: list[str],
    target: str,
    predicted: str = None,
    positive=None,
    scenario_type: str = "",
) -> dict[str, Any]:
    """
    Create a single test scenario.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    df : pd.DataFrame
        The dataframe
    sensitive : List[str]
        Sensitive attributes
    target : str
        Target variable
    predicted : str, optional
        Predicted variable
    positive : Any, optional
        Positive class value
    scenario_type : str
        Type of scenario

    Returns
    -------
    Dict[str, Any]
        Scenario dictionary
    """
    try:
        # Create dataset
        dataset = Dataset(
            df=df,
            sensitive=sensitive,
            real_target=target,
            predicted_target=predicted,
            positive_target=positive,
        )

        scenario = {
            "dataset_name": dataset_name,
            "scenario_type": scenario_type,
            "description": f"{dataset_name} - {scenario_type} - {target}",
            "input": {
                "sensitive_attributes": sensitive,
                "target_variable": target,
                "predicted_variable": predicted,
                "positive_class": positive,
                "data_shape": df.shape,
                "sensitive_groups": {},
            },
            "outputs": {},
        }

        # Get sensitive group information
        groups_data = dataset.get_data_for_all_groups()
        for group in groups_data:
            if len(group.sensitive_group) == 1:
                group_key = group.sensitive_group[0]
            else:
                group_key = "_".join(group.sensitive_group)

            scenario["input"]["sensitive_groups"][group_key] = {
                "sensitive_group": group.sensitive_group,
                "size": len(group.data),
                "target_distribution": group.data[target].value_counts().to_dict(),
            }

        # Calculate fairness metrics if predictions available
        if predicted is not None:
            try:
                fairness_metrics = SupersetFairnessMetrics(dataset)
                fairness_results = fairness_metrics.rank()
                fairness_summaries = fairness_metrics.summary()

                # Clean up fairness results to avoid tuple string representations
                clean_fairness_results = []
                for result in fairness_results:  # Include all combinations
                    result_dict = to_dict(result)
                    # Clean sensitive_group representation (removed key and values fields)

                    # Clean rankings to avoid tuple representations
                    if "rankings" in result_dict:
                        clean_rankings = {}
                        for metric_name, rankings in result_dict["rankings"].items():
                            clean_rankings[metric_name] = []
                            for rank in rankings:
                                clean_rank = to_dict(rank)
                                # Clean sensitive_group representation (removed key and values fields)
                                clean_rankings[metric_name].append(clean_rank)
                        result_dict["rankings"] = clean_rankings

                    clean_fairness_results.append(result_dict)

                # Clean up fairness summaries
                clean_fairness_summaries = []
                for summary in fairness_summaries:
                    summary_dict = to_dict(summary)
                    # Clean sensitive_group representation (removed key and values fields)
                    clean_fairness_summaries.append(summary_dict)

                # Clean fairness results are ready for assignment
                scenario["outputs"]["fairness_metrics"] = {
                    "available_metrics": list(fairness_results[0].rankings.keys())
                    if fairness_results
                    else [],
                    "sample_results": clean_fairness_results,
                    "summary_results": clean_fairness_summaries,
                }
            except Exception as e:
                scenario["outputs"]["fairness_metrics"] = {"error": str(e)}

            try:
                performance_metrics = SupersetPerformanceMetrics(dataset)
                performance_results = performance_metrics.evaluate()

                # Clean up performance results
                clean_performance_results = []
                for result in performance_results:  # Include all combinations
                    clean_result = {"data_length": len(result.data)}
                    # Clean sensitive_group representation (removed key and values fields)
                    clean_result["sensitive_group"] = list(result.sensitive_group) if isinstance(result.sensitive_group, (list, tuple)) else [str(result.sensitive_group)]

                    clean_performance_results.append(clean_result)

                scenario["outputs"]["performance_metrics"] = {
                    "available_metrics": [
                        "selection_rate",
                        "performance_metric",
                        "confusion_matrix",
                    ],
                    "sample_results": clean_performance_results,
                }
            except Exception as e:
                scenario["outputs"]["performance_metrics"] = {"error": str(e)}

        # Calculate basic statistics
        scenario["outputs"]["basic_statistics"] = {
            "total_samples": len(df),
            "target_distribution": df[target].value_counts().to_dict(),
            "sensitive_group_counts": {},
        }

        if len(sensitive) == 1:
            group_counts = df[sensitive[0]].value_counts().to_dict()
            scenario["outputs"]["basic_statistics"]["sensitive_group_counts"] = (
                group_counts
            )
        else:
            # Start with individual sensitive group counts
            clean_group_counts = {}
            for attr in sensitive:
                individual_counts = df[attr].value_counts().to_dict()
                clean_group_counts.update(individual_counts)

            # Add intersectional combinations (skip single-element tuples as we already have individual counts)
            group_counts = df.groupby(sensitive).size().to_dict()
            for k, v in group_counts.items():
                if isinstance(k, tuple):
                    if len(k) > 1:  # Only add multi-element combinations
                        clean_key = "_".join(str(x) for x in k)
                        clean_group_counts[clean_key] = v
                # Skip non-tuple keys as they would be single attributes already covered

            scenario["outputs"]["basic_statistics"]["sensitive_group_counts"] = (
                clean_group_counts
            )

        return scenario

    except Exception as e:
        return {
            "dataset_name": dataset_name,
            "scenario_type": scenario_type,
            "description": f"{dataset_name} - {scenario_type} - {target}",
            "error": str(e),
            "input": {
                "sensitive_attributes": sensitive,
                "target_variable": target,
                "predicted_variable": predicted,
                "positive_class": positive,
            },
        }


def create_api_examples() -> dict[str, Any]:
    """
    Create API usage examples for the interface documentation.

    Returns
    -------
    Dict[str, Any]
        API examples
    """
    examples = {
        "basic_usage": {
            "description": "Basic fairness analysis with single sensitive attribute",
            "code": """
# Load data
import pandas as pd
from fair_mango.dataset.dataset import Dataset
from fair_mango.metrics.superset import SupersetFairnessMetrics

df = pd.read_csv("heart_data.csv")

# Create dataset
dataset = Dataset(
    df=df,
    sensitive=["Sex"],
    real_target="HeartDisease",
    predicted_target="HeartDiseasePred",
    positive_target=1
)

# Calculate fairness metrics
fairness = SupersetFairnessMetrics(dataset)
results = fairness.rank()

# Results contain rankings for each metric
for result in results:
    print(f"Sensitive attribute: {result['sensitive_group']}")
    for metric, rankings in result['rankings'].items():
        print(f"  {metric}:")
        for rank in rankings:
            print(f"    {rank['sensitive_group']}: {rank['score']:.4f}")
""",
            "expected_output_structure": {
                "type": "list",
                "items": {
                    "sensitive_group": "list of sensitive attribute names",
                    "rankings": {
                        "metric_name": [
                            {
                                "sensitive_group": "group values",
                                "score": "numeric score",
                            }
                        ]
                    },
                },
            },
        },
        "multiple_sensitive": {
            "description": "Analysis with multiple sensitive attributes",
            "code": """
# Multiple sensitive attributes
dataset = Dataset(
    df=df,
    sensitive=["Sex", "ChestPainType"],
    real_target="HeartDisease",
    predicted_target="HeartDiseasePred",
    positive_target=1
)

fairness = SupersetFairnessMetrics(dataset)
results = fairness.rank()

# Results will include individual and intersectional analysis
# - Individual: Sex only, ChestPainType only
# - Intersectional: Sex × ChestPainType combinations
""",
        },
        "performance_metrics": {
            "description": "Performance metrics analysis",
            "code": """
from fair_mango.metrics.superset import SupersetPerformanceMetrics

# Performance metrics
performance = SupersetPerformanceMetrics(dataset)
results = performance.evaluate()

# Available metrics: selection_rate, performance_metric, confusion_matrix
""",
        },
        "no_predictions": {
            "description": "Fairness analysis without predictions (ground truth only)",
            "code": """
# Analyze fairness in ground truth labels
dataset = Dataset(
    df=df,
    sensitive=["Sex"],
    real_target="HeartDisease",
    positive_target=1
    # No predicted_target specified
)

# Only demographic parity metrics available
fairness = SupersetFairnessMetrics(dataset)
results = fairness.rank()
""",
        },
        "different_positive_class": {
            "description": "Using different positive class values",
            "code": """
# String positive class
dataset = Dataset(
    df=df,
    sensitive=["Sex"],
    real_target="ExerciseAngina",
    predicted_target="ExerciseAnginaPred",
    positive_target="Y"  # String value
)

# Boolean positive class
# positive_target=True

# Numeric positive class
# positive_target=1
""",
        },
    }

    return examples


def create_metric_definitions() -> dict[str, Any]:
    """
    Create definitions and interpretations for all metrics.

    Returns
    -------
    Dict[str, Any]
        Metric definitions
    """
    return {
        "fairness_metrics": {
            "demographic_parity_difference": {
                "description": "Difference in positive prediction rates between groups",
                "formula": "P(Ŷ=1|A=a) - P(Ŷ=1|A=b)",
                "interpretation": {
                    "range": "[-1, 1]",
                    "fair_value": "0",
                    "positive_values": "Group a has higher positive prediction rate",
                    "negative_values": "Group b has higher positive prediction rate",
                },
                "use_case": "Ensures equal treatment regardless of sensitive attribute",
            },
            "demographic_parity_ratio": {
                "description": "Ratio of positive prediction rates between groups",
                "formula": "P(Ŷ=1|A=a) / P(Ŷ=1|A=b)",
                "interpretation": {
                    "range": "[0, ∞]",
                    "fair_value": "1",
                    "values_gt_1": "Group a has higher positive prediction rate",
                    "values_lt_1": "Group b has higher positive prediction rate",
                },
            },
            "equal_opportunity_difference": {
                "description": "Difference in true positive rates between groups",
                "formula": "P(Ŷ=1|Y=1,A=a) - P(Ŷ=1|Y=1,A=b)",
                "interpretation": {
                    "range": "[-1, 1]",
                    "fair_value": "0",
                    "meaning": "Equal opportunity for positive outcomes",
                },
            },
            "equalised_odds_difference": {
                "description": "Difference in both TPR and FPR between groups",
                "interpretation": {
                    "meaning": "Equal accuracy across groups",
                    "fair_value": "0 for both TPR and FPR differences",
                },
            },
            "disparate_impact_ratio": {
                "description": "Ratio of positive outcomes in actual labels",
                "formula": "P(Y=1|A=a) / P(Y=1|A=b)",
                "interpretation": {
                    "legal_threshold": "0.8 (80% rule)",
                    "fair_value": "1",
                },
            },
        },
        "performance_metrics": {
            "accuracy": {
                "description": "Proportion of correct predictions",
                "formula": "(TP + TN) / (TP + TN + FP + FN)",
                "range": "[0, 1]",
                "best_value": "1",
            },
            "precision": {
                "description": "Proportion of positive predictions that are correct",
                "formula": "TP / (TP + FP)",
                "range": "[0, 1]",
                "best_value": "1",
            },
            "recall": {
                "description": "Proportion of actual positives correctly identified",
                "formula": "TP / (TP + FN)",
                "range": "[0, 1]",
                "best_value": "1",
            },
            "f1_score": {
                "description": "Harmonic mean of precision and recall",
                "formula": "2 * (precision * recall) / (precision + recall)",
                "range": "[0, 1]",
                "best_value": "1",
            },
        },
    }


def save_interface_documentation(
    scenarios: list[dict], examples: dict, metrics: dict, output_dir: str = "tests/data"
) -> None:
    """
    Save comprehensive interface documentation.

    Parameters
    ----------
    scenarios : List[Dict]
        Test scenarios
    examples : Dict
        API examples
    metrics : Dict
        Metric definitions
    output_dir : str
        Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save scenarios
    with open(output_path / "interface_test_scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2, default=str)

    # Save examples
    with open(output_path / "api_examples.json", "w") as f:
        json.dump(examples, f, indent=2)

    # Save metric definitions
    with open(output_path / "metric_definitions.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Create comprehensive markdown documentation
    create_interface_guide(scenarios, examples, metrics, output_path)


def create_interface_guide(
    scenarios: list[dict], examples: dict, metrics: dict, output_path: Path
) -> None:
    """
    Create a comprehensive interface guide.

    Parameters
    ----------
    scenarios : List[Dict]
        Test scenarios
    examples : Dict
        API examples
    metrics : Dict
        Metric definitions
    output_path : Path
        Output directory path
    """
    guide = f"""
# Fair-Mango Interface Testing Guide

This guide provides comprehensive examples and test scenarios for the Fair-Mango interface.

## Quick Start

### Basic Usage
```python
{examples["basic_usage"]["code"]}
```

## Available Datasets

The following synthetic datasets are available for testing:

1. **Heart Disease** (`heart_data.csv`) - Medical diagnosis fairness
2. **Hiring** (`hiring_data.csv`) - Employment fairness
3. **Loan Approval** (`loan_approval_data.csv`) - Financial fairness
4. **Criminal Justice** (`criminal_justice_data.csv`) - Recidivism prediction
5. **College Admission** (`college_admission_data.csv`) - Educational fairness

## Test Scenarios

We have generated {len(scenarios)} test scenarios covering:

"""

    # Count scenarios by type
    scenario_types = {}
    for scenario in scenarios:
        stype = scenario.get("scenario_type", "unknown")
        scenario_types[stype] = scenario_types.get(stype, 0) + 1

    for stype, count in scenario_types.items():
        guide += f"- **{stype.replace('_', ' ').title()}**: {count} scenarios\n"

    guide += """

### Scenario Structure

Each test scenario includes:
- **Input parameters**: sensitive attributes, target variables, positive class
- **Data characteristics**: sample size, group distributions
- **Expected outputs**: available metrics, sample results
- **Error handling**: graceful degradation for edge cases

## API Examples

"""

    for example_name, example_data in examples.items():
        guide += f"""
### {example_name.replace("_", " ").title()}

{example_data["description"]}

```python
{example_data["code"]}
```

"""

    guide += """
## Fairness Metrics

### Available Metrics

"""

    for metric_name, metric_info in metrics["fairness_metrics"].items():
        guide += f"""
#### {metric_name.replace("_", " ").title()}

**Description**: {metric_info["description"]}

"""
        if "formula" in metric_info:
            guide += f"**Formula**: `{metric_info['formula']}`\n\n"

        if "interpretation" in metric_info:
            interp = metric_info["interpretation"]
            guide += "**Interpretation**:\n"
            for key, value in interp.items():
                guide += f"- {key.replace('_', ' ').title()}: {value}\n"
            guide += "\n"

    guide += """
## Performance Metrics

"""

    for metric_name, metric_info in metrics["performance_metrics"].items():
        guide += f"""
### {metric_name.replace("_", " ").title()}

**Description**: {metric_info["description"]}
**Formula**: `{metric_info["formula"]}`
**Range**: {metric_info["range"]}
**Best Value**: {metric_info["best_value"]}

"""

    guide += """
## Interface Testing Checklist

### Data Loading
- [ ] CSV file upload functionality
- [ ] Column selection for sensitive attributes
- [ ] Target variable selection
- [ ] Positive class specification
- [ ] Data validation and error messages

### Metric Calculation
- [ ] Fairness metrics computation
- [ ] Performance metrics computation
- [ ] Handling of missing predictions
- [ ] Error handling for invalid inputs

### Results Display
- [ ] Metric rankings visualization
- [ ] Group comparison tables
- [ ] Interactive charts and graphs
- [ ] Export functionality

### Edge Cases
- [ ] Single group in sensitive attribute
- [ ] Missing values in data
- [ ] Imbalanced datasets
- [ ] Large datasets (performance)
- [ ] Invalid column selections

## Sample Test Cases

### Test Case 1: Basic Fairness Analysis
```python
# Load heart disease data
df = pd.read_csv("tests/data/heart_data.csv")
dataset = Dataset(df, ["Sex"], "HeartDisease", "HeartDiseasePred", 1)

# Should return fairness metrics for Male vs Female groups
fairness = SupersetFairnessMetrics(dataset)
results = fairness.rank()

assert len(results) == 1  # One sensitive attribute
assert "demographic_parity_difference" in results[0]['rankings']
assert len(results[0]['rankings']['demographic_parity_difference']) == 2  # M and F
```

### Test Case 2: Multiple Sensitive Attributes
```python
# Multiple sensitive attributes
dataset = Dataset(df, ["Sex", "ChestPainType"], "HeartDisease", "HeartDiseasePred", 1)
fairness = SupersetFairnessMetrics(dataset)
results = fairness.rank()

# Should return results for Sex, ChestPainType, and Sex×ChestPainType
assert len(results) == 3
```

### Test Case 3: No Predictions
```python
# Ground truth only
dataset = Dataset(df, ["Sex"], "HeartDisease", positive_target=1)
fairness = SupersetFairnessMetrics(dataset)
results = fairness.rank()

# Should only have demographic parity metrics
available_metrics = results[0]['rankings'].keys()
assert "demographic_parity_difference" in available_metrics
assert "equal_opportunity_difference" not in available_metrics
```

## Error Scenarios

### Invalid Column Names
```python
# Should raise KeyError
try:
    dataset = Dataset(df, ["NonexistentColumn"], "HeartDisease")
    assert False, "Should have raised KeyError"
except KeyError:
    pass  # Expected
```

### Same Column for Multiple Purposes
```python
# Should raise AttributeError
try:
    dataset = Dataset(df, ["Sex"], "Sex")  # Same column as sensitive and target
    assert False, "Should have raised AttributeError"
except AttributeError:
    pass  # Expected
```

## Performance Benchmarks

- **Small datasets** (< 1,000 rows): < 1 second
- **Medium datasets** (1,000 - 10,000 rows): < 5 seconds
- **Large datasets** (10,000+ rows): < 30 seconds

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Accessibility

- Screen reader compatible
- Keyboard navigation
- High contrast mode
- Responsive design

"""

    with open(output_path / "INTERFACE_TESTING_GUIDE.md", "w") as f:
        f.write(guide)


if __name__ == "__main__":
    print("Creating interface test examples and documentation...")
    print("=" * 60)

    # Generate test scenarios
    print("Generating test scenarios...")
    scenarios = create_interface_test_scenarios()
    print(f"Generated {len(scenarios)} test scenarios")

    # Create API examples
    print("Creating API examples...")
    examples = create_api_examples()

    # Create metric definitions
    print("Creating metric definitions...")
    metrics = create_metric_definitions()

    # Save all documentation
    print("Saving documentation...")
    save_interface_documentation(scenarios, examples, metrics)

    print("\nInterface testing documentation created!")
    print("\nFiles generated:")
    print("  - tests/data/interface_test_scenarios.json")
    print("  - tests/data/api_examples.json")
    print("  - tests/data/metric_definitions.json")
    print("  - tests/data/INTERFACE_TESTING_GUIDE.md")

    # Print summary statistics
    successful_scenarios = [s for s in scenarios if "error" not in s]
    error_scenarios = [s for s in scenarios if "error" in s]

    print("\nScenario Summary:")
    print(f"  - Successful: {len(successful_scenarios)}")
    print(f"  - Errors: {len(error_scenarios)}")

    if error_scenarios:
        print("\nError scenarios:")
        for scenario in error_scenarios[:5]:  # Show first 5 errors
            print(f"  - {scenario['description']}: {scenario['error']}")

    print("\nInterface testing resources are ready!")
