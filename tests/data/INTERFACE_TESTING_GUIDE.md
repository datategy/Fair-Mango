
# Fair-Mango Interface Testing Guide

This guide provides comprehensive examples and test scenarios for the Fair-Mango interface.

## Quick Start

### Basic Usage
```python

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

```

## Available Datasets

The following synthetic datasets are available for testing:

1. **Heart Disease** (`heart_data.csv`) - Medical diagnosis fairness
2. **Hiring** (`hiring_data.csv`) - Employment fairness
3. **Loan Approval** (`loan_approval_data.csv`) - Financial fairness
4. **Criminal Justice** (`criminal_justice_data.csv`) - Recidivism prediction
5. **College Admission** (`college_admission_data.csv`) - Educational fairness

## Test Scenarios

We have generated 25 test scenarios covering:

- **Single Sensitive**: 13 scenarios
- **Multiple Sensitive**: 6 scenarios
- **No Predictions**: 6 scenarios


### Scenario Structure

Each test scenario includes:
- **Input parameters**: sensitive attributes, target variables, positive class
- **Data characteristics**: sample size, group distributions
- **Expected outputs**: available metrics, sample results
- **Error handling**: graceful degradation for edge cases

## API Examples


### Basic Usage

Basic fairness analysis with single sensitive attribute

```python

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

```


### Multiple Sensitive

Analysis with multiple sensitive attributes

```python

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

```


### Performance Metrics

Performance metrics analysis

```python

from fair_mango.metrics.superset import SupersetPerformanceMetrics

# Performance metrics
performance = SupersetPerformanceMetrics(dataset)
results = performance.evaluate()

# Available metrics: selection_rate, performance_metric, confusion_matrix

```


### No Predictions

Fairness analysis without predictions (ground truth only)

```python

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

```


### Different Positive Class

Using different positive class values

```python

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

```


## Fairness Metrics

### Available Metrics


#### Demographic Parity Difference

**Description**: Difference in positive prediction rates between groups

**Formula**: `P(Ŷ=1|A=a) - P(Ŷ=1|A=b)`

**Interpretation**:
- Range: [-1, 1]
- Fair Value: 0
- Positive Values: Group a has higher positive prediction rate
- Negative Values: Group b has higher positive prediction rate


#### Demographic Parity Ratio

**Description**: Ratio of positive prediction rates between groups

**Formula**: `P(Ŷ=1|A=a) / P(Ŷ=1|A=b)`

**Interpretation**:
- Range: [0, ∞]
- Fair Value: 1
- Values Gt 1: Group a has higher positive prediction rate
- Values Lt 1: Group b has higher positive prediction rate


#### Equal Opportunity Difference

**Description**: Difference in true positive rates between groups

**Formula**: `P(Ŷ=1|Y=1,A=a) - P(Ŷ=1|Y=1,A=b)`

**Interpretation**:
- Range: [-1, 1]
- Fair Value: 0
- Meaning: Equal opportunity for positive outcomes


#### Equalised Odds Difference

**Description**: Difference in both TPR and FPR between groups

**Interpretation**:
- Meaning: Equal accuracy across groups
- Fair Value: 0 for both TPR and FPR differences


#### Disparate Impact Ratio

**Description**: Ratio of positive outcomes in actual labels

**Formula**: `P(Y=1|A=a) / P(Y=1|A=b)`

**Interpretation**:
- Legal Threshold: 0.8 (80% rule)
- Fair Value: 1


## Performance Metrics


### Accuracy

**Description**: Proportion of correct predictions
**Formula**: `(TP + TN) / (TP + TN + FP + FN)`
**Range**: [0, 1]
**Best Value**: 1


### Precision

**Description**: Proportion of positive predictions that are correct
**Formula**: `TP / (TP + FP)`
**Range**: [0, 1]
**Best Value**: 1


### Recall

**Description**: Proportion of actual positives correctly identified
**Formula**: `TP / (TP + FN)`
**Range**: [0, 1]
**Best Value**: 1


### F1 Score

**Description**: Harmonic mean of precision and recall
**Formula**: `2 * (precision * recall) / (precision + recall)`
**Range**: [0, 1]
**Best Value**: 1


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
