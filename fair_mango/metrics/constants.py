"""Constants for fairness metrics."""

DEFAULT_BIAS_THRESHOLDS = {
    "demographic_parity_difference": 0.1,
    "disparate_impact_difference": 0.1,
    "equal_opportunity_difference": 0.1,
    "false_positive_rate_difference": 0.1,
    "equalised_odds_difference": 0.1,
    "demographic_parity_ratio": 0.8,
    "disparate_impact_ratio": 0.8,
    "equal_opportunity_ratio": 0.8,
    "false_positive_rate_ratio": 0.8,
    "equalised_odds_ratio": 0.8,
}
