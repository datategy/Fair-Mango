"""Fair-Mango metrics package.

This package provides fairness and performance metrics with global caching capabilities.
"""

from fair_mango.metrics.cache import GlobalMetricCache, global_cache
from fair_mango.metrics.mixins import CacheableMixin, DatasetCacheableMixin
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
)
from fair_mango.metrics.superset import (
    SupersetFairnessMetrics,
    SupersetPerformanceMetrics,
)

__all__ = [
    "GlobalMetricCache",
    "global_cache",
    "CacheableMixin",
    "DatasetCacheableMixin",
    "ConfusionMatrix",
    "DemographicParityDifference",
    "DemographicParityRatio",
    "DisparateImpactDifference",
    "DisparateImpactRatio",
    "EqualisedOddsDifference",
    "EqualisedOddsRatio",
    "EqualOpportunityDifference",
    "EqualOpportunityRatio",
    "FalsePositiveRateDifference",
    "FalsePositiveRateRatio",
    "PerformanceMetric",
    "SelectionRate",
    "SupersetFairnessMetrics",
    "SupersetPerformanceMetrics",
]
