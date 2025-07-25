# Fair-Mango Caching Implementation

This document describes the comprehensive caching system implemented in Fair-Mango to optimize performance metrics computation across fairness metrics.

## Overview

The caching system addresses three key optimization areas:

1. **Shared Performance Metric Computations** - Avoids redundant calculations when multiple fairness metrics rely on the same performance metrics
2. **Dataset-Level Caching** - Implements caching in superset classes to reuse computations across sensitive attribute combinations
3. **Expensive Operations Caching** - Caches costly operations like confusion matrix calculations that multiple fairness metrics depend on

## Architecture

### Global Cache Manager (`cache.py`)

The `GlobalMetricCache` class provides a singleton cache that can be shared across all metric instances:

```python
from fair_mango.metrics.cache import global_cache

# The cache is automatically shared across all metric computations
result = global_cache.compute_cached(func, "cache_prefix", *args, **kwargs)
```

**Key Features:**
- Singleton pattern ensures one global cache instance
- Automatic cache key generation using MD5 hashing
- Support for complex data structures in cache keys
- Thread-safe operations

### Mixin Classes (`mixins.py`)

#### `CacheableMixin`
Provides caching functionality for individual metric classes:

```python
class MyMetric(Metric, CacheableMixin):
    def compute_something(self, data):
        return self.compute_cached_performance_metric(
            "my_metric",
            some_expensive_function,
            real_values,
            predicted_values
        )
```

#### `DatasetCacheableMixin`
Provides dataset-level caching for superset classes:

```python
class MySuperset(Superset, DatasetCacheableMixin):
    def process_datasets(self):
        for dataset in self.datasets:
            dataset_id = self.get_dataset_cache_id(dataset)
            result = self.compute_cached_dataset_metric(
                dataset_id,
                "metric_name",
                compute_function
            )
```

## Implementation Details

### Updated Metric Classes

#### `PerformanceMetric` and `ConfusionMatrix`
- Now inherit from `CacheableMixin`
- Use global cache 
- Share computations across different metric instances

```python

# Using global shared cache
class PerformanceMetric(Metric, CacheableMixin):
    def _compute_cached_metric(self, metric_name, metric_func, real_values, predicted_values):
        return self.compute_cached_performance_metric(
            metric_name, metric_func, real_values, predicted_values
        )
```

#### `SupersetFairnessMetrics` and `SupersetPerformanceMetrics`
- Now inherit from `DatasetCacheableMixin`
- Cache metric computations at the dataset level
- Avoid recomputing metrics for identical datasets

```python
# Before: Computed metrics independently for each dataset
for pair in self.pairs:
    dataset = Dataset(...)
    metric = MetricClass(dataset)
    result = metric.rank()

# After: Uses dataset-level caching
for pair in self.pairs:
    dataset = Dataset(...)
    dataset_id = self.get_dataset_cache_id(dataset)
    
    def compute_metric():
        metric = MetricClass(dataset)
        return metric.rank()
    
    result = self.compute_cached_dataset_metric(
        dataset_id, "metric_name_rank", compute_metric
    )
```

### Cache Key Generation

The system uses sophisticated cache key generation:

1. **Function-based keys**: Include function name and all arguments
2. **Dataset-based keys**: Hash dataset characteristics (sensitive attributes, targets, data shape)
3. **Collision-resistant**: Uses MD5 hashing with pickle serialization
4. **Type-aware**: Handles iterables, primitives, and complex objects

## Performance Benefits

### Before Implementation
- Each fairness metric computed performance metrics independently
- Superset classes recomputed metrics for each sensitive attribute combination
- No sharing of expensive operations like confusion matrix calculations

### After Implementation
- **Shared Computations**: Multiple fairness metrics reuse the same performance metric calculations
- **Dataset Caching**: Identical datasets in superset operations use cached results
- **Global Optimization**: Expensive operations are computed once and reused across the entire application

## Usage Examples

### Basic Usage (Automatic)
```python
from fair_mango.metrics import PerformanceMetric, ConfusionMatrix
from fair_mango.dataset import Dataset

# Caching happens automatically
perf_metric = PerformanceMetric(dataset)
conf_matrix = ConfusionMatrix(dataset)

# If both use the same underlying data, computations are shared
result1 = perf_metric()
result2 = conf_matrix()  # Reuses cached confusion matrix calculations
```

### Superset Usage (Automatic)
```python
from fair_mango.metrics import SupersetFairnessMetrics

superset = SupersetFairnessMetrics(dataset)

# Dataset-level caching automatically optimizes computations
rankings = superset.rank()
summaries = superset.summary()  # Reuses cached metric computations
```

### Manual Cache Control
```python
from fair_mango.metrics.cache import global_cache

# Clear cache if needed
global_cache.clear()

# Check cache status
print(f"Cache has {len(global_cache._cache)} entries")
```

## Cache Management

### Memory Considerations
- Cache grows with unique computations
- Consider clearing cache for long-running applications
- Monitor memory usage in production environments

### Cache Invalidation
- Cache persists for the application lifetime
- Manual clearing available via `global_cache.clear()`
- No automatic expiration (by design for performance)

## Migration Guide

### For Existing Code
No changes required! The caching system is backward-compatible:

```python
# This code continues to work unchanged
metric = PerformanceMetric(dataset)
result = metric()

# But now benefits from automatic caching
```

### For Custom Metrics
To add caching to custom metrics:

```python
# Before
class CustomMetric(Metric):
    def compute(self):
        return expensive_computation()

# After
class CustomMetric(Metric, CacheableMixin):
    def compute(self):
        return self.compute_cached_metric(
            "custom_computation",
            expensive_computation,
            self.data,
            self.parameters
        )
```

## Testing

The caching system includes comprehensive tests:

```bash
# Run cache-specific tests
python -m pytest tests/metrics/test_cache.py

# Run integration tests
python -m pytest tests/metrics/test_metrics.py
python -m pytest tests/metrics/test_superset.py
```

## Future Enhancements

1. **Persistent Caching**: Save cache to disk for cross-session persistence
2. **Cache Size Limits**: Implement LRU eviction for memory management
3. **Cache Analytics**: Provide cache hit/miss statistics
4. **Distributed Caching**: Support for multi-process environments

## Troubleshooting

### Common Issues

1. **Memory Usage**: If memory usage is high, clear the cache periodically
2. **Stale Results**: Cache persists across runs; clear if data changes
3. **Import Errors**: Ensure all new modules are properly imported

### Debug Mode
```python
# Enable debug logging to see cache operations
import logging
logging.basicConfig(level=logging.DEBUG)

# Check cache contents
from fair_mango.metrics.cache import global_cache
print(f"Cache keys: {list(global_cache._cache.keys())}")
```