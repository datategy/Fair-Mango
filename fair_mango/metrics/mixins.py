"""Mixin classes for Fair-Mango metrics.

This module provides mixin classes that add caching functionality
to metric classes.
"""

from typing import Any, Callable
from fair_mango.metrics.cache import global_cache


class CacheableMixin:
    """Mixin class that provides caching functionality to metric classes.
    
    This mixin allows metric classes to use the global cache for
    expensive computations like confusion matrices and performance metrics.
    """
    
    def compute_cached_metric(
        self,
        metric_name: str,
        metric_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Compute a metric with global caching.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric for cache prefixing
        metric_func : Callable
            The metric function to compute
        *args : Any
            Function arguments
        **kwargs : Any
            Function keyword arguments
            
        Returns
        -------
        Any
            The computed metric value
        """
        return global_cache.compute_cached(
            metric_func,
            f"{self.__class__.__name__}_{metric_name}",
            *args,
            **kwargs
        )
    
    def compute_cached_confusion_matrix(
        self,
        real_values: Any,
        predicted_values: Any,
        **kwargs
    ) -> Any:
        """Compute confusion matrix with global caching.
        
        Parameters
        ----------
        real_values : Any
            Real target values
        predicted_values : Any
            Predicted target values
        **kwargs : Any
            Additional arguments for confusion matrix computation
            
        Returns
        -------
        Any
            Confusion matrix result
        """
        from sklearn.metrics import confusion_matrix
        
        return global_cache.compute_cached(
            confusion_matrix,
            "confusion_matrix",
            real_values,
            predicted_values,
            **kwargs
        )
    
    def compute_cached_performance_metric(
        self,
        metric_name: str,
        metric_func: Callable,
        real_values: Any,
        predicted_values: Any,
        **kwargs
    ) -> Any:
        """Compute performance metric with global caching.
        
        Parameters
        ----------
        metric_name : str
            Name of the performance metric
        metric_func : Callable
            The performance metric function
        real_values : Any
            Real target values
        predicted_values : Any
            Predicted target values
        **kwargs : Any
            Additional arguments for the metric function
            
        Returns
        -------
        Any
            Performance metric result
        """
        return global_cache.compute_cached(
            metric_func,
            f"performance_{metric_name}",
            real_values,
            predicted_values,
            **kwargs
        )


class DatasetCacheableMixin:
    """Mixin class that provides dataset-level caching functionality.
    
    This mixin is designed for superset classes that work with
    multiple datasets and need to cache computations at the dataset level.
    """
    
    def compute_cached_dataset_metric(
        self,
        dataset_id: str,
        metric_name: str,
        metric_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Compute a metric with dataset-level caching.
        
        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset
        metric_name : str
            Name of the metric
        metric_func : Callable
            The metric function to compute
        *args : Any
            Function arguments
        **kwargs : Any
            Function keyword arguments
            
        Returns
        -------
        Any
            The computed metric value
        """
        return global_cache.compute_cached(
            metric_func,
            f"dataset_{dataset_id}_{metric_name}",
            *args,
            **kwargs
        )
    
    def get_dataset_cache_id(self, dataset) -> str:
        """Generate a unique cache ID for a dataset.
        
        Parameters
        ----------
        dataset : Dataset
            The dataset object
            
        Returns
        -------
        str
            Unique dataset identifier for caching
        """
        import hashlib
        import pickle
        
        key_data = [
            tuple(dataset.sensitive) if hasattr(dataset, 'sensitive') and dataset.sensitive is not None else None,
            tuple(dataset.real_target) if hasattr(dataset, 'real_target') and dataset.real_target is not None else None,
            tuple(dataset.predicted_target) if hasattr(dataset, 'predicted_target') and dataset.predicted_target is not None else None,
            tuple(dataset.positive_target) if hasattr(dataset, 'positive_target') and dataset.positive_target is not None else None,
        ]
        
        if hasattr(dataset, 'data') and hasattr(dataset.data, 'shape'):
            key_data.append(dataset.data.shape)
        
        key_str = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(key_str).hexdigest()[:16]  # Use first 16 chars for readability