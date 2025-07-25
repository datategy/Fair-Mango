"""Global caching system for Fair-Mango metrics.

This module provides a centralized caching mechanism to avoid redundant
computations across different fairness and performance metrics.
"""

from typing import Any, Hashable
import hashlib
import pickle
from collections.abc import Callable


class GlobalMetricCache:
    """Global cache for metric computations.
    
    This singleton class provides a centralized cache that can be shared
    across all metric instances to avoid redundant computations.
    """
    
    _instance = None
    _cache: dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments.
        
        Parameters
        ----------
        *args : Any
            Positional arguments
        **kwargs : Any
            Keyword arguments
            
        Returns
        -------
        str
            Unique cache key
        """
        key_data = []
        
        for arg in args:
            if hasattr(arg, '__iter__') and not isinstance(arg, (str, bytes)):
                try:
                    key_data.append(tuple(arg))
                except TypeError:
                    key_data.append((str(arg),))
            else:
                key_data.append((arg,))
        
        for k, v in sorted(kwargs.items()):
            if hasattr(v, '__iter__') and not isinstance(v, (str, bytes)):
                try:
                    key_data.append((k, tuple(v)))
                except TypeError:
                    key_data.append((k, (str(v),)))
            else:
                key_data.append((k, (v,)))
        
        key_str = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(key_str).hexdigest()
    
    def get(self, cache_key: str) -> Any:
        """Get value from cache.
        
        Parameters
        ----------
        cache_key : str
            Cache key
            
        Returns
        -------
        Any
            Cached value or None if not found
        """
        return self._cache.get(cache_key)
    
    def set(self, cache_key: str, value: Any) -> None:
        """Set value in cache.
        
        Parameters
        ----------
        cache_key : str
            Cache key
        value : Any
            Value to cache
        """
        self._cache[cache_key] = value
    
    def has(self, cache_key: str) -> bool:
        """Check if key exists in cache.
        
        Parameters
        ----------
        cache_key : str
            Cache key
            
        Returns
        -------
        bool
            True if key exists
        """
        return cache_key in self._cache
    
    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
    
    def compute_cached(
        self,
        func: Callable,
        cache_prefix: str,
        *args,
        **kwargs
    ) -> Any:
        """Compute a function with caching.
        
        Parameters
        ----------
        func : Callable
            Function to compute
        cache_prefix : str
            Prefix for cache key (e.g., 'confusion_matrix', 'performance_metric')
        *args : Any
            Function arguments
        **kwargs : Any
            Function keyword arguments
            
        Returns
        -------
        Any
            Function result (cached or computed)
        """
        cache_key = f"{cache_prefix}_{self._generate_cache_key(func.__name__, *args, **kwargs)}"
        
        if self.has(cache_key):
            return self.get(cache_key)
        
        result = func(*args, **kwargs)
        self.set(cache_key, result)
        return result


global_cache = GlobalMetricCache()