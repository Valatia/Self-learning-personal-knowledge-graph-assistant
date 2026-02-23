"""
Performance optimization service for REXI - implements caching, monitoring, and optimization.
"""

import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from functools import wraps, lru_cache
import json

from rexi.config.settings import get_settings

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Service for optimizing REXI system performance."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.settings = get_settings()
        
        # Performance metrics
        self.query_times = defaultdict(list)
        self.memory_usage = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=1000)
        self.cache_hit_rates = defaultdict(float)
        self.error_rates = defaultdict(float)
        
        # Caching systems
        self.query_cache = {}
        self.embedding_cache = {}
        self.graph_cache = {}
        self.cache_timestamps = {}
        
        # Optimization parameters
        self.cache_ttl = 3600  # 1 hour
        self.max_cache_size = 10000
        self.query_timeout = 30  # seconds
        self.memory_threshold = 0.8  # 80% memory usage
        self.cpu_threshold = 0.9  # 90% CPU usage
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.performance_alerts = deque(maxlen=100)
        
        # Optimization strategies
        self.optimization_strategies = {
            "query_caching": True,
            "connection_pooling": True,
            "batch_processing": True,
            "memory_management": True,
            "graph_pruning": True
        }
        
        logger.info("Performance optimizer initialized")
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check for performance issues
                self._check_performance_issues()
                
                # Apply optimizations if needed
                self._apply_optimizations()
                
                # Sleep for monitoring interval
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.append({
                "timestamp": datetime.utcnow().isoformat(),
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            })
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.append({
                "timestamp": datetime.utcnow().isoformat(),
                "percent": cpu_percent
            })
            
            # Cache statistics
            self._update_cache_stats()
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _update_cache_stats(self):
        """Update cache statistics."""
        try:
            # Query cache hit rate
            total_queries = sum(len(times) for times in self.query_times.values())
            if total_queries > 0:
                cache_hits = len(self.query_cache)
                self.cache_hit_rates["query_cache"] = cache_hits / total_queries
            
            # Memory cache hit rate
            cache_size = len(self.embedding_cache) + len(self.graph_cache)
            self.cache_hit_rates["memory_cache"] = cache_size / self.max_cache_size
            
        except Exception as e:
            logger.error(f"Failed to update cache stats: {e}")
    
    def _check_performance_issues(self):
        """Check for performance issues and create alerts."""
        try:
            alerts = []
            
            # Check memory usage
            if self.memory_usage:
                latest_memory = self.memory_usage[-1]
                if latest_memory["percent"] > self.memory_threshold * 100:
                    alerts.append({
                        "type": "memory_high",
                        "severity": "warning",
                        "message": f"Memory usage at {latest_memory['percent']:.1f}%",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Check CPU usage
            if self.cpu_usage:
                latest_cpu = self.cpu_usage[-1]
                if latest_cpu["percent"] > self.cpu_threshold * 100:
                    alerts.append({
                        "type": "cpu_high",
                        "severity": "warning",
                        "message": f"CPU usage at {latest_cpu['percent']:.1f}%",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Check query performance
            for query_type, times in self.query_times.items():
                if len(times) >= 10:
                    avg_time = np.mean(times[-10:])
                    if avg_time > 5.0:  # 5 second threshold
                        alerts.append({
                            "type": "query_slow",
                            "severity": "warning",
                            "message": f"{query_type} queries averaging {avg_time:.2f}s",
                            "timestamp": datetime.utcnow().isoformat()
                        })
            
            # Add alerts to queue
            for alert in alerts:
                self.performance_alerts.append(alert)
                logger.warning(f"Performance alert: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Failed to check performance issues: {e}")
    
    def _apply_optimizations(self):
        """Apply performance optimizations based on current metrics."""
        try:
            # Check if optimizations are needed
            if self.memory_usage and self.memory_usage[-1]["percent"] > self.memory_threshold * 100:
                self._optimize_memory()
            
            if self.cache_hit_rates["query_cache"] < 0.3:
                self._optimize_caching()
            
            # Clean expired cache entries
            self._clean_expired_cache()
            
        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")
    
    def _optimize_memory(self):
        """Optimize memory usage."""
        try:
            logger.info("Optimizing memory usage")
            
            # Clean old cache entries
            self._clean_expired_cache()
            
            # Reduce cache size if needed
            total_cache_size = len(self.query_cache) + len(self.embedding_cache) + len(self.graph_cache)
            if total_cache_size > self.max_cache_size * 0.8:
                self._reduce_cache_size()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
    
    def _optimize_caching(self):
        """Optimize caching strategy."""
        try:
            logger.info("Optimizing caching strategy")
            
            # Increase cache TTL for frequently accessed items
            self.cache_ttl = min(7200, self.cache_ttl * 1.2)  # Max 2 hours
            
            # Preload common queries
            self._preload_common_queries()
            
        except Exception as e:
            logger.error(f"Caching optimization failed: {e}")
    
    def _clean_expired_cache(self):
        """Clean expired cache entries."""
        try:
            current_time = time.time()
            
            # Clean query cache
            expired_keys = [
                key for key, timestamp in self.cache_timestamps.items()
                if current_time - timestamp > self.cache_ttl
            ]
            
            for key in expired_keys:
                self.query_cache.pop(key, None)
                self.embedding_cache.pop(key, None)
                self.graph_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
            
            if expired_keys:
                logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            logger.error(f"Cache cleaning failed: {e}")
    
    def _reduce_cache_size(self):
        """Reduce cache size by removing least recently used items."""
        try:
            # Sort cache items by access time
            sorted_items = sorted(
                self.cache_timestamps.items(),
                key=lambda x: x[1]
            )
            
            # Remove oldest 25% of items
            items_to_remove = len(sorted_items) // 4
            for key, _ in sorted_items[:items_to_remove]:
                self.query_cache.pop(key, None)
                self.embedding_cache.pop(key, None)
                self.graph_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
            
            logger.info(f"Reduced cache size by removing {items_to_remove} items")
            
        except Exception as e:
            logger.error(f"Cache size reduction failed: {e}")
    
    def _preload_common_queries(self):
        """Preload common queries into cache."""
        try:
            # This would be implemented based on actual query patterns
            # For now, just log the intent
            logger.info("Preloading common queries into cache")
            
        except Exception as e:
            logger.error(f"Query preloading failed: {e}")
    
    def cache_query_result(self, query_key: str, result: Any, ttl: Optional[int] = None):
        """Cache a query result."""
        try:
            if len(self.query_cache) >= self.max_cache_size:
                self._reduce_cache_size()
            
            self.query_cache[query_key] = result
            self.cache_timestamps[query_key] = time.time()
            
            if ttl:
                # Store custom TTL
                self.cache_timestamps[f"{query_key}_ttl"] = ttl
            
        except Exception as e:
            logger.error(f"Failed to cache query result: {e}")
    
    def get_cached_result(self, query_key: str) -> Optional[Any]:
        """Get cached query result."""
        try:
            # Check if result exists and is not expired
            if query_key in self.query_cache:
                timestamp = self.cache_timestamps.get(query_key, 0)
                custom_ttl = self.cache_timestamps.get(f"{query_key}_ttl", self.cache_ttl)
                
                if time.time() - timestamp < custom_ttl:
                    return self.query_cache[query_key]
                else:
                    # Remove expired entry
                    self.query_cache.pop(query_key, None)
                    self.cache_timestamps.pop(query_key, None)
                    self.cache_timestamps.pop(f"{query_key}_ttl", None)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached result: {e}")
            return None
    
    def record_query_time(self, query_type: str, duration: float):
        """Record query execution time."""
        try:
            self.query_times[query_type].append(duration)
            
            # Keep only last 100 measurements
            if len(self.query_times[query_type]) > 100:
                self.query_times[query_type] = self.query_times[query_type][-100:]
            
        except Exception as e:
            logger.error(f"Failed to record query time: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        try:
            stats = {
                "system_metrics": {
                    "memory": list(self.memory_usage)[-10:] if self.memory_usage else [],
                    "cpu": list(self.cpu_usage)[-10:] if self.cpu_usage else []
                },
                "query_performance": {},
                "cache_performance": dict(self.cache_hit_rates),
                "alerts": list(self.performance_alerts)[-10:] if self.performance_alerts else [],
                "optimization_status": dict(self.optimization_strategies)
            }
            
            # Calculate query performance stats
            for query_type, times in self.query_times.items():
                if times:
                    stats["query_performance"][query_type] = {
                        "avg_time": np.mean(times),
                        "min_time": min(times),
                        "max_time": max(times),
                        "count": len(times)
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {}
    
    def optimize_query(self, query: str, query_type: str) -> Dict:
        """Optimize a specific query."""
        try:
            optimization_result = {
                "original_query": query,
                "query_type": query_type,
                "optimizations_applied": [],
                "estimated_improvement": 0.0
            }
            
            # Check if query is cached
            cache_key = f"{query_type}:{hash(query)}"
            cached_result = self.get_cached_result(cache_key)
            
            if cached_result:
                optimization_result["optimizations_applied"].append("cache_hit")
                optimization_result["estimated_improvement"] = 0.9  # 90% improvement
                return optimization_result
            
            # Apply query optimizations
            optimized_query = self._apply_query_optimizations(query, query_type)
            
            if optimized_query != query:
                optimization_result["optimized_query"] = optimized_query
                optimization_result["optimizations_applied"].append("query_rewrite")
                optimization_result["estimated_improvement"] = 0.2  # 20% improvement
            
            # Cache the optimization result
            self.cache_query_result(cache_key, optimization_result)
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return {
                "original_query": query,
                "query_type": query_type,
                "error": str(e)
            }
    
    def _apply_query_optimizations(self, query: str, query_type: str) -> str:
        """Apply specific optimizations to a query."""
        try:
            # This would contain actual query optimization logic
            # For now, return the original query
            return query
            
        except Exception as e:
            logger.error(f"Query optimization application failed: {e}")
            return query
    
    def performance_decorator(self, query_type: str):
        """Decorator for performance monitoring of functions."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.record_query_time(query_type, duration)
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    self.record_query_time(f"{query_type}_error", duration)
                    raise
            
            return wrapper
        return decorator
    
    def get_optimization_recommendations(self) -> List[Dict]:
        """Get performance optimization recommendations."""
        try:
            recommendations = []
            
            # Analyze query performance
            for query_type, times in self.query_times.items():
                if len(times) >= 10:
                    avg_time = np.mean(times[-10:])
                    
                    if avg_time > 5.0:
                        recommendations.append({
                            "type": "query_optimization",
                            "priority": "high",
                            "target": query_type,
                            "description": f"{query_type} queries are slow (avg: {avg_time:.2f}s)",
                            "suggestion": "Consider adding indexes or optimizing query structure"
                        })
            
            # Analyze cache performance
            if self.cache_hit_rates["query_cache"] < 0.3:
                recommendations.append({
                    "type": "cache_optimization",
                    "priority": "medium",
                    "target": "query_cache",
                    "description": f"Low cache hit rate: {self.cache_hit_rates['query_cache']:.2f}",
                    "suggestion": "Increase cache size or adjust caching strategy"
                })
            
            # Analyze memory usage
            if self.memory_usage and self.memory_usage[-1]["percent"] > 80:
                recommendations.append({
                    "type": "memory_optimization",
                    "priority": "high",
                    "target": "memory",
                    "description": f"High memory usage: {self.memory_usage[-1]['percent']:.1f}%",
                    "suggestion": "Consider implementing memory management or graph pruning"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {e}")
            return []
    
    def apply_optimization_recommendations(self, recommendations: List[Dict]) -> Dict:
        """Apply optimization recommendations."""
        try:
            applied = []
            
            for rec in recommendations:
                rec_type = rec["type"]
                
                if rec_type == "query_optimization":
                    # This would trigger query optimization
                    applied.append({
                        "recommendation": rec,
                        "status": "scheduled",
                        "message": "Query optimization scheduled"
                    })
                
                elif rec_type == "cache_optimization":
                    self._optimize_caching()
                    applied.append({
                        "recommendation": rec,
                        "status": "applied",
                        "message": "Cache optimization applied"
                    })
                
                elif rec_type == "memory_optimization":
                    self._optimize_memory()
                    applied.append({
                        "recommendation": rec,
                        "status": "applied",
                        "message": "Memory optimization applied"
                    })
            
            return {
                "applied_count": len(applied),
                "applied_optimizations": applied
            }
            
        except Exception as e:
            logger.error(f"Failed to apply optimization recommendations: {e}")
            return {"error": str(e)}
