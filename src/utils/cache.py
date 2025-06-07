"""
Caching Layer with Redis Integration
Provides distributed caching for API responses, ML models, and computational results
"""

import json
import pickle
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import redis
import hashlib
import logging
import os
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


class CacheManager:
    """Redis-based cache manager for the Digital Twin system"""
    
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_client = redis.from_url(redis_url, decode_responses=False)
        self.default_ttl = 3600  # 1 hour default TTL
        
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate a cache key from prefix and parameters"""
        key_data = json.dumps(kwargs, sort_keys=True)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value in cache with optional TTL"""
        try:
            ttl = ttl or self.default_ttl
            
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = pickle.dumps(value)
            
            return self.redis_client.setex(key, ttl, serialized_value)
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from cache"""
        try:
            cached_data = self.redis_client.get(key)
            if cached_data is None:
                return None
                
            # Try JSON first, then pickle
            try:
                return json.loads(cached_data)
            except (json.JSONDecodeError, TypeError):
                return pickle.loads(cached_data)
                
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Failed to clear pattern {pattern}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = self.redis_client.info()
            return {
                "used_memory": info.get("used_memory_human"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}


# Global cache manager instance
cache_manager = CacheManager()


class ForecastCache:
    """Specialized cache for forecasting results"""
    
    @staticmethod
    def get_forecast_key(community_id: str, start_time: datetime, 
                        end_time: datetime, model_type: str) -> str:
        """Generate cache key for forecast results"""
        return cache_manager._generate_key(
            "forecast",
            community_id=community_id,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            model_type=model_type
        )
    
    @staticmethod
    def cache_forecast(community_id: str, start_time: datetime, 
                      end_time: datetime, model_type: str, 
                      forecast_data: Dict[str, Any], ttl: int = 1800) -> bool:
        """Cache forecast results (30 min default TTL)"""
        key = ForecastCache.get_forecast_key(
            community_id, start_time, end_time, model_type
        )
        return cache_manager.set(key, forecast_data, ttl)
    
    @staticmethod
    def get_cached_forecast(community_id: str, start_time: datetime, 
                           end_time: datetime, model_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached forecast results"""
        key = ForecastCache.get_forecast_key(
            community_id, start_time, end_time, model_type
        )
        return cache_manager.get(key)


class OptimizationCache:
    """Specialized cache for optimization results"""
    
    @staticmethod
    def get_optimization_key(community_id: str, objective: str, 
                           horizon_hours: int, current_state: Dict[str, Any]) -> str:
        """Generate cache key for optimization results"""
        return cache_manager._generate_key(
            "optimization",
            community_id=community_id,
            objective=objective,
            horizon_hours=horizon_hours,
            current_state_hash=hashlib.md5(
                json.dumps(current_state, sort_keys=True).encode()
            ).hexdigest()[:16]
        )
    
    @staticmethod
    def cache_optimization(community_id: str, objective: str, 
                          horizon_hours: int, current_state: Dict[str, Any],
                          optimization_result: Dict[str, Any], ttl: int = 900) -> bool:
        """Cache optimization results (15 min default TTL)"""
        key = OptimizationCache.get_optimization_key(
            community_id, objective, horizon_hours, current_state
        )
        return cache_manager.set(key, optimization_result, ttl)
    
    @staticmethod
    def get_cached_optimization(community_id: str, objective: str, 
                               horizon_hours: int, current_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve cached optimization results"""
        key = OptimizationCache.get_optimization_key(
            community_id, objective, horizon_hours, current_state
        )
        return cache_manager.get(key)


class MacroTwinCache:
    """Specialized cache for Macro Twin state predictions"""
    
    @staticmethod
    def get_state_key(community_id: str, current_time: datetime) -> str:
        """Generate cache key for state predictions"""
        # Round to nearest 15 minutes for better cache hits
        rounded_time = current_time.replace(
            minute=(current_time.minute // 15) * 15, 
            second=0, 
            microsecond=0
        )
        return cache_manager._generate_key(
            "macro_twin_state",
            community_id=community_id,
            time=rounded_time.isoformat()
        )
    
    @staticmethod
    def cache_state_prediction(community_id: str, current_time: datetime,
                              prediction_data: Dict[str, Any], ttl: int = 300) -> bool:
        """Cache state predictions (5 min default TTL)"""
        key = MacroTwinCache.get_state_key(community_id, current_time)
        return cache_manager.set(key, prediction_data, ttl)
    
    @staticmethod
    def get_cached_state_prediction(community_id: str, current_time: datetime) -> Optional[Dict[str, Any]]:
        """Retrieve cached state predictions"""
        key = MacroTwinCache.get_state_key(community_id, current_time)
        return cache_manager.get(key)


def cached_api_response(ttl: int = 300, cache_key_func: Optional[callable] = None):
    """Decorator for caching API responses"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default key generation from function name and args
                key_data = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': {k: str(v) for k, v in kwargs.items() if k != 'db'}
                }
                cache_key = cache_manager._generate_key("api_response", **key_data)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            # Cache the result
            cache_manager.set(cache_key, result.dict() if hasattr(result, 'dict') else result, ttl)
            logger.info(f"Cached result for {func.__name__}")
            
            return result
        return wrapper
    return decorator


class CacheWarmup:
    """Cache warming utilities for improved performance"""
    
    @staticmethod
    async def warm_forecast_cache(community_ids: List[str], hours_ahead: int = 24):
        """Pre-populate forecast cache for multiple communities"""
        logger.info(f"Warming forecast cache for {len(community_ids)} communities")
        
        from ..forecasting.models import ForecastingService
        
        current_time = datetime.utcnow()
        end_time = current_time + timedelta(hours=hours_ahead)
        
        for community_id in community_ids:
            try:
                forecasting_service = ForecastingService(community_id)
                
                # Warm cache for different model types
                for model_type in ['lstm', 'prophet']:
                    forecast_result = await forecasting_service.generate_forecast(
                        start_time=current_time,
                        end_time=end_time,
                        model_type=model_type
                    )
                    
                    ForecastCache.cache_forecast(
                        community_id, current_time, end_time, 
                        model_type, forecast_result
                    )
                    
                logger.info(f"Warmed forecast cache for community {community_id}")
                
            except Exception as e:
                logger.error(f"Failed to warm forecast cache for {community_id}: {e}")
    
    @staticmethod
    async def warm_optimization_cache(community_ids: List[str]):
        """Pre-populate optimization cache for common scenarios"""
        logger.info(f"Warming optimization cache for {len(community_ids)} communities")
        
        from ..optimization.engine import OptimizationEngine
        from ..data.models import OptimizationRequest, OptimizationObjective, TariffStructure
        
        for community_id in community_ids:
            try:
                # Create sample optimization requests for common scenarios
                sample_requests = [
                    OptimizationRequest(
                        objective=OptimizationObjective.MINIMIZE_COST,
                        horizon_hours=24,
                        current_soc_kwh=50.0,
                        forecasted_load=[100.0] * 96,  # Sample data
                        forecasted_pv=[80.0] * 96,     # Sample data
                        current_tariff=TariffStructure(
                            time_of_use_rates={"peak": 0.3, "off_peak": 0.15},
                            feed_in_tariff=0.08,
                            demand_charge=10.0,
                            fixed_charge_daily=5.0
                        )
                    )
                ]
                
                # Run optimizations and cache results
                for request in sample_requests:
                    optimization_engine = OptimizationEngine()
                    result = optimization_engine.optimize_dispatch_and_tariff(request)
                    
                    OptimizationCache.cache_optimization(
                        community_id, 
                        request.objective.value,
                        request.horizon_hours,
                        {"soc": request.current_soc_kwh},
                        result.dict()
                    )
                    
                logger.info(f"Warmed optimization cache for community {community_id}")
                
            except Exception as e:
                logger.error(f"Failed to warm optimization cache for {community_id}: {e}")


class CacheMonitor:
    """Monitor cache performance and health"""
    
    @staticmethod
    def get_cache_metrics() -> Dict[str, Any]:
        """Get comprehensive cache metrics"""
        stats = cache_manager.get_stats()
        
        # Calculate hit rate
        hits = stats.get('keyspace_hits', 0)
        misses = stats.get('keyspace_misses', 0)
        total_requests = hits + misses
        
        hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "redis_stats": stats,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
            "cache_efficiency": "excellent" if hit_rate > 80 else "good" if hit_rate > 60 else "needs_improvement"
        }
    
    @staticmethod
    def clear_expired_cache():
        """Clear expired cache entries (Redis handles this automatically, but we can force it)"""
        try:
            # Get all keys and check TTL
            keys = cache_manager.redis_client.keys("*")
            expired_count = 0
            
            for key in keys:
                ttl = cache_manager.redis_client.ttl(key)
                if ttl == -2:  # Key doesn't exist (expired)
                    expired_count += 1
            
            logger.info(f"Found {expired_count} expired keys")
            return expired_count
            
        except Exception as e:
            logger.error(f"Failed to check expired cache: {e}")
            return 0


# Export main components
__all__ = [
    'CacheManager',
    'cache_manager',
    'ForecastCache',
    'OptimizationCache',
    'MacroTwinCache',
    'cached_api_response',
    'CacheWarmup',
    'CacheMonitor'
]
