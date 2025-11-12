"""
Logging decorators for application components
"""
import time
from functools import wraps
from typing import Callable, Any
from logging import getLogger


def log_method_calls(func: Callable) -> Callable:
    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        logger = getLogger(func.__module__)
        method_name = f"{func.__qualname__}"
        
        start_time = time.time()
        logger.info(f"Starting {method_name} with args: {args[1:] if args else []}, kwargs: {kwargs}")
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {method_name} in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {method_name} after {execution_time:.3f}s: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        logger = getLogger(func.__module__)
        method_name = f"{func.__qualname__}"
        
        start_time = time.time()
        logger.info(f"Starting {method_name} with args: {args[1:] if args else []}, kwargs: {kwargs}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {method_name} in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {method_name} after {execution_time:.3f}s: {e}")
            raise
    
    if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_ITERABLE_COROUTINE
        return async_wrapper
    else:
        return sync_wrapper


def log_grpc_calls(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, request, context):
        logger = getLogger(func.__module__)
        method_name = f"{func.__qualname__}"
        
        start_time = time.time()
        logger.info(f"gRPC call: {method_name}, request: {request}")
        
        try:
            result = func(self, request, context)
            execution_time = time.time() - start_time
            logger.info(f"gRPC response: {method_name} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"gRPC error in {method_name} after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper
