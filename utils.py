import functools
import logging
import time
import asyncio
from typing import Callable, Dict, List, Any, TypeVar, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VendorResearchAgent")

# Type definitions
T = TypeVar('T')
R = TypeVar('R')

# Caching decorator
def memoize(func: Callable) -> Callable:
    """Cache results of function calls to avoid redundant LLM requests."""
    cache = {}
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            # For async functions, we need to await the result
            cache[key] = await func(*args, **kwargs)
        return cache[key]
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    # Determine if the decorated function is async or sync
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

# Batching helpers
def create_vendor_batches(count: int, mix: Dict[str, int]) -> List[Tuple[str, int]]:
    """
    Create batches for vendor generation based on count and requested mix.
    
    Args:
        count: Total number of vendors to generate
        mix: Dictionary with business type as key and percentage as value
             e.g. {'manufacturer': 40, 'distributor': 30, 'retailer': 30}
    
    Returns:
        List of tuples with (business_type, batch_count)
    """
    batches = []
    remaining = count
    
    # Calculate how many vendors of each type to generate
    for business_type, percentage in mix.items():
        type_count = int(count * percentage / 100)
        if type_count > 0:
            # Determine number of batches (10-20 vendors per batch)
            batch_size = min(20, max(10, type_count // 3))
            num_full_batches = type_count // batch_size
            
            # Create full batches
            for _ in range(num_full_batches):
                batches.append((business_type, batch_size))
                remaining -= batch_size
            
            # Handle remaining items
            if type_count % batch_size > 0:
                batches.append((business_type, type_count % batch_size))
                remaining -= type_count % batch_size
    
    # Handle any remaining vendors due to rounding
    if remaining > 0:
        # Add remaining to the largest batch type
        largest_type = max(mix.items(), key=lambda x: x[1])[0]
        batches.append((largest_type, remaining))
    
    return batches

# Error handling utility
def retry_with_backoff(max_retries: int = 3, initial_backoff: float = 1.0):
    """
    Decorator for retrying function calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            backoff = initial_backoff
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {backoff:.2f}s")
                    # Use asyncio.sleep for async functions
                    await asyncio.sleep(backoff)
                    backoff *= 2  # Exponential backoff
                    
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            backoff = initial_backoff
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {backoff:.2f}s")
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
        
        # Determine if the decorated function is async or sync
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

# CSS for styling
def get_css() -> str:
    """Returns CSS for styling the Streamlit app."""
    return """
    <style>
        .vendor-card {
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease-in-out;
            background-color: white;
        }
        
        .vendor-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        
        .vendor-name {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .vendor-score {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.9rem;
            margin-left: 0.5rem;
        }
        
        .score-high {
            background-color: #d4edda;
            color: #155724;
        }
        
        .score-medium {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .score-low {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        .vendor-info {
            margin-top: 0.5rem;
            color: #6c757d;
        }
        
        .vendor-description {
            margin: 0.7rem 0;
        }
        
        .vendor-website {
            color: #007bff;
            text-decoration: none;
        }
        
        .vendor-contact {
            margin: 0.5rem 0;
            color: #495057;
        }
        
        .vendor-specialization {
            display: inline-block;
            background-color: #e2e3e5;
            color: #383d41;
            padding: 0.2rem 0.5rem;
            margin: 0.2rem;
            border-radius: 15px;
            font-size: 0.8rem;
        }
        
        .skeleton-loader {
            animation: pulse 1.5s infinite;
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            height: 1rem;
            margin-bottom: 0.5rem;
            border-radius: 4px;
        }
        
        @keyframes pulse {
            0% {
                background-position: 0% 0%;
            }
            100% {
                background-position: -200% 0%;
            }
        }
    </style>
    """

# Skeleton loader HTML
def get_skeleton_card_html() -> str:
    """Returns HTML for a skeleton loader card."""
    return """
    <div class="vendor-card">
        <div class="skeleton-loader" style="height: 1.5rem; width: 60%;"></div>
        <div class="skeleton-loader" style="height: 5rem;"></div>
        <div class="skeleton-loader" style="width: 40%;"></div>
        <div class="skeleton-loader" style="width: 70%;"></div>
        <div style="display: flex; gap: 0.5rem;">
            <div class="skeleton-loader" style="width: 25%;"></div>
            <div class="skeleton-loader" style="width: 25%;"></div>
            <div class="skeleton-loader" style="width: 25%;"></div>
        </div>
    </div>
    """ 