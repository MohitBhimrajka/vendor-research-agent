import asyncio
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
import time
import random

from llm_service import LLMService
from utils import create_vendor_batches, logger

@dataclass
class VendorDetail:
    """Class for storing vendor details."""
    name: str
    description: str
    website: str
    contact: str
    specializations: List[str]
    relevance_score: int
    business_type: str
    simplified_type: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VendorDetail':
        """Create a VendorDetail from a dictionary."""
        vendor = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            website=data.get("website", ""),
            contact=data.get("contact", ""),
            specializations=data.get("specializations", []),
            relevance_score=data.get("relevance_score", 0),
            business_type=data.get("business_type", "")
        )
        
        # Add simplified type
        if vendor.business_type:
            if 'manufacturer' in vendor.business_type.lower():
                vendor.simplified_type = 'Manufacturer'
            elif 'distributor' in vendor.business_type.lower() or 'supplier' in vendor.business_type.lower():
                vendor.simplified_type = 'Distributor/Supplier'
            elif 'retailer' in vendor.business_type.lower() or 'seller' in vendor.business_type.lower():
                vendor.simplified_type = 'Retailer'
            else:
                vendor.simplified_type = vendor.business_type
        
        return vendor

class VendorManager:
    """Class for managing vendor discovery and research."""
    
    def __init__(self, llm_service: LLMService):
        """
        Initialize the vendor manager.
        
        Args:
            llm_service: An instance of LLMService
        """
        self.llm_service = llm_service
        self.discovered_vendors: Set[str] = set()
        
    async def find_vendors(self, term: str, count: int, mix: Dict[str, int], 
                            country: str = None, region: str = None) -> List[str]:
        """
        Find vendor names based on a term, count, and business type mix.
        
        Args:
            term: Search term or category
            count: Total number of vendors to find
            mix: Dictionary with business type as key and percentage as value
                 e.g. {'manufacturer': 40, 'distributor': 30, 'retailer': 30}
            country: Optional country to focus on
            region: Optional region/state within the country
                 
        Returns:
            List of unique vendor names
        """
        # Create batches based on the requested mix
        batches = create_vendor_batches(count, mix)
        logger.info(f"Created {len(batches)} batches for finding {count} vendors")
        
        all_vendors = []
        
        # Prepare all batch search tasks
        async def process_batch(business_type: str, batch_count: int) -> List[str]:
            try:
                batch_vendors = await self.llm_service.find_vendor_names(
                    term, batch_count, business_type, country, region
                )
                
                # Filter out any vendors that have already been discovered
                new_vendors = [v for v in batch_vendors if v not in self.discovered_vendors]
                
                # If we didn't get enough vendors, try again for the remaining
                if len(new_vendors) < batch_count:
                    remaining = batch_count - len(new_vendors)
                    logger.info(f"Need {remaining} more vendors for {business_type} batch (duplicates filtered)")
                    
                    # Use a different search term for retry to avoid duplicates
                    retry_term = f"{term} alternative {business_type}"
                    
                    try:
                        retry_vendors = await self.llm_service.find_vendor_names(
                            retry_term, remaining, business_type, country, region
                        )
                        
                        # Filter and add
                        retry_new = [v for v in retry_vendors if v not in self.discovered_vendors 
                                     and v not in new_vendors]
                        new_vendors.extend(retry_new)
                    except Exception as e:
                        logger.error(f"Error during retry vendor search: {e}")
                
                return new_vendors
            except Exception as e:
                logger.error(f"Error processing batch for {business_type}: {e}")
                return []
                
        # Parallelism for smaller batches
        batch_concurrency = 3  # Process up to 3 batch types in parallel
        
        # Group batches by business_type for more efficient processing
        grouped_batches = {}
        for business_type, batch_count in batches:
            if business_type not in grouped_batches:
                grouped_batches[business_type] = 0
            grouped_batches[business_type] += batch_count
        
        # Create tasks for each business type
        batch_tasks = []
        for business_type, total_count in grouped_batches.items():
            batch_tasks.append(process_batch(business_type, total_count))
        
        # Process batches in parallel
        if batch_tasks:
            # Process in batches of batch_concurrency
            for i in range(0, len(batch_tasks), batch_concurrency):
                current_tasks = batch_tasks[i:i+batch_concurrency]
                batch_results = await asyncio.gather(*current_tasks, return_exceptions=True)
                
                # Process results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch {i+j} failed: {result}")
                    else:
                        # Update discovered vendors
                        self.discovered_vendors.update(result)
                        all_vendors.extend(result)
                
                # Short pause between batch groups
                if i + batch_concurrency < len(batch_tasks):
                    await asyncio.sleep(0.2)
        
        logger.info(f"Found {len(all_vendors)} unique vendors across all batches")
        return all_vendors
    
    async def research_vendor(self, name: str, term: str, business_type: str, 
                           country: str = None, region: str = None) -> VendorDetail:
        """
        Research a vendor with additional details.
        
        Args:
            name: Vendor name
            term: Original search term
            business_type: Type of business
            country: Optional country to focus on
            region: Optional region/state within the country
            
        Returns:
            VendorDetail object with researched information
        """
        vendor_data = await self.llm_service.research_vendor(name, term, business_type, country, region)
        return VendorDetail.from_dict(vendor_data)
    
    async def research_vendors_batch(self, 
                                  vendors: List[Tuple[str, str]], 
                                  term: str,
                                  country: str = None,
                                  region: str = None,
                                  concurrency: int = 25,  # Increased from 20 to 25
                                  with_progress_callback = None) -> List[VendorDetail]:
        """
        Research multiple vendors in parallel.
        
        Args:
            vendors: List of tuples (vendor_name, business_type)
            term: Original search term
            country: Optional country to focus on
            region: Optional region/state within the country
            concurrency: Maximum number of concurrent research operations
            with_progress_callback: Optional callback function to report progress
            
        Returns:
            List of VendorDetail objects with researched information
        """
        semaphore = asyncio.Semaphore(concurrency)
        results = []
        total_vendors = len(vendors)
        completed = 0
        
        # Create task completion tracking
        if with_progress_callback:
            with_progress_callback(0, total_vendors)
            
        async def process_vendor(name: str, b_type: str, idx: int) -> VendorDetail:
            nonlocal completed
            async with semaphore:
                try:
                    start_time = time.time()
                    result = await self.research_vendor(name, term, b_type, country, region)
                    elapsed = time.time() - start_time
                    logger.info(f"Researched vendor {name} in {elapsed:.2f}s")
                    
                    completed += 1
                    if with_progress_callback:
                        with_progress_callback(completed, total_vendors)
                    return result
                except Exception as e:
                    logger.error(f"Error researching vendor {name}: {e}")
                    completed += 1
                    if with_progress_callback:
                        with_progress_callback(completed, total_vendors)
                    
                    # Return a default/fallback vendor detail
                    return VendorDetail(
                        name=name,
                        description=f"A {b_type} specializing in {term}.",
                        website=f"https://www.{name.lower().replace(' ', '')}.com",
                        contact="contact@example.com",
                        specializations=[term],
                        relevance_score=5,
                        business_type=b_type
                    )
        
        # Create tasks for all vendors
        all_tasks = [process_vendor(name, b_type, i) for i, (name, b_type) in enumerate(vendors)]
        
        # Process tasks in optimized batches to avoid overwhelming the system
        vendor_details = []
        batch_size = 30  # Increased from 25 to 30
        
        # Dynamic batch sizing based on total vendor count
        if total_vendors > 50:
            batch_size = 35
        elif total_vendors < 15:
            batch_size = 15
        
        logger.info(f"Processing {total_vendors} vendors in batches of {batch_size} with concurrency {concurrency}")
        
        for i in range(0, len(all_tasks), batch_size):
            batch_tasks = all_tasks[i:i+batch_size]
            batch_start = time.time()
            
            try:
                # Add timeout to gather to prevent hanging
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results, handling exceptions
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        vendor_idx = i + j
                        if vendor_idx < len(vendors):
                            name, b_type = vendors[vendor_idx]
                            logger.error(f"Failed to research vendor {name}: {result}")
                            vendor_details.append(VendorDetail(
                                name=name,
                                description=f"A {b_type} specializing in {term}.",
                                website=f"https://www.{name.lower().replace(' ', '')}.com",
                                contact="contact@example.com",
                                specializations=[term],
                                relevance_score=5,
                                business_type=b_type
                            ))
                    else:
                        vendor_details.append(result)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Continue with the next batch
            
            # Log batch completion
            batch_elapsed = time.time() - batch_start
            logger.info(f"Completed batch {i//batch_size + 1}/{(len(all_tasks) + batch_size - 1)//batch_size} in {batch_elapsed:.2f}s")
            
            # Short pause between batches to prevent rate limiting but only if not the last batch
            if i + batch_size < len(all_tasks):
                await asyncio.sleep(0.2)
        
        return vendor_details 