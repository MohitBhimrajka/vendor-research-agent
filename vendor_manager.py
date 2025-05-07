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
        
        # Process batches one by one (we'll still have parallelism inside LLMService)
        for business_type, batch_count in batches:
            # Find vendors for this batch
            batch_vendors = await self.llm_service.find_vendor_names(
                term, batch_count, business_type, country, region
            )
            
            # Filter out any vendors that have already been discovered
            new_vendors = [v for v in batch_vendors if v not in self.discovered_vendors]
            
            # Add to the set of discovered vendors
            self.discovered_vendors.update(new_vendors)
            all_vendors.extend(new_vendors)
            
            # If we didn't get enough vendors, try again for the remaining
            if len(new_vendors) < batch_count:
                remaining = batch_count - len(new_vendors)
                logger.info(f"Need {remaining} more vendors for batch (duplicates filtered)")
                
                # Use a different search term for retry to avoid duplicates
                retry_term = f"{term} alternative"
                
                # Try again with a different search approach - using a new async call
                try:
                    retry_vendors = await self.llm_service.find_vendor_names(
                        retry_term, remaining, business_type, country, region
                    )
                    
                    # Filter and add
                    retry_new = [v for v in retry_vendors if v not in self.discovered_vendors]
                    self.discovered_vendors.update(retry_new)
                    all_vendors.extend(retry_new)
                except Exception as e:
                    logger.error(f"Error during retry vendor search: {e}")
                    # Continue with what we have rather than failing completely
        
        # Return the vendors found
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
                                  concurrency: int = 20,  # Increased from 15 to 20
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
        
        async def process_vendor(name: str, b_type: str, idx: int) -> VendorDetail:
            async with semaphore:
                try:
                    result = await self.research_vendor(name, term, b_type, country, region)
                    if with_progress_callback:
                        with_progress_callback(idx + 1, len(vendors))
                    return result
                except Exception as e:
                    logger.error(f"Error researching vendor {name}: {e}")
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
        
        # Create tasks for all vendors - process them in batches for even better parallel performance
        all_tasks = [process_vendor(name, b_type, i) for i, (name, b_type) in enumerate(vendors)]
        
        # Use a chunked approach to avoid overwhelming the system with too many concurrent tasks
        vendor_details = []
        batch_size = 25  # Process up to 25 vendors at once (increased from 20)
        
        for i in range(0, len(all_tasks), batch_size):
            batch_tasks = all_tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch_tasks)
            vendor_details.extend(batch_results)
        
        return vendor_details 