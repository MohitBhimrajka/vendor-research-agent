import os
import asyncio
import json
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

from utils import logger, retry_with_backoff, memoize

# Load environment variables
load_dotenv()

class LLMService:
    """Service for interacting with LLM (Gemini)."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-pro-preview-05-06"):
        """
        Initialize the LLM service.
        
        Args:
            api_key: API key for Gemini. If None, tries to get from environment.
            model: Model name to use for generation.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        self.model = model
        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"LLM Service initialized with model: {model}")
    
    @retry_with_backoff(max_retries=3, initial_backoff=1.0)
    async def generate_text(self, prompt: str, temperature: float = 0.7, 
                           timeout: int = 300, use_search: bool = False) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Temperature for generation (higher = more creative)
            timeout: Timeout in seconds
            use_search: Whether to enable Google Search tool
            
        Returns:
            Generated text response
        """
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=temperature,
            )
            
            # Add Google Search tool if requested
            if use_search:
                generate_content_config.tools = [
                    types.Tool(google_search=types.GoogleSearch()),
                ]
            
            # Create an event loop if there isn't one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Define a function to wrap synchronous API call
            def sync_call():
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=generate_content_config,
                )
                return response.text
            
            # Run the synchronous function in a separate thread with timeout
            future = loop.run_in_executor(None, sync_call)
            result = await asyncio.wait_for(future, timeout=timeout)
            
            return result
        except asyncio.TimeoutError:
            logger.error(f"LLM request timed out after {timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    async def generate_batch(self, prompts: List[str], temperature: float = 0.7, 
                            concurrency: int = 20, use_search: bool = False) -> List[str]:
        """
        Generate responses for multiple prompts concurrently.
        
        Args:
            prompts: List of prompts to send to the LLM
            temperature: Temperature for generation
            concurrency: Maximum number of concurrent requests
            use_search: Whether to enable Google Search tool
            
        Returns:
            List of generated responses in the same order as prompts
        """
        # Use semaphore to control maximum concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_prompt(prompt: str) -> str:
            async with semaphore:
                try:
                    return await self.generate_text(prompt, temperature, use_search=use_search)
                except Exception as e:
                    logger.error(f"Failed to process prompt: {e}")
                    return f"Error processing request: {str(e)}"
        
        # Process prompts in optimized batches
        processed_results = []
        batch_size = 15  # Increased from 10
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            tasks = [process_prompt(prompt) for prompt in batch_prompts]
            
            try:
                # Add timeout to gather to prevent hanging
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any exceptions
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process prompt: {result}")
                        processed_results.append(f"Error processing request: {str(result)}")
                    else:
                        processed_results.append(result)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Add error placeholders for the entire batch
                processed_results.extend([f"Error processing request: {str(e)}"] * len(batch_prompts))
            
            # Short pause between batches to prevent rate limiting
            await asyncio.sleep(0.1)
        
        return processed_results
    
    @memoize
    async def disambiguate_term(self, term: str) -> List[Dict[str, str]]:
        """
        Generate multiple interpretations of a search term.
        
        Args:
            term: The search term to disambiguate
            
        Returns:
            List of dicts with 'interpretation' and 'description' keys
        """
        prompt = f"""
        I need to find vendors related to "{term}". Please provide 2-3 different interpretations of what this term could refer to 
        in a business context. For each interpretation, provide a brief description of that interpretation.
        
        Format your response as a JSON array of objects, each with 'interpretation' and 'description' properties:
        [
            {{"interpretation": "...", "description": "..."}},
            {{"interpretation": "...", "description": "..."}}
        ]
        
        Only include the raw JSON in your response, no other text. DO NOT use Markdown code blocks (```). Just output the plain JSON directly.
        """
        
        try:
            response = await self.generate_text(prompt, temperature=0.3)
            
            # Clean the response by removing Markdown code blocks if present
            cleaned_response = response
            if "```json" in response:
                cleaned_response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                cleaned_response = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(cleaned_response)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {response}")
            logger.error(f"JSON error: {e}")
            # Fallback to a simple interpretation
            return [{"interpretation": term, "description": f"Vendors related to {term}"}]
    
    # Removed memoize for now to avoid coroutine reuse issues
    async def find_vendor_names(self, term: str, count: int, business_type: str, 
                               country: str = None, region: str = None) -> List[str]:
        """
        Find real vendor names for a given term and business type.
        
        Args:
            term: Search term or category
            count: Number of vendors to find
            business_type: Type of vendor (manufacturer, distributor, retailer, etc.)
            country: Optional country to focus on
            region: Optional region/state within the country
            
        Returns:
            List of vendor names
        """
        # Create a unique search key for this specific request
        search_key = f"{term}_{count}_{business_type}_{country}_{region}_{int(time.time())}"
        logger.info(f"Finding vendors with search key: {search_key}")
        
        location_context = ""
        if country and country != "Other":
            location_context = f" based in {country}"
            if region:
                location_context += f", specifically in the {region} region"
        
        prompt = f"""
        Use Google Search to find exactly {count} real, existing vendors that are {business_type}s in the "{term}" industry or category{location_context}.
        
        Please search for actual companies that exist, not fictional ones. Use search queries like:
        - "top {business_type}s for {term} {country if country else ''}"
        - "{term} {business_type} companies {location_context}"
        - "leading {term} {business_type}s {location_context}"
        - "list of {term} {business_type} suppliers {location_context}"
        
        Guidelines:
        - Only include real companies that actually exist
        - Search for diverse companies (varying sizes, focus areas)
        - Use the company's official name
        - Include a mix of well-known and specialized companies
        - Ensure names are unique and not duplicated
        - Continue searching until you find exactly {count} vendors
        
        Format your response as a JSON array of strings, one for each vendor name:
        ["Vendor Name 1", "Vendor Name 2", ...]
        
        Only include the raw JSON in your response, no other text. DO NOT use Markdown code blocks (```). Just output the plain JSON directly.
        """
        
        try:
            # Use a longer timeout for vendor searches
            response = await self.generate_text(prompt, temperature=0.2, use_search=True, timeout=600)
            
            # Clean the response by removing Markdown code blocks if present
            cleaned_response = response
            if "```json" in response:
                cleaned_response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                cleaned_response = response.split("```")[1].split("```")[0].strip()
                
            result = json.loads(cleaned_response)
            return result  # Return all vendors found
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {response}")
            logger.error(f"JSON error: {e}")
            # Fallback to simple vendor names
            return [f"{term.capitalize()} {business_type.capitalize()} {i+1}" for i in range(count)]
        except Exception as e:
            logger.error(f"Error finding vendors: {e}")
            # Fallback in case of any other errors
            return [f"{term.capitalize()} {business_type.capitalize()} {i+1}" for i in range(count)]
    
    @memoize
    async def research_vendor(self, vendor_name: str, term: str, business_type: str,
                            country: str = None, region: str = None) -> Dict[str, Any]:
        """
        Research a vendor with additional details.
        
        Args:
            vendor_name: Name of the vendor to research
            term: The original search term/category
            business_type: Type of business
            country: Optional country to focus on
            region: Optional region/state within the country
            
        Returns:
            Dictionary with vendor details
        """
        location_context = ""
        if country and country != "Other":
            location_context = f" operating in {country}"
            if region:
                location_context += f", specifically in the {region} region"
        
        prompt = f"""
        Use Google Search to find detailed information about "{vendor_name}", a real {business_type} in the "{term}" industry{location_context}.
        
        Search for the actual company's details. Use search queries like:
        - "{vendor_name} company information"
        - "{vendor_name} official website"
        - "{vendor_name} contact information"
        - "{vendor_name} products services {term}"
        
        Generate a realistic company profile with these fields based on what you find:
        - description: A 1-2 sentence company description based on their actual business
        - website: The company's actual website URL (if found)
        - contact: Real contact information (email and/or phone) if publicly available
        - specializations: List of 3-5 specific focus areas or specialties based on their actual business
        - relevance_score: A score from 1-10 indicating how relevant this vendor is to "{term}" (10 being highest)
          IMPORTANT: Vary the relevance scores. Assign scores across the full range (1-10) based on:
          * How directly the vendor's actual products/services relate to the search term
          * How specialized they are in the main category versus having a broader focus
          * Their estimated market share or prominence in the industry
          * The breadth of their product/service offerings
        - business_type: Provide a specific subtype of "{business_type}" that best describes their actual business model
        
        Format your response as a JSON object with these exact field names:
        {{
            "description": "...",
            "website": "...",
            "contact": "...",
            "specializations": ["...", "...", ...],
            "relevance_score": X,
            "business_type": "..."
        }}
        
        Only include the raw JSON in your response, no other text. DO NOT use Markdown code blocks (```). Just output the plain JSON directly.
        """
        
        try:
            # Use a longer timeout for vendor research
            response = await self.generate_text(prompt, temperature=0.3, use_search=True, timeout=300)
            
            # Clean the response by removing Markdown code blocks if present
            cleaned_response = response
            if "```json" in response:
                cleaned_response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                cleaned_response = response.split("```")[1].split("```")[0].strip()
                
            result = json.loads(cleaned_response)
            result["name"] = vendor_name  # Add name to the result
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response for vendor {vendor_name}: {response}")
            logger.error(f"JSON error: {e}")
            # Fallback to basic vendor details
            return {
                "name": vendor_name,
                "description": f"A {business_type} specializing in {term}.",
                "website": f"https://www.{vendor_name.lower().replace(' ', '')}.com",
                "contact": "contact@example.com",
                "specializations": [term],
                "relevance_score": 5,
                "business_type": business_type
            }
        except Exception as e:
            logger.error(f"Error researching vendor {vendor_name}: {e}")
            # Fallback in case of any other errors
            return {
                "name": vendor_name,
                "description": f"A {business_type} specializing in {term}.",
                "website": f"https://www.{vendor_name.lower().replace(' ', '')}.com",
                "contact": "contact@example.com",
                "specializations": [term],
                "relevance_score": 5,
                "business_type": business_type
            } 