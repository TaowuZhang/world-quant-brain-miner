"""
Alpha Fetcher for WorldQuant Brain API
Fetches alphas from the WorldQuant Brain API with authentication and filtering.
"""

import requests
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaFetcher:
    """Fetches alphas from WorldQuant Brain API"""
    
    def __init__(self, credential_file: str = "credential.txt"):
        """Initialize the AlphaFetcher with credentials"""
        self.base_url = "https://api.worldquantbrain.com"
        self.session = requests.Session()
        self.credentials = self._load_credentials(credential_file)
        self._authenticate()
    
    def _load_credentials(self, credential_file: str) -> Tuple[str, str]:
        """Load credentials from file"""
        try:
            with open(credential_file, 'r') as f:
                credentials = json.load(f)
            
            if len(credentials) == 2:
                return credentials[0], credentials[1]
            else:
                raise ValueError("Credential file must contain exactly 2 elements: [email, password]")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Credential file {credential_file} not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in credential file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading credentials: {e}")
    
    def _authenticate(self):
        """Authenticate with WorldQuant Brain API"""
        try:
            from requests.auth import HTTPBasicAuth
            
            auth_url = f"{self.base_url}/authentication"
            
            response = self.session.post(
                auth_url,
                auth=HTTPBasicAuth(self.credentials[0], self.credentials[1])
            )
            
            if response.status_code == 201:
                logger.info("Successfully authenticated with WorldQuant Brain API")
            else:
                logger.error(f"Authentication failed: {response.status_code}")
                response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    def fetch_alphas(self, 
                    limit: int = 10, 
                    offset: int = 0,
                    status: str = "UNSUBMITTED,IS_FAIL",
                    date_from: Optional[str] = None,
                    date_to: Optional[str] = None,
                    order: str = "-dateCreated",
                    hidden: bool = False,
                    min_sharpe: Optional[float] = None,
                    min_fitness: Optional[float] = None,
                    min_margin: Optional[float] = None) -> Dict:
        """
        Fetch alphas from the API
        
        Args:
            limit: Number of alphas to fetch per request
            offset: Starting offset for pagination
            status: Alpha status filter (e.g., "UNSUBMITTED,IS_FAIL")
            date_from: Start date filter (ISO format)
            date_to: End date filter (ISO format)
            order: Sort order (e.g., "-dateCreated")
            hidden: Include hidden alphas
            min_sharpe: Minimum Sharpe ratio filter
            min_fitness: Minimum fitness filter
            min_margin: Minimum margin filter
            
        Returns:
            Dictionary containing alpha data and pagination info
        """
        try:
            url = f"{self.base_url}/users/self/alphas"
            params = {
                "limit": limit,
                "offset": offset,
                "status": status.replace(",", "\x1F"),  # Use unit separator instead of comma
                "order": order,
                "hidden": str(hidden).lower()
            }
            
            # Add date filters if provided
            if date_from:
                params["dateCreated>="] = date_from
            if date_to:
                params["dateCreated<"] = date_to
            
            # Add performance filters if provided
            if min_sharpe is not None:
                params["is.sharpe>"] = min_sharpe
            if min_fitness is not None:
                params["is.fitness>"] = min_fitness
            if min_margin is not None:
                params["is.margin>"] = min_margin
            
            logger.info(f"Fetching alphas with params: {params}")
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched {len(data.get('results', []))} alphas")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch alphas: {e}")
            raise
    
    def fetch_all_alphas(self, 
                        status: str = "UNSUBMITTED,IS_FAIL",
                        date_from: Optional[str] = None,
                        date_to: Optional[str] = None,
                        max_alphas: Optional[int] = None,
                        min_sharpe: Optional[float] = None,
                        min_fitness: Optional[float] = None,
                        min_margin: Optional[float] = None) -> List[Dict]:
        """
        Fetch all alphas with pagination
        
        Args:
            status: Alpha status filter
            date_from: Start date filter (ISO format)
            date_to: End date filter (ISO format)
            max_alphas: Maximum number of alphas to fetch (None for all)
            min_sharpe: Minimum Sharpe ratio filter
            min_fitness: Minimum fitness filter
            min_margin: Minimum margin filter
            
        Returns:
            List of all alpha dictionaries
        """
        all_alphas = []
        offset = 0
        batch_size = 100  # Default batch size
        
        while True:
            try:
                # Calculate how many alphas we still need
                remaining_needed = None
                if max_alphas:
                    remaining_needed = max_alphas - len(all_alphas)
                    if remaining_needed <= 0:
                        break
                    # Use smaller batch size if we need fewer alphas
                    current_limit = min(batch_size, remaining_needed)
                else:
                    current_limit = batch_size
                
                data = self.fetch_alphas(
                    limit=current_limit,
                    offset=offset,
                    status=status,
                    date_from=date_from,
                    date_to=date_to,
                    min_sharpe=min_sharpe,
                    min_fitness=min_fitness,
                    min_margin=min_margin
                )
                
                results = data.get('results', [])
                if not results:
                    break
                
                all_alphas.extend(results)
                logger.info(f"Fetched {len(all_alphas)} alphas so far...")
                
                # Check if we've reached the maximum
                if max_alphas and len(all_alphas) >= max_alphas:
                    all_alphas = all_alphas[:max_alphas]
                    break
                
                # Check if there are more pages
                if not data.get('next'):
                    break
                
                offset += current_limit
                
                # Add a small delay to be respectful to the API
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching alphas at offset {offset}: {e}")
                break
        
        logger.info(f"Total alphas fetched: {len(all_alphas)}")
        return all_alphas
    
    def get_correlation_data(self, alpha_id: str, max_retries: int = 10) -> Dict:
        """
        Get correlation data for a specific alpha with retry logic for async processing
        
        The WorldQuant Brain API processes correlation data asynchronously:
        - First call: Returns 200 with empty response (processing started)
        - Subsequent calls: Still processing, returns 200 with empty response
        - Final call: Returns 200 with actual correlation data
        
        Args:
            alpha_id: The alpha ID to get correlations for
            max_retries: Maximum number of retry attempts (increased for async processing)
            
        Returns:
            Dictionary containing correlation data
        """
        url = f"{self.base_url}/alphas/{alpha_id}/correlations/prod"
        
        for attempt in range(max_retries + 1):
            try:
                response = self.session.get(url)
                
                # Check for rate limiting (429)
                if response.status_code == 429:
                    retry_after = float(response.headers.get('Retry-After', 2.0))
                    wait_time = retry_after * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limited for alpha {alpha_id} (attempt {attempt + 1}), waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                
                # Check for empty response (200 but no content) - API is still processing
                if response.status_code == 200 and not response.content:
                    if attempt < max_retries:
                        # Use longer delays for async processing (5, 10, 15, 20, 25, 30, 35, 40, 45, 50 seconds)
                        wait_time = 5 + (attempt * 5)  # Progressive delay: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
                        logger.info(f"Correlation data still processing for alpha {alpha_id} (attempt {attempt + 1}), waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"Correlation data still processing for alpha {alpha_id} after {max_retries} retries (timeout)")
                        logger.warning(f"Response status: {response.status_code}, Headers: {dict(response.headers)}")
                        return {"error": "Processing timeout - correlation data not ready"}
                
                # Check rate limit headers and add extra delay if needed
                remaining = response.headers.get('RateLimit-Remaining', '60')
                if remaining.isdigit() and int(remaining) < 10:
                    logger.warning(f"Rate limit getting low ({remaining} remaining), adding extra delay...")
                    time.sleep(1.0)
                
                response.raise_for_status()
                
                # If we get here, we have a successful response with content
                try:
                    data = response.json()
                    logger.info(f"Successfully fetched correlation data for alpha {alpha_id} after {attempt + 1} attempts")
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON for alpha {alpha_id}: {e}")
                    return {"error": "Invalid JSON response"}
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    wait_time = 5 + (attempt * 5)  # Progressive delay for network issues too
                    logger.warning(f"Request failed for alpha {alpha_id} (attempt {attempt + 1}): {e}, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to fetch correlation data for alpha {alpha_id} after {max_retries} retries: {e}")
                    return {"error": str(e)}
        
        # This should never be reached, but just in case
        return {"error": "Max retries exceeded"}

def main():
    """Example usage of AlphaFetcher"""
    try:
        # Initialize fetcher
        fetcher = AlphaFetcher()
        
        # Fetch recent alphas (last 3 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        
        date_from = start_date.strftime("%Y-%m-%dT%H:%M:%S-04:00")
        date_to = end_date.strftime("%Y-%m-%dT%H:%M:%S-04:00")
        
        # Fetch alphas
        alphas = fetcher.fetch_all_alphas(
            status="UNSUBMITTED,IS_FAIL",
            date_from=date_from,
            date_to=date_to,
            max_alphas=50  # Limit for testing
        )
        
        print(f"Fetched {len(alphas)} alphas")
        
        # Example: Get correlation data for first alpha
        if alphas:
            first_alpha = alphas[0]
            alpha_id = first_alpha['id']
            print(f"Getting correlation data for alpha: {alpha_id}")
            
            correlation_data = fetcher.get_correlation_data(alpha_id)
            print(f"Correlation data: {json.dumps(correlation_data, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
