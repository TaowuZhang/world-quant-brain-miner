import argparse
import requests
import json
import os
import re
from time import sleep
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Tuple
import time
import logging
import random

# Configure logging at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('improved_alpha_miner.log')
    ]
)
logger = logging.getLogger(__name__)

class ImprovedAlphaExpressionMiner:
    def __init__(self, credentials_path: str):
        logger.info("Initializing ImprovedAlphaExpressionMiner")
        self.sess = requests.Session()
        # Set longer timeouts
        self.sess.timeout = (30, 300)  # (connect_timeout, read_timeout)
        self.setup_auth(credentials_path)
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit_delay = 15  # seconds between requests
        
    def setup_auth(self, credentials_path: str) -> None:
        """Set up authentication with WorldQuant Brain."""
        logger.info(f"Loading credentials from {credentials_path}")
        with open(credentials_path) as f:
            credentials = json.load(f)
        
        username, password = credentials
        self.sess.auth = HTTPBasicAuth(username, password)
        
        logger.info("Authenticating with WorldQuant Brain...")
        response = self.sess.post('https://api.worldquantbrain.com/authentication')
        logger.info(f"Authentication response status: {response.status_code}")
        
        if response.status_code != 201:
            logger.error(f"Authentication failed: {response.text}")
            raise Exception(f"Authentication failed: {response.text}")
        logger.info("Authentication successful")

    def rate_limit_wait(self):
        """Implement rate limiting to avoid overwhelming the API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds")
            sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1

    def parse_expression(self, expression: str) -> List[Dict]:
        """Parse the alpha expression to find numeric parameters and their positions."""
        logger.info(f"Parsing expression: {expression}")
        parameters = []
        # Match numbers that:
        # 1. Are preceded by '(' or ',' or space
        # 2. Are not part of a variable name (not preceded/followed by letters)
        # 3. Can be integers or decimals
        for match in re.finditer(r'(?<=[,()\s])(-?\d*\.?\d+)(?![a-zA-Z])', expression):
            number_str = match.group()
            try:
                number = float(number_str)
            except ValueError:
                continue
            start_pos = match.start()
            end_pos = match.end()
            parameters.append({
                'value': number,
                'start': start_pos,
                'end': end_pos,
                'context': expression[max(0, start_pos-20):min(len(expression), end_pos+20)],
                'is_integer': number.is_integer()
            })
            logger.debug(f"Found parameter: {number} at position {start_pos}-{end_pos}")
        
        logger.info(f"Found {len(parameters)} parameters to vary")
        return parameters

    def get_user_parameter_selection(self, parameters: List[Dict]) -> List[Dict]:
        """Interactively get user selection for parameters to vary."""
        if not parameters:
            logger.info("No parameters found in expression")
            return []

        print("\nFound the following parameters in the expression:")
        for i, param in enumerate(parameters, 1):
            print(f"{i}. Value: {param['value']} | Context: ...{param['context']}...")

        while True:
            try:
                selection = input("\nEnter the numbers of parameters to vary (comma-separated, or 'all'): ")
                if selection.lower() == 'all':
                    selected_indices = list(range(len(parameters)))
                else:
                    selected_indices = [int(x.strip())-1 for x in selection.split(',')]
                    if not all(0 <= i < len(parameters) for i in selected_indices):
                        raise ValueError("Invalid parameter number")
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")

        selected_params = [parameters[i] for i in selected_indices]
        return selected_params

    def get_parameter_ranges(self, parameters: List[Dict]) -> List[Dict]:
        """Get range and step size for each selected parameter."""
        for param in parameters:
            while True:
                try:
                    print(f"\nParameter: {param['value']} | Context: ...{param['context']}...")
                    range_input = input("Enter range (e.g., '10' for ±10, or '5,15' for 5 to 15): ")
                    if ',' in range_input:
                        min_val, max_val = map(float, range_input.split(','))
                    else:
                        range_val = float(range_input)
                        min_val = param['value'] - range_val
                        max_val = param['value'] + range_val

                    step = float(input("Enter step size: "))
                    if step <= 0:
                        raise ValueError("Step size must be positive")
                    if min_val >= max_val:
                        raise ValueError("Min value must be less than max value")

                    # If the original value was an integer, ensure step is also an integer
                    if param['is_integer'] and not step.is_integer():
                        print("Warning: Original value is integer, rounding step to nearest integer")
                        step = round(step)

                    param['min'] = min_val
                    param['max'] = max_val
                    param['step'] = step
                    break
                except ValueError as e:
                    print(f"Invalid input: {e}. Please try again.")

        return parameters

    def generate_variations(self, expression: str, parameters: List[Dict]) -> List[str]:
        """Generate variations of the expression based on user-selected parameters and ranges."""
        logger.info("Generating variations based on selected parameters")
        variations = []
        
        # Sort parameters in reverse order to modify from end to start
        parameters.sort(reverse=True, key=lambda x: x['start'])
        
        # Generate all combinations of parameter values
        param_values = []
        for param in parameters:
            values = []
            current = param['min']
            while current <= param['max']:
                # Format the number appropriately based on whether it's an integer
                if param['is_integer']:
                    value = str(int(round(current)))
                else:
                    # Format to remove trailing zeros and unnecessary decimal points
                    value = f"{current:.10f}".rstrip('0').rstrip('.')
                values.append(value)
                current += param['step']
            
            # Add original value if not already included
            original_value = str(int(param['value'])) if param['is_integer'] else f"{param['value']:.10f}".rstrip('0').rstrip('.')
            if original_value not in values:
                values.append(original_value)
            
            param_values.append(values)
        
        # Generate all combinations
        from itertools import product
        for value_combination in product(*param_values):
            new_expr = expression
            for param, value in zip(parameters, value_combination):
                new_expr = new_expr[:param['start']] + value + new_expr[param['end']:]
            variations.append(new_expr)
            logger.debug(f"Generated variation: {new_expr}")
        
        logger.info(f"Generated {len(variations)} total variations")
        return variations

    def test_alpha(self, alpha_expression: str, max_retries: int = 3) -> Dict:
        """Test an alpha expression using WorldQuant Brain simulation with improved error handling."""
        logger.info(f"Testing alpha: {alpha_expression}")
        
        simulation_data = {
            'type': 'REGULAR',
            'settings': {
                'instrumentType': 'EQUITY',
                'region': 'USA',
                'universe': 'TOP3000',
                'delay': 1,
                'decay': 0,
                'neutralization': 'INDUSTRY',
                'truncation': 0.01,
                'pasteurization': 'ON',
                'unitHandling': 'VERIFY',
                'nanHandling': 'OFF',
                'language': 'FASTEXPR',
                'visualization': False,
            },
            'regular': alpha_expression
        }

        for attempt in range(max_retries):
            try:
                # Rate limiting
                self.rate_limit_wait()
                
                sim_resp = self.sess.post('https://api.worldquantbrain.com/simulations', json=simulation_data)
                logger.info(f"Simulation creation response: {sim_resp.status_code}")
                
                if sim_resp.status_code == 429:  # Rate limited
                    wait_time = int(sim_resp.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    sleep(wait_time)
                    continue
                    
                if sim_resp.status_code != 201:
                    logger.error(f"Simulation creation failed: {sim_resp.text}")
                    if attempt < max_retries - 1:
                        wait_time = 30 * (2 ** attempt)  # Exponential backoff
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        sleep(wait_time)
                        continue
                    return {"status": "error", "message": sim_resp.text}

                sim_progress_url = sim_resp.headers.get('location')
                if not sim_progress_url:
                    logger.error("No simulation ID received in response headers")
                    return {"status": "error", "message": "No simulation ID received"}
                
                logger.info(f"Monitoring simulation at: {sim_progress_url}")
                
                # Monitor simulation progress with improved timeout handling
                result = self.monitor_simulation(sim_progress_url, max_timeout_minutes=30)
                if result:
                    return result
                else:
                    logger.error(f"Simulation monitoring failed for: {alpha_expression}")
                    return {"status": "error", "message": "Simulation monitoring failed"}
                    
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 30 * (2 ** attempt)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    sleep(wait_time)
                else:
                    logger.error(f"Simulation timed out after {max_retries} attempts")
                    return {"status": "error", "message": "Simulation timed out"}
                    
            except Exception as e:
                logger.error(f"Error in simulation (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 30 * (2 ** attempt)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    sleep(wait_time)
                else:
                    logger.error(f"Failed to test alpha after {max_retries} attempts")
                    return {"status": "error", "message": str(e)}
        
        return {"status": "error", "message": "Max retries exceeded"}

    def monitor_simulation(self, sim_progress_url: str, max_timeout_minutes: int = 30) -> Dict:
        """Monitor simulation progress with improved timeout handling."""
        start_time = time.time()
        max_timeout_seconds = max_timeout_minutes * 60
        base_sleep_time = 10
        max_sleep_time = 120
        
        attempt = 0
        
        while (time.time() - start_time) < max_timeout_seconds:
            attempt += 1
            elapsed_minutes = (time.time() - start_time) / 60
            
            try:
                # Rate limiting
                self.rate_limit_wait()
                
                sim_progress_resp = self.sess.get(sim_progress_url)
                
                # Handle empty response
                if not sim_progress_resp.text.strip():
                    logger.debug(f"Empty response, simulation still initializing... (elapsed: {elapsed_minutes:.1f} minutes)")
                    sleep_time = min(base_sleep_time * (1.2 ** (attempt - 1)), max_sleep_time)
                    sleep(sleep_time)
                    continue
                
                # Try to parse JSON response
                try:
                    progress_data = sim_progress_resp.json()
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode JSON response: {sim_progress_resp.text[:200]}...")
                    sleep_time = min(base_sleep_time * (1.2 ** attempt), max_sleep_time)
                    sleep(sleep_time)
                    continue
                
                status = progress_data.get("status")
                logger.info(f"Simulation status: {status} (elapsed: {elapsed_minutes:.1f} minutes)")
                
                if status == "COMPLETE" or status == "WARNING":
                    logger.info("Simulation completed successfully")
                    return {"status": "success", "result": progress_data}
                elif status in ["FAILED", "ERROR"]:
                    logger.error(f"Simulation failed: {progress_data}")
                    return {"status": "error", "message": progress_data}
                
                # Exponential backoff for polling
                sleep_time = min(base_sleep_time * (1.2 ** (attempt - 1)), max_sleep_time)
                sleep(sleep_time)
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout on monitoring attempt {attempt}: {str(e)}")
                if (time.time() - start_time) < max_timeout_seconds:
                    sleep_time = min(base_sleep_time * (2 ** attempt), max_sleep_time)
                    logger.info(f"Waiting {sleep_time} seconds before retry...")
                    sleep(sleep_time)
                else:
                    logger.error(f"Simulation monitoring timed out after {max_timeout_minutes} minutes")
                    return {"status": "error", "message": "Simulation monitoring timed out"}
                    
            except Exception as e:
                logger.error(f"Request error: {str(e)}")
                if (time.time() - start_time) < max_timeout_seconds:
                    sleep_time = min(base_sleep_time * (1.5 ** attempt), max_sleep_time)
                    sleep(sleep_time)
                else:
                    logger.error(f"Simulation monitoring failed after {max_timeout_minutes} minutes")
                    return {"status": "error", "message": str(e)}

        logger.error(f"Simulation monitoring timed out after {max_timeout_minutes} minutes")
        return {"status": "error", "message": "Simulation monitoring timed out"}

def main():
    parser = argparse.ArgumentParser(description='Mine alpha expression variations with improved rate limiting')
    parser.add_argument('--credentials', type=str, default='./credential.txt',
                      help='Path to credentials file (default: ./credential.txt)')
    parser.add_argument('--expression', type=str, required=True,
                      help='Base alpha expression to mine variations from')
    parser.add_argument('--output', type=str, default='mined_expressions.json',
                      help='Output file for results (default: mined_expressions.json)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level (default: INFO)')
    parser.add_argument('--rate-limit-delay', type=int, default=15,
                      help='Seconds to wait between requests (default: 15)')
    parser.add_argument('--max-variations', type=int, default=100,
                      help='Maximum number of variations to test (default: 100)')
    
    args = parser.parse_args()
    
    # Update log level if specified
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Starting improved alpha expression mining with parameters:")
    logger.info(f"Expression: {args.expression}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Rate limit delay: {args.rate_limit_delay} seconds")
    logger.info(f"Max variations: {args.max_variations}")
    
    miner = ImprovedAlphaExpressionMiner(args.credentials)
    miner.rate_limit_delay = args.rate_limit_delay
    
    # Parse expression and get parameters
    parameters = miner.parse_expression(args.expression)
    
    # Get user selection for parameters to vary
    selected_params = miner.get_user_parameter_selection(parameters)
    
    if not selected_params:
        logger.info("No parameters selected for variation")
        return
    
    # Get ranges and steps for selected parameters
    selected_params = miner.get_parameter_ranges(selected_params)
    
    # Generate variations
    variations = miner.generate_variations(args.expression, selected_params)
    
    # Limit the number of variations if too many
    if len(variations) > args.max_variations:
        logger.info(f"Limiting variations from {len(variations)} to {args.max_variations}")
        # Randomly sample variations to get a good distribution
        import random
        random.seed(42)  # For reproducibility
        variations = random.sample(variations, args.max_variations)
    
    # Test variations
    results = []
    total = len(variations)
    successful_tests = 0
    
    for i, var in enumerate(variations, 1):
        logger.info(f"Testing variation {i}/{total}: {var}")
        result = miner.test_alpha(var)
        
        if result["status"] == "success":
            logger.info(f"Successful test for: {var}")
            results.append({
                "expression": var,
                "result": result["result"]
            })
            successful_tests += 1
        else:
            logger.error(f"Failed to test variation: {var}")
            logger.error(f"Error: {result['message']}")
        
        # Progress update every 10 tests
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{total} tests completed, {successful_tests} successful")
    
    # Save results
    logger.info(f"Saving {len(results)} results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Mining complete. {successful_tests}/{total} variations tested successfully")

if __name__ == "__main__":
    main()
