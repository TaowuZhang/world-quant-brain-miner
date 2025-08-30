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

# Configure logging at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('alpha_miner.log')
    ]
)
logger = logging.getLogger(__name__)

class AlphaExpressionMiner:
    def __init__(self, credentials_path: str):
        logger.info("Initializing AlphaExpressionMiner")
        self.sess = requests.Session()
        self.credentials_path = credentials_path  # Store for reauth
        self.setup_auth(credentials_path)
        
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

    def remove_alpha_from_hopeful(self, expression: str, hopeful_file: str = "hopeful_alphas.json") -> bool:
        """Remove a mined alpha from hopeful_alphas.json."""
        try:
            if not os.path.exists(hopeful_file):
                logger.warning(f"Hopeful alphas file {hopeful_file} not found")
                return False
            
            # Create backup before modifying
            backup_file = f"{hopeful_file}.backup.{int(time.time())}"
            import shutil
            shutil.copy2(hopeful_file, backup_file)
            logger.debug(f"Created backup: {backup_file}")
            
            with open(hopeful_file, 'r') as f:
                hopeful_alphas = json.load(f)
            
            # Find and remove the alpha with matching expression
            original_count = len(hopeful_alphas)
            removed_alphas = []
            remaining_alphas = []
            
            for alpha in hopeful_alphas:
                if alpha.get('expression') == expression:
                    removed_alphas.append(alpha)
                else:
                    remaining_alphas.append(alpha)
            
            removed_count = len(removed_alphas)
            
            if removed_count > 0:
                # Save the updated file
                with open(hopeful_file, 'w') as f:
                    json.dump(remaining_alphas, f, indent=2)
                logger.info(f"Removed {removed_count} alpha(s) with expression '{expression}' from {hopeful_file}")
                logger.debug(f"Remaining alphas in file: {len(remaining_alphas)}")
                return True
            else:
                logger.info(f"No matching alpha found in {hopeful_file} for expression: {expression}")
                logger.debug(f"Available expressions: {[alpha.get('expression', 'N/A') for alpha in hopeful_alphas[:5]]}")
                return False
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {hopeful_file}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error removing alpha from {hopeful_file}: {e}")
            return False

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

    def get_parameter_ranges(self, parameters: List[Dict], auto_mode: bool = False) -> List[Dict]:
        """Get range and step size for each selected parameter."""
        for param in parameters:
            if auto_mode:
                # Use default ranges for automated mode
                original_value = param['value']
                if param['is_integer']:
                    # For integers, use ±20% range with step of 1
                    range_val = max(1, abs(original_value) * 0.2)
                    min_val = max(1, original_value - range_val)
                    max_val = original_value + range_val
                    step = 1
                else:
                    # For floats, use ±10% range with step of 10% of the range
                    range_val = abs(original_value) * 0.1
                    min_val = original_value - range_val
                    max_val = original_value + range_val
                    step = range_val / 5  # 5 steps across the range
                
                logger.info(f"Auto mode: Parameter {param['value']} -> range [{min_val:.2f}, {max_val:.2f}], step {step:.2f}")
            else:
                # Interactive mode - get user input
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

                        break
                    except ValueError as e:
                        print(f"Invalid input: {e}. Please try again.")

            param['min'] = min_val
            param['max'] = max_val
            param['step'] = step

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

    def test_alpha_batch(self, alpha_expressions: List[str]) -> List[Dict]:
        """Test multiple alpha expressions using multi_simulate approach."""
        logger.info(f"Testing batch of {len(alpha_expressions)} alphas using multi_simulate")
        
        # Group alphas into pools of 10 for better management
        pool_size = 10
        alpha_pools = []
        for i in range(0, len(alpha_expressions), pool_size):
            pool = alpha_expressions[i:i + pool_size]
            alpha_pools.append(pool)
        
        logger.info(f"Created {len(alpha_pools)} pools of size {pool_size}")
        
        all_results = []
        
        for pool_idx, pool in enumerate(alpha_pools):
            logger.info(f"Processing pool {pool_idx + 1}/{len(alpha_pools)} with {len(pool)} alphas")
            
            progress_urls = []
            alpha_mapping = {}  # Map progress URLs to alpha expressions
            
            # Submit all alphas in this pool
            for alpha_idx, alpha in enumerate(pool):
                logger.info(f"Submitting alpha {alpha_idx + 1}/{len(pool)} in pool {pool_idx + 1}")
                
                try:
                    # Generate simulation data
                    simulation_data = {
                        'type': 'REGULAR',
                        'settings': {
                            'instrumentType': 'EQUITY',
                            'region': 'USA',
                            'universe': 'TOP3000',
                            'delay': 1,
                            'decay': 0,
                            'neutralization': 'INDUSTRY',
                            'truncation': 0.08,
                            'pasteurization': 'ON',
                            'unitHandling': 'VERIFY',
                            'nanHandling': 'OFF',
                            'language': 'FASTEXPR',
                            'visualization': False,
                        },
                        'regular': alpha
                    }
                    
                    # Submit simulation
                    simulation_response = self.sess.post('https://api.worldquantbrain.com/simulations', 
                                                       json=simulation_data)
                    
                    # Handle authentication errors
                    if simulation_response.status_code == 401:
                        logger.info("Session expired, re-authenticating...")
                        self.setup_auth(self.credentials_path)
                        simulation_response = self.sess.post('https://api.worldquantbrain.com/simulations', 
                                                           json=simulation_data)
                    
                    if simulation_response.status_code != 201:
                        logger.error(f"Simulation API error for alpha {alpha}: {simulation_response.text}")
                        continue
                    
                    simulation_progress_url = simulation_response.headers.get('Location')
                    if not simulation_progress_url:
                        logger.error(f"No Location header in response for alpha {alpha}")
                        continue
                    
                    progress_urls.append(simulation_progress_url)
                    alpha_mapping[simulation_progress_url] = alpha
                    logger.info(f"Successfully submitted alpha {alpha_idx + 1}, got progress URL: {simulation_progress_url}")
                    
                except Exception as e:
                    logger.error(f"Error submitting alpha {alpha}: {str(e)}")
                    continue
            
            # Monitor progress for this pool
            if progress_urls:
                pool_results = self._monitor_pool_progress(progress_urls, alpha_mapping)
                all_results.extend(pool_results)
                logger.info(f"Pool {pool_idx + 1} completed with {len(pool_results)} successful simulations")
            
            # Wait between pools to avoid overwhelming the API
            if pool_idx + 1 < len(alpha_pools):
                logger.info(f"Waiting 1 second before next pool...")
                sleep(1)
        
        logger.info(f"Multi-simulate batch complete: {len(all_results)} successful simulations")
        return all_results

    def _monitor_pool_progress(self, progress_urls: List[str], alpha_mapping: Dict[str, str]) -> List[Dict]:
        """Monitor progress for a pool of simulations."""
        results = []
        max_wait_time = 3600  # 1 hour maximum wait time
        start_time = time.time()
        
        while progress_urls and (time.time() - start_time) < max_wait_time:
            logger.info(f"Monitoring {len(progress_urls)} simulations in pool...")
            
            completed_urls = []
            for progress_url in progress_urls:
                try:
                    # Check simulation status
                    sim_progress_resp = self.sess.get(progress_url)
                    
                    # Handle rate limits
                    if sim_progress_resp.status_code == 429:
                        retry_after = sim_progress_resp.headers.get("Retry-After", "60")
                        logger.info(f"Rate limit hit, waiting {retry_after} seconds...")
                        time.sleep(int(float(retry_after)))
                        continue
                    
                    if sim_progress_resp.status_code != 200:
                        logger.error(f"Failed to check progress for {progress_url}: {sim_progress_resp.status_code}")
                        completed_urls.append(progress_url)
                        continue
                    
                    # Handle empty response
                    if not sim_progress_resp.text.strip():
                        logger.debug("Empty response, simulation still initializing...")
                        continue
                    
                    # Try to parse JSON response
                    try:
                        sim_result = sim_progress_resp.json()
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to decode JSON response: {sim_progress_resp.text}")
                        continue
                    
                    status = sim_result.get("status")
                    
                    if status == "COMPLETE" or status == "WARNING":
                        alpha_expression = alpha_mapping.get(progress_url, "unknown")
                        logger.info(f"Simulation completed successfully for: {alpha_expression}")
                        results.append({
                            "expression": alpha_expression,
                            "result": sim_result
                        })
                        completed_urls.append(progress_url)
                        
                    elif status in ["FAILED", "ERROR"]:
                        alpha_expression = alpha_mapping.get(progress_url, "unknown")
                        logger.error(f"Simulation failed for alpha: {alpha_expression}")
                        completed_urls.append(progress_url)
                    
                    # Handle simulation limits
                    elif "SIMULATION_LIMIT_EXCEEDED" in sim_progress_resp.text:
                        alpha_expression = alpha_mapping.get(progress_url, "unknown")
                        logger.info(f"Simulation limit exceeded for alpha: {alpha_expression}")
                        completed_urls.append(progress_url)
                
                except Exception as e:
                    logger.error(f"Error monitoring progress for {progress_url}: {str(e)}")
                    completed_urls.append(progress_url)
            
            # Remove completed URLs
            for url in completed_urls:
                progress_urls.remove(url)
            
            # Wait before next check
            if progress_urls:
                sleep(10)
        
        if progress_urls:
            logger.warning(f"Pool monitoring timeout reached, {len(progress_urls)} simulations still pending")
        
        return results

    def test_alpha(self, alpha_expression: str) -> Dict:
        """Test a single alpha expression (legacy method for backward compatibility)."""
        logger.info(f"Testing single alpha: {alpha_expression}")
        results = self.test_alpha_batch([alpha_expression])
        if results:
            return {"status": "success", "result": results[0]["result"]}
        else:
            return {"status": "error", "message": "Simulation failed"}

def main():
    parser = argparse.ArgumentParser(description='Mine alpha expression variations')
    parser.add_argument('--credentials', type=str, default='./credential.txt',
                      help='Path to credentials file (default: ./credential.txt)')
    parser.add_argument('--expression', type=str, required=True,
                      help='Base alpha expression to mine variations from')
    parser.add_argument('--output', type=str, default='mined_expressions.json',
                      help='Output file for results (default: mined_expressions.json)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level (default: INFO)')
    parser.add_argument('--auto-mode', action='store_true',
                      help='Run in automated mode without user interaction')
    parser.add_argument('--output-file', type=str, default='mined_expressions.json',
                      help='Output file for results (default: mined_expressions.json)')
    
    args = parser.parse_args()
    
    # Update log level if specified
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Starting alpha expression mining with parameters:")
    logger.info(f"Expression: {args.expression}")
    logger.info(f"Output file: {args.output}")
    
    miner = AlphaExpressionMiner(args.credentials)
    
    # Parse expression and get parameters
    parameters = miner.parse_expression(args.expression)
    
    # Get parameter selection (automated or interactive)
    if args.auto_mode:
        # In auto mode, select all parameters
        selected_params = parameters
        logger.info(f"Auto mode: selected all {len(selected_params)} parameters")
    else:
        # Get user selection for parameters to vary
        selected_params = miner.get_user_parameter_selection(parameters)
    
    if not selected_params:
        logger.info("No parameters selected for variation")
        # Still remove the alpha from hopeful_alphas.json even if no parameters found
        logger.info("Mining completed (no parameters to vary), removing alpha from hopeful_alphas.json")
        removed = miner.remove_alpha_from_hopeful(args.expression)
        if removed:
            logger.info(f"Successfully removed alpha '{args.expression}' from hopeful_alphas.json")
        else:
            logger.warning(f"Could not remove alpha '{args.expression}' from hopeful_alphas.json (may not exist)")
        return
    
    # Get ranges and steps for selected parameters
    selected_params = miner.get_parameter_ranges(selected_params, auto_mode=args.auto_mode)
    
    # Generate variations
    variations = miner.generate_variations(args.expression, selected_params)
    
    # Test variations using multi_simulate
    logger.info(f"Testing {len(variations)} variations using multi_simulate (10 at a time)")
    results = miner.test_alpha_batch(variations)
    logger.info(f"Successfully tested {len(results)} variations")
    
    # Save results
    output_file = args.output_file if hasattr(args, 'output_file') else args.output
    logger.info(f"Saving {len(results)} results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Always remove the mined alpha from hopeful_alphas.json after completion
    # This prevents the same alpha from being processed again
    logger.info("Mining completed, removing alpha from hopeful_alphas.json")
    removed = miner.remove_alpha_from_hopeful(args.expression)
    if removed:
        logger.info(f"Successfully removed alpha '{args.expression}' from hopeful_alphas.json")
    else:
        logger.warning(f"Could not remove alpha '{args.expression}' from hopeful_alphas.json (may not exist)")
    
    logger.info("Mining complete")

if __name__ == "__main__":
    main()
