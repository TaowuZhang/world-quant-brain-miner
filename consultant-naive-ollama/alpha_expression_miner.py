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
from itertools import product

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
        
        # Define the simulation parameter choices based on the API schema
        self.simulation_choices = {
            'instrumentType': ['EQUITY'],
            'region': ['USA', 'GLB', 'EUR', 'ASI', 'CHN'],
            'universe': {
                'USA': ['TOP3000', 'TOP1000', 'TOP500', 'TOP200', 'ILLIQUID_MINVOL1M', 'TOPSP500'],
                'GLB': ['TOP3000', 'MINVOL1M', 'TOPDIV3000'],
                'EUR': ['TOP2500', 'TOP1200', 'TOP800', 'TOP400', 'ILLIQUID_MINVOL1M'],
                'ASI': ['MINVOL1M', 'ILLIQUID_MINVOL1M'],
                'CHN': ['TOP2000U']
            },
            'delay': {
                'USA': [1, 0],
                'GLB': [1],
                'EUR': [1, 0],
                'ASI': [1],
                'CHN': [0, 1]
            },
            'decay': [0],  # Fixed to 0 to reduce resource usage 
            'neutralization': {
                'USA': ['NONE', 'REVERSION_AND_MOMENTUM', 'STATISTICAL', 'CROWDING', 'FAST', 'SLOW', 'MARKET', 'SECTOR', 'INDUSTRY', 'SUBINDUSTRY', 'SLOW_AND_FAST'],
                'GLB': ['NONE', 'REVERSION_AND_MOMENTUM', 'STATISTICAL', 'CROWDING', 'FAST', 'SLOW', 'MARKET', 'SECTOR', 'INDUSTRY', 'SUBINDUSTRY', 'COUNTRY', 'SLOW_AND_FAST'],
                'EUR': ['NONE', 'REVERSION_AND_MOMENTUM', 'STATISTICAL', 'CROWDING', 'FAST', 'SLOW', 'MARKET', 'SECTOR', 'INDUSTRY', 'SUBINDUSTRY', 'COUNTRY', 'SLOW_AND_FAST'],
                'ASI': ['NONE', 'REVERSION_AND_MOMENTUM', 'STATISTICAL', 'CROWDING', 'FAST', 'SLOW', 'MARKET', 'SECTOR', 'INDUSTRY', 'SUBINDUSTRY', 'COUNTRY', 'SLOW_AND_FAST'],
                'CHN': ['NONE', 'REVERSION_AND_MOMENTUM', 'CROWDING', 'FAST', 'SLOW', 'MARKET', 'SECTOR', 'INDUSTRY', 'SUBINDUSTRY', 'SLOW_AND_FAST']
            },
            'truncation': [0.08],
            'pasteurization': ['ON', 'OFF'],
            'unitHandling': ['VERIFY'],
            'nanHandling': ['ON', 'OFF'],
            'maxTrade': ['OFF', 'ON'],
            'language': ['FASTEXPR'],
            'visualization': [False, True]
        }
        
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
        for value_combination in product(*param_values):
            new_expr = expression
            for param, value in zip(parameters, value_combination):
                new_expr = new_expr[:param['start']] + value + new_expr[param['end']:]
            variations.append(new_expr)
            logger.debug(f"Generated variation: {new_expr}")
        
        logger.info(f"Generated {len(variations)} total variations")
        return variations

    def generate_simulation_configurations(self, target_region: str = None) -> List[Dict]:
        """
        Generate different simulation configurations based on the API schema.
        If target_region is specified, generate ALL possible valid combinations for that region.
        Returns a list of configuration dictionaries.
        """
        logger.info(f"Generating simulation configurations")
        
        if target_region:
            logger.info(f"Focusing on region: {target_region} - will generate ALL possible combinations")
        
        configs = []
        
        # If targeting a specific region, generate ALL possible combinations
        if target_region and target_region in self.simulation_choices['region']:
            logger.info(f"Generating ALL possible combinations for region: {target_region}")
            
            # Get all available options for this region
            available_universes = self.simulation_choices['universe'].get(target_region, [])
            available_delays = self.simulation_choices['delay'].get(target_region, [])
            available_neutralizations = self.simulation_choices['neutralization'].get(target_region, [])
            available_truncations = self.simulation_choices['truncation']
            available_decays = self.simulation_choices['decay']
            available_pasteurizations = self.simulation_choices['pasteurization']
            available_nan_handlings = self.simulation_choices['nanHandling']
             
            logger.info(f"Available options for {target_region}:")
            logger.info(f"  Universes: {available_universes}")
            logger.info(f"  Delays: {available_delays}")
            logger.info(f"  Neutralizations: {available_neutralizations}")
            logger.info(f"  Truncations: {len(available_truncations)} values")
            logger.info(f"  Decays: {len(available_decays)} values (fixed to 0)")
            
            # Calculate total possible combinations
            total_combinations = (len(available_universes) * len(available_delays) * 
                                len(available_neutralizations) * len(available_truncations) * 
                                len(available_decays) * len(available_pasteurizations) * 
                                len(available_nan_handlings))
            logger.info(f"Total possible combinations for {target_region}: {total_combinations:,}")
            
            # Generate ALL combinations
            for universe in available_universes:
                for delay in available_delays:
                    for neutralization in available_neutralizations:
                        for truncation in available_truncations:
                            for decay in available_decays:
                                for pasteurization in available_pasteurizations:
                                    for nan_handling in available_nan_handlings:
                                        config = {
                                            'type': 'REGULAR',
                                            'settings': {
                                                'instrumentType': 'EQUITY',
                                                'region': target_region,
                                                'universe': universe,
                                                'delay': delay,
                                                'decay': decay,
                                                'neutralization': neutralization,
                                                'truncation': truncation,
                                                'pasteurization': pasteurization,
                                                'unitHandling': 'VERIFY',
                                                'nanHandling': nan_handling,
                                                'maxTrade': 'OFF',
                                                'language': 'FASTEXPR',
                                                'visualization': False,
                                            }
                                        }
                                        configs.append(config)
                                        
                                        
            
            logger.info(f"Generated {len(configs)} configurations for region {target_region}")
            logger.info(f"This represents ALL possible combinations for the specified region")
            return configs
        
        # Original logic for multiple regions (handpicked approach)
        # Start with a base configuration
        base_config = {
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
                'maxTrade': 'OFF',
                'language': 'FASTEXPR',
                'visualization': False,
            }
        }
        configs.append(base_config)
        
        # Generate variations by changing key parameters
        # Focus on the most important parameters first
        
        # 1. Different regions with their corresponding universes and delays
        for region in self.simulation_choices['region']:
            if region != 'USA':  # Skip USA as it's in base config
                for universe in self.simulation_choices['universe'].get(region, ['TOP3000']):
                    for delay in self.simulation_choices['delay'].get(region, [1]):
                        config = base_config.copy()
                        config['settings'] = config['settings'].copy()
                        config['settings']['region'] = region
                        config['settings']['universe'] = universe
                        config['settings']['delay'] = delay
                        configs.append(config)
                        

        
        # 2. Different neutralization strategies
        for region in ['USA', 'GLB', 'EUR']:  # Focus on main regions
            for neutralization in self.simulation_choices['neutralization'].get(region, ['INDUSTRY']):
                if neutralization != 'INDUSTRY':  # Skip INDUSTRY as it's in base config
                    config = base_config.copy()
                    config['settings'] = config['settings'].copy()
                    config['settings']['region'] = region
                    config['settings']['neutralization'] = neutralization
                    # Adjust universe and delay based on region
                    config['settings']['universe'] = self.simulation_choices['universe'][region][0]
                    config['settings']['delay'] = self.simulation_choices['delay'][region][0]
                    configs.append(config)
                    

        
        # 3. Decay values - removed to reduce resource usage (fixed to 0)
        
        # 4. Different pasteurization and nan handling
        for pasteurization in ['OFF']:
            for nan_handling in ['ON']:
                config = base_config.copy()
                config['settings'] = config['settings'].copy()
                config['settings']['pasteurization'] = pasteurization
                config['settings']['nanHandling'] = nan_handling
                configs.append(config)
        
        logger.info(f"Generated {len(configs)} simulation configurations")
        return configs

    def save_configurations_to_file(self, configs: List[Dict], filename: str = "simulation_configs.json"):
        """Save simulation configurations to a file for reference."""
        try:
            with open(filename, 'w') as f:
                json.dump(configs, f, indent=2)
            logger.info(f"Saved {len(configs)} configurations to {filename}")
        except Exception as e:
            logger.error(f"Error saving configurations to {filename}: {e}")

    def get_configuration_summary(self, configs: List[Dict]) -> Dict:
        """Get a summary of the configurations for logging."""
        summary = {
            'total_configs': len(configs),
            'regions': set(),
            'universes': set(),
            'neutralizations': set(),
            'truncations': set(),
            'decays': set()
        }
        
        for config in configs:
            settings = config.get('settings', {})
            summary['regions'].add(settings.get('region', 'unknown'))
            summary['universes'].add(settings.get('universe', 'unknown'))
            summary['neutralizations'].add(settings.get('neutralization', 'unknown'))
            summary['truncations'].add(settings.get('truncation', 'unknown'))
            summary['decays'].add(settings.get('decay', 'unknown'))
        
        # Convert sets to lists for JSON serialization
        for key in summary:
            if isinstance(summary[key], set):
                summary[key] = list(summary[key])
        
        return summary

    def test_alpha_batch(self, alpha_expressions: List[str], config_filename: str = "simulation_configs.json", target_region: str = None, max_concurrent: int = 10) -> List[Dict]:
        """Test multiple alpha expressions using multi_simulate approach with different configurations."""
        logger.info(f"Testing batch of {len(alpha_expressions)} alphas using multi_simulate with different configurations")
        
                 # Generate simulation configurations
        simulation_configs = self.generate_simulation_configurations(target_region=target_region)
        logger.info(f"Using {len(simulation_configs)} different simulation configurations")
        
        # Log configuration summary
        config_summary = self.get_configuration_summary(simulation_configs)
        logger.info(f"Configuration summary: {config_summary}")
        
        # Save configurations to file for reference
        self.save_configurations_to_file(simulation_configs, config_filename)
        
        # Calculate total simulations and limit concurrent submissions
        total_simulations = len(alpha_expressions) * len(simulation_configs)
        logger.info(f"Total simulations to submit: {total_simulations} ({len(alpha_expressions)} alphas × {len(simulation_configs)} configs)")
        logger.info(f"Maximum concurrent simulations: {max_concurrent}")
        
        all_results = []
        progress_urls = []
        alpha_mapping = {}  # Map progress URLs to alpha expressions and configs
        
        # Submit simulations with proper concurrent throttling and error handling
        active_count = 0  # Track active simulations
        simulation_queue = []  # Queue for simulations that couldn't be submitted due to errors
        
        for alpha_idx, alpha in enumerate(alpha_expressions):
            logger.info(f"Submitting alpha {alpha_idx + 1}/{len(alpha_expressions)} with {len(simulation_configs)} configurations")
            
            for config_idx, config in enumerate(simulation_configs):
                # Check concurrent limit BEFORE submitting
                if len(progress_urls) >= max_concurrent:
                    logger.info(f"At concurrent limit ({max_concurrent}), waiting for simulations to complete...")
                    # Wait for some simulations to complete before continuing
                    completed_results = self._monitor_pool_progress(progress_urls[:max_concurrent//2], {k: alpha_mapping[k] for k in progress_urls[:max_concurrent//2]})
                    all_results.extend(completed_results)
                    
                    # Remove completed URLs from progress_urls and alpha_mapping
                    completed_urls = [result.get('progress_url', '') for result in completed_results if result.get('progress_url')]
                    for url in completed_urls:
                        if url in progress_urls:
                            progress_urls.remove(url)
                        if url in alpha_mapping:
                            del alpha_mapping[url]
                    
                    active_count = len(progress_urls)  # Update active count
                    logger.info(f"Removed {len(completed_urls)} completed simulations, now have {active_count}/{max_concurrent} active")
                    
                    # If no simulations completed, wait longer before checking again
                    if not completed_urls:
                        logger.info("No simulations completed yet, waiting 10 seconds...")
                        sleep(10)
                    else:
                        # Small delay to avoid tight loop
                        sleep(1)
                
                # Now submit the simulation
                try:
                    # Create simulation data with current configuration
                    simulation_data = config.copy()
                    simulation_data['regular'] = alpha
                    
                    # Add config identifier for tracking
                    config_id = f"config_{config_idx}"
                    
                    logger.debug(f"Submitting alpha with config {config_idx + 1}/{len(simulation_configs)}: {config['settings']['region']}-{config['settings']['universe']}-{config['settings']['neutralization']}")
                    
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
                        error_text = simulation_response.text
                        logger.error(f"Simulation API error for alpha {alpha} with config {config_idx}: {error_text}")
                        
                        # If we hit concurrent limit or other critical errors, queue this simulation and stop submitting
                        if "CONCURRENT_SIMULATION_LIMIT_EXCEEDED" in error_text or "SIMULATION_LIMIT_EXCEEDED" in error_text:
                            logger.warning(f"Critical error detected, queuing remaining simulations and waiting for current ones to complete")
                            # Queue this simulation and all remaining ones
                            for remaining_alpha_idx in range(alpha_idx, len(alpha_expressions)):
                                for remaining_config_idx in range(config_idx if remaining_alpha_idx == alpha_idx else 0, len(simulation_configs)):
                                    remaining_alpha = alpha_expressions[remaining_alpha_idx]
                                    remaining_config = simulation_configs[remaining_config_idx]
                                    simulation_queue.append({
                                        'alpha': remaining_alpha,
                                        'config': remaining_config,
                                        'alpha_idx': remaining_alpha_idx,
                                        'config_idx': remaining_config_idx
                                    })
                            
                            logger.info(f"Queued {len(simulation_queue)} simulations, now waiting for {len(progress_urls)} active simulations to complete")
                            
                            # Wait for ALL current simulations to complete before processing queue
                            if progress_urls:
                                logger.info(f"Waiting for all {len(progress_urls)} active simulations to complete...")
                                final_results = self._monitor_pool_progress(progress_urls, alpha_mapping)
                                all_results.extend(final_results)
                                logger.info(f"All active simulations completed, now processing queued simulations")
                            
                                                                                     # Process the queue with retry logic and failure handling
                            if simulation_queue:
                                logger.info(f"Processing {len(simulation_queue)} queued simulations...")
                                failed_sims = []
                                max_retries = 3
                                retry_count = 0
                                
                                while simulation_queue and retry_count < max_retries:
                                    logger.info(f"Queue processing attempt {retry_count + 1}/{max_retries}, {len(simulation_queue)} simulations remaining")
                                    
                                    for queued_sim in simulation_queue[:]:  # Copy to avoid modification during iteration
                                        # Check if we can submit now
                                        if len(progress_urls) >= max_concurrent:
                                            # Wait for some simulations to complete
                                            completed_results = self._monitor_pool_progress(progress_urls[:max_concurrent//2], {k: alpha_mapping[k] for k in progress_urls[:max_concurrent//2]})
                                            all_results.extend(completed_results)
                                            
                                            # Remove completed URLs
                                            completed_urls = [result.get('progress_url', '') for result in completed_results if result.get('progress_url')]
                                            for url in completed_urls:
                                                if url in progress_urls:
                                                    progress_urls.remove(url)
                                                if url in alpha_mapping:
                                                    del alpha_mapping[url]
                                        
                                        # Submit the queued simulation
                                        try:
                                            simulation_data = queued_sim['config'].copy()
                                            simulation_data['regular'] = queued_sim['alpha']
                                            config_id = f"config_{queued_sim['config_idx']}"
                                            
                                            simulation_response = self.sess.post('https://api.worldquantbrain.com/simulations', json=simulation_data)
                                            
                                            if simulation_response.status_code == 201:
                                                simulation_progress_url = simulation_response.headers.get('Location')
                                                if simulation_progress_url:
                                                    progress_urls.append(simulation_progress_url)
                                                    alpha_mapping[simulation_progress_url] = {
                                                        'alpha': queued_sim['alpha'],
                                                        'config': queued_sim['config'],
                                                        'config_id': config_id
                                                    }
                                                    active_count = len(progress_urls)
                                                    logger.info(f"Submitted queued simulation {active_count}/{max_concurrent} concurrent (alpha {queued_sim['alpha_idx'] + 1}, config {queued_sim['config_idx'] + 1})")
                                                    sleep(0.5)
                                                    # Remove from queue on success
                                                    simulation_queue.remove(queued_sim)
                                                else:
                                                    logger.error(f"No Location header for queued simulation")
                                                    failed_sims.append(queued_sim)
                                            else:
                                                error_text = simulation_response.text
                                                
                                                # If it's a concurrent limit error, keep it in queue for retry
                                                if "CONCURRENT_SIMULATION_LIMIT_EXCEEDED" in error_text:
                                                    # Don't add to failed_sims, keep in queue
                                                    pass
                                                else:
                                                    # Other errors, mark as failed
                                                    logger.error(f"Failed to submit queued simulation: {error_text}")
                                                    failed_sims.append(queued_sim)
                                        
                                        except Exception as e:
                                            logger.error(f"Error submitting queued simulation: {str(e)}")
                                            failed_sims.append(queued_sim)
                                    
                                    # Remove failed simulations from queue
                                    for failed_sim in failed_sims:
                                        if failed_sim in simulation_queue:
                                            simulation_queue.remove(failed_sim)
                                    failed_sims.clear()
                                    
                                    # If we still have simulations in queue, wait before retry with exponential backoff
                                    if simulation_queue:
                                        retry_count += 1
                                        if retry_count < max_retries:
                                            wait_time = 30 * (2 ** (retry_count - 1))  # 30s, 60s, 120s
                                            logger.info(f"Waiting {wait_time} seconds before retry {retry_count + 1}...")
                                            sleep(wait_time)
                                        else:
                                            logger.warning(f"Max retries reached ({max_retries}), giving up on {len(simulation_queue)} remaining simulations")
                                            logger.warning(f"Failed simulations: {[f'alpha {sim['alpha_idx']+1}, config {sim['config_idx']+1}' for sim in simulation_queue]}")
                                
                                # Clear the queue after processing
                                simulation_queue.clear()
                            
                            # Break out of the nested loops since we've handled everything
                            break
                        
                        continue
                    
                    simulation_progress_url = simulation_response.headers.get('Location')
                    if not simulation_progress_url:
                        logger.error(f"No Location header in response for alpha {alpha} with config {config_idx}")
                        continue
                    
                    progress_urls.append(simulation_progress_url)
                    alpha_mapping[simulation_progress_url] = {
                        'alpha': alpha,
                        'config': config,
                        'config_id': config_id
                    }
                    
                    active_count = len(progress_urls)  # Update active count
                    logger.info(f"Submitted simulation {active_count}/{max_concurrent} concurrent (alpha {alpha_idx + 1}, config {config_idx + 1})")
                    
                    # Show current concurrent count every 5 simulations
                    if active_count % 5 == 0:
                        logger.info(f"=== CONCURRENT STATUS === Currently running {active_count}/{max_concurrent} simulations")
                    
                    # Small delay between submissions to avoid overwhelming the API
                    sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Error submitting alpha {alpha} with config {config_idx}: {str(e)}")
                    continue
            
            # If we broke out of the inner loop due to critical errors, also break out of outer loop
            if simulation_queue:
                break
            
        # Monitor all simulations
        if progress_urls:
            logger.info(f"Monitoring {len(progress_urls)} simulations...")
            all_results = self._monitor_pool_progress(progress_urls, alpha_mapping)
            logger.info(f"All simulations completed with {len(all_results)} successful results")
        
        logger.info(f"Multi-simulate batch complete: {len(all_results)} successful simulations")
        return all_results

    def _monitor_pool_progress(self, progress_urls: List[str], alpha_mapping: Dict[str, Dict]) -> List[Dict]:
        """Monitor progress for a pool of simulations and return completed ones."""
        results = []
        max_wait_time = 600  # 10 minutes maximum wait time for this batch
        start_time = time.time()
        
        # Create a copy to avoid modifying the original list
        urls_to_check = progress_urls.copy()
        
        logger.info(f"=== MONITORING START === Monitoring {len(urls_to_check)} simulations")
        logger.info(f"Simulation URLs to monitor: {urls_to_check[:3]}{'...' if len(urls_to_check) > 3 else ''}")
        
        while urls_to_check and (time.time() - start_time) < max_wait_time:
            elapsed_time = time.time() - start_time
            logger.info(f"=== MONITORING CYCLE === Elapsed: {elapsed_time:.1f}s, Active: {len(urls_to_check)} simulations")
            
            completed_urls = []
            for progress_url in urls_to_check[:]:  # Use slice copy to avoid modification during iteration
                try:
                    # Check simulation status
                    logger.debug(f"Checking status for: {progress_url}")
                    sim_progress_resp = self.sess.get(progress_url)
                    
                    # Handle rate limits
                    if sim_progress_resp.status_code == 429:
                        retry_after = sim_progress_resp.headers.get("Retry-After", "60")
                        logger.info(f"Rate limit hit, waiting {retry_after} seconds...")
                        time.sleep(int(float(retry_after)))
                        continue
                    
                    if sim_progress_resp.status_code != 200:
                        logger.error(f"Failed to check progress for {progress_url}: HTTP {sim_progress_resp.status_code}")
                        logger.error(f"Response headers: {dict(sim_progress_resp.headers)}")
                        logger.error(f"Response body: {sim_progress_resp.text[:200]}")
                        completed_urls.append(progress_url)
                        continue
                    
                    # Handle empty response
                    if not sim_progress_resp.text.strip():
                        logger.debug(f"Empty response for {progress_url}, simulation still initializing...")
                        continue
                    
                    # Try to parse JSON response
                    try:
                        sim_result = sim_progress_resp.json()
                        logger.debug(f"Raw progress response for {progress_url}: {sim_result}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to decode JSON response for {progress_url}: {e}")
                        logger.warning(f"Raw response text: {sim_progress_resp.text[:500]}")
                        continue
                    
                    # Check if this is a progress response or completed result
                    if "progress" in sim_result:
                        # This is a progress update, simulation still running
                        progress_value = sim_result.get("progress", 0)
                        alpha_info = alpha_mapping.get(progress_url, {})
                        alpha_expression = alpha_info.get('alpha', "unknown")
                        config_id = alpha_info.get('config_id', "unknown")
                        
                        logger.info(f"=== PROGRESS === Simulation {config_id} for '{alpha_expression}' is {progress_value*100:.1f}% complete")
                        logger.debug(f"Progress URL: {progress_url}")
                        # Don't mark as completed, keep monitoring
                        continue
                    
                    # If no progress field, check for status field (completed simulation)
                    status = sim_result.get("status")
                    logger.debug(f"Simulation {progress_url} status: {status}")
                    
                    if status == "COMPLETE" or status == "WARNING":
                        alpha_info = alpha_mapping.get(progress_url, {})
                        alpha_expression = alpha_info.get('alpha', "unknown")
                        config = alpha_info.get('config', {})
                        config_id = alpha_info.get('config_id', "unknown")
                        
                        logger.info(f"=== SUCCESS === Simulation completed for: {alpha_expression} with config {config_id}")
                        logger.info(f"Status: {status}, Progress URL: {progress_url}")
                        logger.info(f"Result keys: {list(sim_result.keys()) if isinstance(sim_result, dict) else 'Not a dict'}")
                        
                        # Log key result information
                        if isinstance(sim_result, dict):
                            if 'result' in sim_result:
                                result_data = sim_result['result']
                                logger.info(f"Result data type: {type(result_data)}")
                                if isinstance(result_data, dict):
                                    logger.info(f"Result keys: {list(result_data.keys())}")
                                    if 'metrics' in result_data:
                                        logger.info(f"Metrics available: {list(result_data['metrics'].keys()) if isinstance(result_data['metrics'], dict) else 'Not a dict'}")
                        
                        result_entry = {
                            "expression": alpha_expression,
                            "config": config,
                            "config_id": config_id,
                            "result": sim_result,
                            "progress_url": progress_url
                        }
                        results.append(result_entry)
                        completed_urls.append(progress_url)
                        
                        logger.info(f"Added to results: {len(results)} total completed simulations")
                        
                    elif status in ["FAILED", "ERROR"]:
                        alpha_info = alpha_mapping.get(progress_url, {})
                        alpha_expression = alpha_info.get('alpha', "unknown")
                        config_id = alpha_info.get('config_id', "unknown")
                        logger.error(f"=== FAILURE === Simulation failed for alpha: {alpha_expression} with config {config_id}")
                        logger.error(f"Status: {status}, Progress URL: {progress_url}")
                        logger.error(f"Full error response: {sim_result}")
                        completed_urls.append(progress_url)
                    
                    # Handle simulation limits
                    elif "SIMULATION_LIMIT_EXCEEDED" in sim_progress_resp.text:
                        alpha_info = alpha_mapping.get(progress_url, {})
                        alpha_expression = alpha_info.get('alpha', "unknown")
                        config_id = alpha_info.get('config_id', "unknown")
                        logger.info(f"=== LIMIT EXCEEDED === Simulation limit exceeded for alpha: {alpha_expression} with config {config_id}")
                        logger.info(f"Progress URL: {progress_url}")
                        completed_urls.append(progress_url)
                    
                    # Handle other error cases
                    elif status in ["CANCELLED", "TIMEOUT"]:
                        alpha_info = alpha_mapping.get(progress_url, {})
                        alpha_expression = alpha_info.get('alpha', "unknown")
                        config_id = alpha_info.get('config_id', "unknown")
                        logger.warning(f"=== {status.upper()} === Simulation {status.lower()} for alpha: {alpha_expression} with config {config_id}")
                        logger.warning(f"Progress URL: {progress_url}")
                        completed_urls.append(progress_url)
                    
                    # Handle unknown status
                    elif status:
                        logger.warning(f"=== UNKNOWN STATUS === Simulation has unknown status '{status}' for {progress_url}")
                        logger.warning(f"Full response: {sim_result}")
                    else:
                        logger.debug(f"=== PENDING === Simulation still pending for {progress_url}")
                        logger.debug(f"Response keys: {list(sim_result.keys()) if isinstance(sim_result, dict) else 'Not a dict'}")
                
                except Exception as e:
                    logger.error(f"=== EXCEPTION === Error monitoring progress for {progress_url}: {str(e)}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    completed_urls.append(progress_url)
            
            # Log completion summary for this cycle
            if completed_urls:
                logger.info(f"=== CYCLE SUMMARY === Completed {len(completed_urls)} simulations in this cycle")
                logger.info(f"Completed URLs: {completed_urls[:3]}{'...' if len(completed_urls) > 3 else ''}")
                logger.info(f"Results collected so far: {len(results)}")
            else:
                logger.info(f"=== CYCLE SUMMARY === No simulations completed in this cycle")
            
            # Remove completed URLs from our local copy
            for url in completed_urls:
                if url in urls_to_check:
                    urls_to_check.remove(url)
                    logger.debug(f"Removed completed URL from monitoring: {url}")
            
            # If we found some completed simulations, return them immediately
            if results:
                logger.info(f"=== RETURNING RESULTS === Found {len(results)} completed simulations, returning them immediately")
                logger.info(f"Result entries: {[r.get('config_id', 'unknown') for r in results]}")
                return results
            
            # Wait before next check (shorter wait for more responsive monitoring)
            if urls_to_check:
                logger.debug(f"Waiting 5 seconds before next monitoring cycle... ({len(urls_to_check)} simulations still active)")
                sleep(5)
        
        if urls_to_check:
            logger.warning(f"=== MONITORING TIMEOUT === Monitoring timeout reached after {max_wait_time}s")
            logger.warning(f"Still pending: {len(urls_to_check)} simulations")
            logger.warning(f"Pending URLs: {urls_to_check[:3]}{'...' if len(urls_to_check) > 3 else ''}")
        else:
            logger.info(f"=== MONITORING COMPLETE === All {len(progress_urls)} simulations processed")
        
        logger.info(f"=== FINAL SUMMARY === Returning {len(results)} results")
        return results

    def test_alpha(self, alpha_expression: str) -> Dict:
        """Test a single alpha expression (legacy method for backward compatibility)."""
        logger.info(f"Testing single alpha: {alpha_expression}")
        results = self.test_alpha_batch([alpha_expression])
        if results:
            return {"status": "success", "result": results[0]["result"]}
        else:
            return {"status": "error", "message": "Simulation failed"}

    def test_original_expression_with_configs(self, alpha_expression: str, config_filename: str = "simulation_configs.json", target_region: str = None, max_concurrent: int = 10) -> List[Dict]:
        """Test the original alpha expression with different simulation configurations only."""
        logger.info(f"Testing original expression '{alpha_expression}' with different configurations")
        
        # Generate simulation configurations
        simulation_configs = self.generate_simulation_configurations(target_region=target_region)
        logger.info(f"Using {len(simulation_configs)} different simulation configurations")
        
        # Log configuration summary
        config_summary = self.get_configuration_summary(simulation_configs)
        logger.info(f"Configuration summary: {config_summary}")
        
        # Save configurations to file for reference
        self.save_configurations_to_file(simulation_configs, config_filename)
        
        # Test the single expression with all configurations
        results = self.test_alpha_batch([alpha_expression], 
                                      config_filename=config_filename, target_region=target_region, max_concurrent=max_concurrent)
        
        logger.info(f"Successfully tested original expression with {len(results)} configurations")
        return results

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

    parser.add_argument('--save-configs', type=str, default='simulation_configs.json',
                      help='File to save simulation configurations (default: simulation_configs.json)')
    parser.add_argument('--max-concurrent', type=int, default=10,
                      help='Maximum concurrent simulations allowed (default: 10 to avoid API limits)')

    parser.add_argument('--region', type=str, default=None,
                      help='Specify a target region to focus on (e.g., USA, GLB, EUR, ASI, CHN). If specified, will generate ALL possible combinations for that region.')
    parser.add_argument('--skip-parameter-traverse', action='store_true',
                      help='Skip parameter traversal and only test the original expression with different configs')
    parser.add_argument('--change-configs-only', action='store_true',
                      help='Only change simulation configurations without varying parameters (implies --skip-parameter-traverse)')
    
    args = parser.parse_args()
    
    # Update log level if specified
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # If change-configs-only is specified, automatically enable skip-parameter-traverse
    if args.change_configs_only:
        args.skip_parameter_traverse = True
    
    logger.info(f"Starting alpha expression mining with parameters:")
    logger.info(f"Expression: {args.expression}")
    logger.info(f"Output file: {args.output}")
    
    logger.info(f"Max concurrent simulations: {args.max_concurrent}")

    if args.region:
        logger.info(f"Target region: {args.region}")
    if args.skip_parameter_traverse:
        logger.info("Parameter traversal: SKIPPED")
    if args.change_configs_only:
        logger.info("Mode: CHANGE CONFIGS ONLY")
    
    miner = AlphaExpressionMiner(args.credentials)
    
    # Parse expression and get parameters
    parameters = miner.parse_expression(args.expression)
    
    # Check if we should skip parameter traversal
    if args.skip_parameter_traverse or not parameters:
        if not parameters:
            logger.info("No parameters found in expression - skipping parameter traversal")
        else:
            logger.info("User chose to skip parameter traversal")
        
        # Test only the original expression with different configurations
        logger.info(f"Testing original expression with different simulation configurations")
        results = miner.test_original_expression_with_configs(args.expression, config_filename=args.save_configs, target_region=args.region, max_concurrent=args.max_concurrent)
        logger.info(f"Successfully tested original expression with {len(results)} configurations")
        
        # Save results
        output_file = args.output_file if hasattr(args, 'output_file') else args.output
        logger.info(f"Saving {len(results)} results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Remove the mined alpha from hopeful_alphas.json after completion
        logger.info("Mining completed, removing alpha from hopeful_alphas.json")
        removed = miner.remove_alpha_from_hopeful(args.expression)
        if removed:
            logger.info(f"Successfully removed alpha '{args.expression}' from hopeful_alphas.json")
        else:
            logger.warning(f"Could not remove alpha '{args.expression}' from hopeful_alphas.json (may not exist)")
        
        logger.info("Mining complete")
        return
    
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
    
    # Test variations using multi_simulate with different configurations
    logger.info(f"Testing {len(variations)} variations using multi_simulate with different configs")
    results = miner.test_alpha_batch(variations, config_filename=args.save_configs, target_region=args.region, max_concurrent=args.max_concurrent)
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
