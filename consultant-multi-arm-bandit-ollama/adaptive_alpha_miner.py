import argparse
import requests
import json
import os
import time
import logging
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from requests.auth import HTTPBasicAuth
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from datetime import datetime, timedelta
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('adaptive_alpha_miner.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SimulationSettings:
    """Configuration for simulation parameters."""
    region: str = "USA"
    universe: str = "TOP3000"
    instrumentType: str = "EQUITY"
    delay: int = 1  # delay=0 and delay=1 provide different data fields
    decay: int = 0
    neutralization: str = "INDUSTRY"
    truncation: float = 0.08
    pasteurization: str = "ON"
    unitHandling: str = "VERIFY"
    nanHandling: str = "OFF"
    maxTrade: str = "OFF"
    language: str = "FASTEXPR"
    visualization: bool = False
    testPeriod: str = "P5Y0M0D"


@dataclass
class AlphaResult:
    """Result of an alpha simulation."""
    alpha_id: str
    expression: str
    settings: SimulationSettings
    sharpe: float
    fitness: float
    turnover: float
    returns: float
    drawdown: float
    margin: float
    longCount: int
    shortCount: int
    timestamp: float
    success: bool = True
    error_message: str = ""



class MultiArmBandit:
    """Multi-arm bandit for optimizing simulation settings."""
    
    def __init__(self, exploration_rate: float = 0.1, learning_rate: float = 0.01):
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.arms = {}  # Settings -> (reward_sum, count, avg_reward)
        self.total_pulls = 0
        
    def add_arm(self, settings: SimulationSettings):
        """Add a new arm (settings configuration)."""
        key = self._settings_to_key(settings)
        if key not in self.arms:
            self.arms[key] = {'reward_sum': 0, 'count': 0, 'avg_reward': 0, 'settings': settings}
    
    def select_arm(self) -> SimulationSettings:
        """Select an arm using epsilon-greedy strategy."""
        if random.random() < self.exploration_rate:
            # Exploration: select random arm
            key = random.choice(list(self.arms.keys()))
            return self.arms[key]['settings']
        else:
            # Exploitation: select best arm
            best_key = max(self.arms.keys(), 
                          key=lambda k: self.arms[k]['avg_reward'] if self.arms[k]['count'] > 0 else 0)
            return self.arms[best_key]['settings']
    
    def update_reward(self, settings: SimulationSettings, reward: float):
        """Update the reward for an arm."""
        key = self._settings_to_key(settings)
        if key in self.arms:
            arm = self.arms[key]
            arm['count'] += 1
            arm['reward_sum'] += reward
            arm['avg_reward'] = arm['reward_sum'] / arm['count']
            self.total_pulls += 1
    
    def get_best_arm(self) -> Tuple[SimulationSettings, float]:
        """Get the best performing arm."""
        if not self.arms:
            return None, 0
        
        best_key = max(self.arms.keys(), 
                      key=lambda k: self.arms[k]['avg_reward'] if self.arms[k]['count'] > 0 else 0)
        return self.arms[best_key]['settings'], self.arms[best_key]['avg_reward']
    
    def _settings_to_key(self, settings: SimulationSettings) -> str:
        """Convert settings to a unique key."""
        return f"{settings.region}_{settings.universe}_{settings.instrumentType}_{settings.neutralization}_{settings.truncation}"
    
    def save_state(self, filename: str):
        """Save bandit state to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.arms, f)
    
    def load_state(self, filename: str):
        """Load bandit state from file."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.arms = pickle.load(f)

class GeneticAlgorithm:
    """Genetic algorithm for evolving alpha expressions."""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.generation = 0
        self.data_fields = []
        
    def set_data_fields(self, data_fields: List[str]):
        """Set available data fields for expression generation."""
        self.data_fields = data_fields
    
    def initialize_population(self, base_expressions: List[str]):
        """Initialize population with base expressions and variations."""
        self.population = []
        
        # Add base expressions
        for expr in base_expressions:
            self.population.append({
                'expression': expr,
                'fitness': 0,
                'sharpe': 0,
                'generation': 0
            })
        
        # Generate variations
        while len(self.population) < self.population_size:
            parent = random.choice(self.population)
            child = self._mutate_expression(parent['expression'])
            self.population.append({
                'expression': child,
                'fitness': 0,
                'sharpe': 0,
                'generation': 0
            })
    
    def evolve(self, results: List[AlphaResult]) -> List[str]:
        """Evolve population based on results."""
        # Update fitness scores
        for result in results:
            for individual in self.population:
                if individual['expression'] == result.expression:
                    individual['fitness'] = result.fitness
                    individual['sharpe'] = result.sharpe
                    break
        
        # Sort by fitness
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Keep top 20%
        elite_size = max(1, int(self.population_size * 0.2))
        elite = self.population[:elite_size]
        
        # Generate new population
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1['expression'], parent2['expression'])
            else:
                # Mutation
                parent = self._tournament_selection()
                child = self._mutate_expression(parent['expression'])
            
            new_population.append({
                'expression': child,
                'fitness': 0,
                'sharpe': 0,
                'generation': self.generation + 1
            })
        
        self.population = new_population
        self.generation += 1
        
        # Return top expressions
        return [ind['expression'] for ind in self.population[:10]]
    
    def _tournament_selection(self, tournament_size: int = 3):
        """Tournament selection."""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x['fitness'])
    
    def _crossover(self, parent1: str, parent2: str) -> str:
        """Crossover two expressions."""
        # Simple crossover: combine parts of expressions
        parts1 = parent1.split('(')
        parts2 = parent2.split('(')
        
        if len(parts1) > 1 and len(parts2) > 1:
            # Crossover at operator level
            if random.random() < 0.5:
                return parts1[0] + '(' + parts2[1]
            else:
                return parts2[0] + '(' + parts1[1]
        else:
            return parent1
    
    def _mutate_expression(self, expression: str) -> str:
        """Mutate an expression."""
        mutations = [
            self._add_operator,
            self._change_parameter,
            self._swap_operators,
            self._add_winsorize,
            self._add_ts_backfill,
            self._change_data_field
        ]
        
        mutation = random.choice(mutations)
        return mutation(expression)
    
    def _add_operator(self, expression: str) -> str:
        """Add an operator to the expression."""
        operators = ['ts_rank', 'ts_zscore', 'ts_delta', 'ts_sum', 'ts_mean', 'ts_std_dev']
        if '(' in expression:
            operator = random.choice(operators)
            return f"{operator}({expression})"
        return expression
    
    def _change_parameter(self, expression: str) -> str:
        """Change a parameter in the expression."""
        # Simple parameter change
        if 'std=4' in expression:
            return expression.replace('std=4', f'std={random.randint(2, 6)}')
        elif '60' in expression:
            return expression.replace('60', str(random.choice([30, 60, 90, 120])))
        return expression
    
    def _swap_operators(self, expression: str) -> str:
        """Swap operators in the expression."""
        swaps = [
            ('ts_rank', 'ts_zscore'),
            ('ts_sum', 'ts_mean'),
            ('vec_max', 'vec_avg'),
            ('vec_sum', 'vec_ir')
        ]
        
        for old, new in swaps:
            if old in expression:
                return expression.replace(old, new)
        return expression
    
    def _add_winsorize(self, expression: str) -> str:
        """Add winsorize wrapper."""
        if 'winsorize' not in expression:
            return f"winsorize({expression}, std=4)"
        return expression
    
    def _add_ts_backfill(self, expression: str) -> str:
        """Add ts_backfill wrapper."""
        if 'ts_backfill' not in expression:
            return f"ts_backfill({expression}, 60)"
        return expression
    
    def _change_data_field(self, expression: str) -> str:
        """Change data field in the expression."""
        if not self.data_fields:
            return expression
        
        # Find data fields in expression
        for field in self.data_fields:
            if field in expression:
                new_field = random.choice(self.data_fields)
                return expression.replace(field, new_field)
        return expression

class AdaptiveAlphaMiner:
    """Adaptive alpha miner using multi-arm bandit and genetic algorithm."""
    
    def __init__(self, credentials_path: str, ollama_url: str = "http://localhost:11434", ollama_model: str = "deepseek-r1:8b", region: str = None):
        self.sess = requests.Session()
        self.credentials_path = credentials_path
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.setup_auth(credentials_path)
        
        # Initialize components
        self.bandit = MultiArmBandit(exploration_rate=0.2, learning_rate=0.01)
        self.genetic_algo = GeneticAlgorithm(population_size=30, mutation_rate=0.15)
        
        # Performance tracking
        self.best_alpha = None
        self.best_score = 0
        self.results_history = []
        
        # Region consistency - select one region and stick with it
        if region:
            self.selected_region = region
            self.selected_universe = self._get_universe_for_region(region)
            logger.info(f"Using specified region: {self.selected_region} with universe: {self.selected_universe}")
        else:
            self.selected_region = self._select_region()
            self.selected_universe = self._get_universe_for_region(self.selected_region)
            logger.info(f"Randomly selected region: {self.selected_region} with universe: {self.selected_universe}")
        
        # Settings variations - only for the selected region
        self._initialize_settings_variations()
        
        # Load state
        self.load_state()
    
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
            raise Exception(f"Authentication failed: {response.text}")
        logger.info("Authentication successful")
    
    def _select_region(self) -> str:
        """Select a random region to use consistently throughout mining."""
        region_options = ["USA", "GLB", "EUR", "ASI", "CHN"]
        
        # Randomly select a region
        region = random.choice(region_options)
        
        logger.info(f"Randomly selected region: {region}")
        return region
    
    def _get_universe_for_region(self, region: str) -> str:
        """Get a default universe for a given region."""
        region_to_universe = {
            "USA": "TOP3000",
            "GLB": "TOP3000", 
            "EUR": "TOP2500",
            "ASI": "MINVOL1M",
            "CHN": "TOP2000U"
        }
        return region_to_universe.get(region, "TOP3000")
    
    def _get_data_fields_for_region(self, temp_generator, delay: int = 1) -> List[Dict]:
        """Get data fields specific to the selected region with specified delay."""
        all_fields = []
        
        base_params = {
            'delay': delay,
            'instrumentType': 'EQUITY',
            'limit': 20,
            'region': self.selected_region,
            'universe': self.selected_universe
        }
        
        try:
            logger.info(f"Requesting data fields for region {self.selected_region} with universe {self.selected_universe} and delay={delay}")
            
            # First, get available datasets for this region
            datasets_params = {
                'category': 'fundamental',
                'delay': delay,
                'instrumentType': 'EQUITY',
                'region': self.selected_region,
                'universe': self.selected_universe,
                'limit': 50
            }
            
            logger.info(f"Getting available datasets for region {self.selected_region}")
            datasets_response = self.sess.get('https://api.worldquantbrain.com/data-sets', params=datasets_params)
            
            if datasets_response.status_code == 200:
                datasets_data = datasets_response.json()
                available_datasets = datasets_data.get('results', [])
                logger.info(f"Found {len(available_datasets)} available datasets for region {self.selected_region}")
                
                # Extract dataset IDs
                dataset_ids = [ds.get('id') for ds in available_datasets if ds.get('id')]
                logger.info(f"Available dataset IDs: {dataset_ids}")
                
                # If no datasets found, fall back to default datasets
                if not dataset_ids:
                    logger.warning(f"No datasets found for region {self.selected_region}, using default datasets")
                    dataset_ids = ['fundamental6', 'fundamental2', 'analyst4', 'model16', 'model51', 'news12']
            else:
                logger.warning(f"Failed to get datasets for region {self.selected_region}: {datasets_response.text[:500]}")
                # Fall back to default datasets
                dataset_ids = ['fundamental6', 'fundamental2', 'analyst4', 'model16', 'model51', 'news12']
            
            # Now fetch fields from available datasets
            for dataset in dataset_ids:
                # First get the count
                params = base_params.copy()
                params['dataset.id'] = dataset
                params['limit'] = 1  # Just to get count efficiently
                
                logger.info(f"Getting field count for dataset: {dataset}")
                count_response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params)
                
                if count_response.status_code == 200:
                    count_data = count_response.json()
                    total_fields = count_data.get('count', 0)
                    logger.info(f"Total fields in {dataset}: {total_fields}")
                    
                    if total_fields > 0:
                        # Generate random offset
                        max_offset = max(0, total_fields - base_params['limit'])
                        random_offset = random.randint(0, max_offset)
                        
                        # Fetch random subset
                        params['offset'] = random_offset
                        params['limit'] = min(20, total_fields)  # Don't exceed total fields
                        
                        logger.info(f"Fetching fields for {dataset} with offset {random_offset}")
                        response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params)
                        
                        if response.status_code == 200:
                            data = response.json()
                            fields = data.get('results', [])
                            logger.info(f"Found {len(fields)} fields in {dataset}")
                            all_fields.extend(fields)
                        else:
                            logger.warning(f"Failed to fetch fields for {dataset}: {response.text[:500]}")
                else:
                    logger.warning(f"Failed to get count for {dataset}: {count_response.text[:500]}")
            
            # Remove duplicates if any
            unique_fields = {field['id']: field for field in all_fields}.values()
            logger.info(f"Total unique fields found for region {self.selected_region} with delay={delay}: {len(unique_fields)}")
            return list(unique_fields)
            
        except Exception as e:
            logger.error(f"Failed to fetch data fields for region {self.selected_region} with delay={delay}: {e}")
            return []
    
    def _initialize_settings_variations(self):
        """Initialize different settings configurations for the bandit using only the selected region."""
        base_settings = SimulationSettings()
        
        # Use only the selected region and its default universe
        region = self.selected_region
        universe = self.selected_universe
        
        neutralizations = ["INDUSTRY", "SECTOR", "MARKET", "NONE", "SLOW_AND_FAST", "FAST", "SLOW"]
        truncations = [0.05, 0.08, 0.1, 0.15]
        
        # Region-specific delay availability
        # ASI and CHN regions only support delay=1 for EQUITY
        if region in ["ASI", "CHN"]:
            delays = [1]  # Only delay 1 for ASI/CHN
        else:
            delays = [0, 1]  # Both delays for other regions
        
        # Add max trade settings for ASI and CHN regions
        max_trade_options = ["OFF"]
        if region in ["ASI", "CHN"]:
            max_trade_options = ["ON", "OFF"]  # Enable max trade for ASI and CHN
        
        for delay in delays:
        for neutralization in neutralizations:
            for truncation in truncations:
                    for max_trade in max_trade_options:
                settings = SimulationSettings(
                            region=region,
                            universe=universe,
                    instrumentType="EQUITY",
                            delay=delay,
                    neutralization=neutralization,
                    truncation=truncation,
                    maxTrade=max_trade
                )
                self.bandit.add_arm(settings)
        
        logger.info(f"Initialized {len(self.bandit.arms)} settings variations for universe {universe} in region {region}")
        
        # Validate that all arms use the correct region
        self._validate_bandit_region_consistency()
    
    def _validate_bandit_region_consistency(self):
        """Validate that all bandit arms use the correct region."""
        inconsistent_arms = []
        for key, arm_data in self.bandit.arms.items():
            settings = arm_data['settings']
            if settings.region != self.selected_region:
                inconsistent_arms.append({
                    'key': key,
                    'region': settings.region,
                    'expected_region': self.selected_region
                })
        
        if inconsistent_arms:
            logger.error(f"Found {len(inconsistent_arms)} inconsistent bandit arms:")
            for arm in inconsistent_arms:
                logger.error(f"  Arm {arm['key']}: region={arm['region']} (expected: {arm['expected_region']})")
            raise ValueError("Bandit arms are not consistent with selected region")
        else:
            logger.info("All bandit arms are consistent with selected region")
    
    def generate_alpha_expressions(self, count: int = 10) -> List[str]:
        """Generate alpha expressions using Ollama with random combinations of data fields and operators."""
        try:
            # Import and use the existing alpha generator to get data fields and operators
            from alpha_generator_ollama import AlphaGenerator
            
            # Create a temporary generator instance to access the APIs
            temp_generator = AlphaGenerator(self.credentials_path, self.ollama_url)
            
            # Get data fields based on region capabilities
            if self.selected_region in ["ASI", "CHN"]:
                # ASI and CHN regions only support delay=1
                data_fields = self._get_data_fields_for_region(temp_generator, delay=1)
                logger.info(f"Fetched {len(data_fields)} fields with delay=1 for region {self.selected_region}")
            else:
                # Other regions support both delays
                data_fields_delay_0 = self._get_data_fields_for_region(temp_generator, delay=0)
                data_fields_delay_1 = self._get_data_fields_for_region(temp_generator, delay=1)
                
                # Combine data fields from both delays, removing duplicates
                all_data_fields = data_fields_delay_0 + data_fields_delay_1
                unique_data_fields = {field['id']: field for field in all_data_fields}.values()
                data_fields = list(unique_data_fields)
                
                logger.info(f"Fetched {len(data_fields_delay_0)} fields with delay=0, {len(data_fields_delay_1)} fields with delay=1")
            
            operators = temp_generator.get_operators()
            logger.info(f"Total unique data fields: {len(data_fields)} and {len(operators)} operators")
            
            # Set data fields for genetic algorithm
            self.genetic_algo.set_data_fields([field['id'] for field in data_fields])
            
            # Extract field IDs and operator names with descriptions
            field_info = [(field['id'], field.get('description', 'No description')) for field in data_fields]
            operator_info = [(op['name'], op.get('description', 'No description')) for op in operators]
            
            # Define wrapper functions with descriptions
            wrappers = [
                ("winsorize", "Winsorizes data to specified standard deviations"),
                ("ts_backfill", "Backfills missing values in time series"),
                ("ts_forward_fill", "Forward fills missing values in time series"),
                ("ts_clean", "Cleans time series data"),
                ("ts_scale", "Scales time series data"),
                ("ts_normalize", "Normalizes time series data")
            ]
            
            # Define time periods for lookback
            lookback_periods = [5, 10, 22, 60, 120, 240, 512]
            
            # Define std values for winsorize
            std_values = [2, 3, 4, 5, 6]
            
            expressions = []
            
            for i in range(count):
                # Randomly select components from the actual API data
                selected_fields = random.sample(field_info, min(3, len(field_info)))
                selected_operators = random.sample(operator_info, min(5, len(operator_info)))
                selected_wrappers = random.sample(wrappers, min(2, len(wrappers)))
                lookback = random.choice(lookback_periods)
                std_val = random.choice(std_values)
                
                # Format fields and operators with descriptions
                fields_formatted = [f"{field_id} ({desc})" for field_id, desc in selected_fields]
                operators_formatted = [f"{op_name} ({desc})" for op_name, desc in selected_operators]
                wrappers_formatted = [f"{wrap_name} ({desc})" for wrap_name, desc in selected_wrappers]
                
                # Create prompt for Ollama
                prompt = f"""Generate 3 WorldQuant Brain alpha expressions. Return ONLY valid JSON.

Fields: {', '.join(fields_formatted)}
Operators: {', '.join(operators_formatted)}
Wrapper Functions: {', '.join(wrappers_formatted)}
Lookback: {lookback}

REQUIRED JSON FORMAT (no other text):
{{"expressions": ["expression1", "expression2", "expression3"]}}

Rules:
1. Use only provided fields and operators
2. Follow syntax: operator(field_id, lookback) or wrapper(operator(field_id, lookback))
3. Return ONLY the JSON object
4. No explanations, no markdown, no thinking

EXAMPLE:
{{"expressions": ["ts_rank(anl4_dez1afv4_est, 60)", "ts_delta(fnd6_newa1_ib, 120)", "winsorize(ts_zscore(anl4_ady_numest, 240), std=4)"]}}

JSON:"""
                
                # Call Ollama API
                ollama_response = self._call_ollama(prompt)
                
                if ollama_response and ollama_response.strip():
                    # Clean up the response to extract JSON
                    response_text = ollama_response.strip()
                    
                    # Remove thinking tags and explanations
                    if '<think>' in response_text:
                        parts = response_text.split('</think>')
                        if len(parts) > 1:
                            response_text = parts[-1].strip()
                    
                    # Extract JSON from markdown code blocks
                    if '```json' in response_text:
                        json_start = response_text.find('```json') + 7
                        json_end = response_text.find('```', json_start)
                        if json_end > json_start:
                            response_text = response_text[json_start:json_end].strip()
                    elif '```' in response_text:
                        json_start = response_text.find('```') + 3
                        json_end = response_text.find('```', json_start)
                        if json_end > json_start:
                            response_text = response_text[json_start:json_end].strip()
                    
                    # Try to find JSON object in the response
                    if '{' in response_text and '}' in response_text:
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        if json_end > json_start:
                            response_text = response_text[json_start:json_end].strip()
                    
                    # Try to parse as JSON
                    try:
                        import json
                        
                        # If response is just a single opening brace, it's malformed
                        if response_text.strip() == '{':
                            logger.warning("Received malformed JSON (just opening brace)")
                            raise json.JSONDecodeError("Malformed JSON", response_text, 0)
                        
                        parsed_response = json.loads(response_text)
                        
                        # Handle different JSON formats
                        if isinstance(parsed_response, dict):
                            if 'expressions' in parsed_response:
                                # Array format: {"expressions": ["expr1", "expr2", "expr3"]}
                                generated_expressions = parsed_response['expressions']
                            elif 'expression' in parsed_response:
                                # Single expression format: {"expression": "expr1"}
                                generated_expressions = [parsed_response['expression']]
                            else:
                                logger.warning(f"Unknown JSON format: {parsed_response}")
                                generated_expressions = []
                        elif isinstance(parsed_response, list):
                            # Direct array format: ["expr1", "expr2", "expr3"]
                            generated_expressions = parsed_response
                        else:
                            logger.warning(f"Invalid JSON format: {parsed_response}")
                            generated_expressions = []
                        
                        # Validate and add expressions
                    field_ids = [field_id for field_id, _ in selected_fields]
                        for expr in generated_expressions:
                            if isinstance(expr, str) and expr.strip():
                                expr = expr.strip()
                                # Validate that it contains at least one of our selected fields
                                if any(field_id in expr for field_id in field_ids):
                        # Additional validation - check for basic syntax
                                    if '(' in expr and ')' in expr:
                                        expressions.append(expr)
                                        logger.info(f"Generated expression: {expr}")
                        else:
                                        logger.warning(f"Generated expression has invalid syntax: {expr}")
                    else:
                                    logger.warning(f"Generated expression doesn't contain selected fields: {expr}")
                        
                        # If we got valid expressions, continue to next iteration
                        if expressions:
                            continue
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON response: {e}")
                        logger.warning(f"Raw response: {response_text}")
                        
                        # Try to fix common JSON issues
                        try:
                            # Fix single quotes to double quotes
                            fixed_json = response_text.replace("'", '"')
                            # Fix unquoted property names
                            import re
                            fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
                            # Try parsing the fixed JSON
                            parsed_response = json.loads(fixed_json)
                            
                            # Handle the fixed response
                            if isinstance(parsed_response, dict):
                                if 'expressions' in parsed_response:
                                    generated_expressions = parsed_response['expressions']
                                elif 'expression' in parsed_response:
                                    generated_expressions = [parsed_response['expression']]
                                else:
                                    generated_expressions = []
                            elif isinstance(parsed_response, list):
                                generated_expressions = parsed_response
                            else:
                                generated_expressions = []
                            
                            # Validate and add expressions
                            field_ids = [field_id for field_id, _ in selected_fields]
                            for expr in generated_expressions:
                                if isinstance(expr, str) and expr.strip():
                                    expr = expr.strip()
                                    if any(field_id in expr for field_id in field_ids):
                                        if '(' in expr and ')' in expr:
                                            expressions.append(expr)
                                            logger.info(f"Generated expression (fixed JSON): {expr}")
                            
                            if expressions:
                                continue
                                
                        except Exception as fix_error:
                            logger.warning(f"Failed to fix JSON: {fix_error}")
                    
                    # If JSON parsing failed or no valid expressions, use fallback
                        field_id, _ = random.choice(selected_fields)
                        operator_name, _ = random.choice(selected_operators)
                        fallback_expr = f"{operator_name}({field_id}, {lookback})"
                        expressions.append(fallback_expr)
                        logger.info(f"Using fallback expression {i+1}: {fallback_expr}")
                else:
                    # Fallback if Ollama fails
                    field_id, _ = random.choice(selected_fields)
                    operator_name, _ = random.choice(selected_operators)
                    fallback_expr = f"{operator_name}({field_id}, {lookback})"
                    expressions.append(fallback_expr)
                    logger.info(f"Ollama failed, using fallback expression {i+1}: {fallback_expr}")
            
            return expressions[:count]
            
        except Exception as e:
            logger.error(f"Error generating alpha expressions with Ollama: {e}")
            # Fallback to simple expressions
            fallback_expressions = []
            for i in range(count):
                field = "act_12m_eps_value"  # Default field
                operator = random.choice(["ts_delta", "ts_rank", "ts_zscore"])
                fallback_expr = f"{operator}({field}, 60)"
                fallback_expressions.append(fallback_expr)
            return fallback_expressions
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API to generate alpha expression with retry logic."""
        import requests
        
        ollama_url = getattr(self, 'ollama_url', 'http://localhost:11434')
        model = getattr(self, 'ollama_model', 'deepseek-r1:8b')
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,  # Lower temperature for more consistent output
                "top_p": 0.9,
                "num_predict": 2500,  # Increased to ensure complete response
            }
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Ollama API call attempt {attempt + 1}/{max_retries}")
                response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=3000)
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', '').strip()
                    
                    # Check if response is too short (likely incomplete)
                    if len(response_text) < 10:
                        logger.warning(f"Ollama response too short: '{response_text}'")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return None
                    
                    return response_text
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(10)  # Wait before retry
                        continue
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama API timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(30)  # Longer wait for timeout
                    continue
                return None
            except Exception as e:
                logger.error(f"Error calling Ollama API (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                return None
        
        return None
    
    def calculate_reward(self, result: AlphaResult) -> float:
        """Calculate reward based on alpha performance."""
        if not result.success:
            return 0
        
        # Multi-objective reward function
        sharpe_weight = 0.4
        fitness_weight = 0.3
        turnover_weight = 0.2
        returns_weight = 0.1
        
        # Normalize values
        sharpe_score = min(result.sharpe / 3.0, 1.0)  # Cap at 3.0
        fitness_score = min(result.fitness / 2.0, 1.0)  # Cap at 2.0
        turnover_score = 1.0 - min(result.turnover / 0.7, 1.0)  # Lower is better
        returns_score = min(result.returns / 0.5, 1.0)  # Cap at 0.5
        
        reward = (sharpe_score * sharpe_weight + 
                 fitness_score * fitness_weight + 
                 turnover_score * turnover_weight + 
                 returns_score * returns_weight)
        
        # Bonus for exceptional performance
        if result.sharpe > 2.0 and result.fitness > 1.5:
            reward *= 1.5
        
        return reward
    
    def simulate_alpha(self, expression: str, settings: SimulationSettings) -> Optional[AlphaResult]:
        """Simulate a single alpha with given settings."""
        try:
            simulation_data = {
                'type': 'REGULAR',
                'settings': asdict(settings),
                'regular': expression
            }
            
            # Submit simulation
            response = self.sess.post('https://api.worldquantbrain.com/simulations', 
                                     json=simulation_data)
            
            if response.status_code == 401:
                logger.warning("Authentication expired, refreshing...")
                self.setup_auth(self.credentials_path)
                response = self.sess.post('https://api.worldquantbrain.com/simulations', 
                                         json=simulation_data)
            
            if response.status_code != 201:
                logger.error(f"Simulation failed: {response.text}")
                return AlphaResult(
                    alpha_id="",
                    expression=expression,
                    settings=settings,
                    sharpe=0, fitness=0, turnover=0, returns=0, drawdown=0, margin=0,
                    longCount=0, shortCount=0, timestamp=time.time(), success=False,
                    error_message=response.text
                )
            
            # Get progress URL
            progress_url = response.headers.get('Location')
            if not progress_url:
                return AlphaResult(
                    alpha_id="", expression=expression, settings=settings,
                    sharpe=0, fitness=0, turnover=0, returns=0, drawdown=0, margin=0,
                    longCount=0, shortCount=0, timestamp=time.time(), success=False,
                    error_message="No progress URL"
                )
            
            # Monitor progress
            result = self._monitor_simulation(progress_url)
            if result:
                result.expression = expression
                result.settings = settings
                result.timestamp = time.time()
            
            return result
            
        except Exception as e:
            logger.error(f"Error simulating alpha: {str(e)}")
            return AlphaResult(
                alpha_id="", expression=expression, settings=settings,
                sharpe=0, fitness=0, turnover=0, returns=0, drawdown=0, margin=0,
                longCount=0, shortCount=0, timestamp=time.time(), success=False,
                error_message=str(e)
            )
    
    def _monitor_simulation(self, progress_url: str) -> Optional[AlphaResult]:
        """Monitor simulation progress."""
        max_wait_time = 1800  # 30 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = self.sess.get(progress_url)
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status')
                    
                    if status == 'COMPLETE':
                        alpha_id = data.get('alpha')
                        is_data = data.get('is', {})
                        
                        return AlphaResult(
                            alpha_id=alpha_id,
                            expression="",  # Will be set by caller
                            settings=None,  # Will be set by caller
                            sharpe=is_data.get('sharpe', 0),
                            fitness=is_data.get('fitness', 0),
                            turnover=is_data.get('turnover', 0),
                            returns=is_data.get('returns', 0),
                            drawdown=is_data.get('drawdown', 0),
                            margin=is_data.get('margin', 0),
                            longCount=is_data.get('longCount', 0),
                            shortCount=is_data.get('shortCount', 0),
                            timestamp=time.time()
                        )
                    
                    elif status in ['FAILED', 'ERROR']:
                        logger.error(f"Simulation failed: {data.get('message', 'Unknown error')}")
                        return None
                
                # Wait before next check
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error monitoring simulation: {str(e)}")
                time.sleep(10)
        
        logger.warning("Simulation monitoring timed out")
        return None
    
    def multi_simulate_alpha_batch(self, expressions: List[str], settings: SimulationSettings) -> List[Optional[AlphaResult]]:
        """Simulate multiple alphas in parallel using multi-simulate functionality."""
        try:
            # Create simulation data for each expression
            simulation_data_list = []
            for expression in expressions:
                sim_data = {
                    'type': 'REGULAR',
                    'settings': asdict(settings),
                    'regular': expression
                }
                simulation_data_list.append(sim_data)
            
            # Submit all simulations
            progress_urls = []
            for i, sim_data in enumerate(simulation_data_list):
                try:
                    response = self.sess.post('https://api.worldquantbrain.com/simulations', 
                                             json=sim_data)
                    
                    if response.status_code == 401:
                        logger.warning("Authentication expired, refreshing...")
                        self.setup_auth(self.credentials_path)
                        response = self.sess.post('https://api.worldquantbrain.com/simulations', 
                                                 json=sim_data)
                    
                    if response.status_code != 201:
                        logger.error(f"Simulation {i+1} failed: {response.text}")
                        progress_urls.append(None)
                        continue
                    
                    progress_url = response.headers.get('Location')
                    if not progress_url:
                        logger.error(f"No progress URL for simulation {i+1}")
                        progress_urls.append(None)
                        continue
                    
                    progress_urls.append(progress_url)
                    logger.info(f"Submitted simulation {i+1}, got progress URL: {progress_url}")
                    
                except Exception as e:
                    logger.error(f"Error submitting simulation {i+1}: {str(e)}")
                    progress_urls.append(None)
                    continue
            
            # Monitor all progress URLs
            results = self._monitor_multi_progress(progress_urls, expressions, settings)
            return results
            
        except Exception as e:
            logger.error(f"Error in multi-simulate batch: {str(e)}")
            return [None] * len(expressions)
    
    def _monitor_multi_progress(self, progress_urls: List[Optional[str]], expressions: List[str], settings: SimulationSettings) -> List[Optional[AlphaResult]]:
        """Monitor multiple simulation progress URLs."""
        results = [None] * len(progress_urls)
        max_wait_time = 1800  # 30 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            all_complete = True
            
            for i, progress_url in enumerate(progress_urls):
                if progress_url is None or results[i] is not None:
                    continue
                
                try:
                    response = self.sess.get(progress_url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        status = data.get('status')
                        
                        if status == 'COMPLETE':
                            alpha_id = data.get('alpha')
                            is_data = data.get('is', {})
                            
                            results[i] = AlphaResult(
                                alpha_id=alpha_id,
                                expression=expressions[i],
                                settings=settings,
                                sharpe=is_data.get('sharpe', 0),
                                fitness=is_data.get('fitness', 0),
                                turnover=is_data.get('turnover', 0),
                                returns=is_data.get('returns', 0),
                                drawdown=is_data.get('drawdown', 0),
                                margin=is_data.get('margin', 0),
                                longCount=is_data.get('longCount', 0),
                                shortCount=is_data.get('shortCount', 0),
                                timestamp=time.time()
                            )
                            logger.info(f"Simulation {i+1} completed successfully")
                            
                        elif status in ['FAILED', 'ERROR']:
                            logger.error(f"Simulation {i+1} failed: {data.get('message', 'Unknown error')}")
                            results[i] = AlphaResult(
                                alpha_id="",
                                expression=expressions[i],
                                settings=settings,
                                sharpe=0, fitness=0, turnover=0, returns=0, drawdown=0, margin=0,
                                longCount=0, shortCount=0, timestamp=time.time(), success=False,
                                error_message=data.get('message', 'Unknown error')
                            )
                        else:
                            all_complete = False
                    
                except Exception as e:
                    logger.error(f"Error monitoring simulation {i+1}: {str(e)}")
                    all_complete = False
            
            if all_complete:
                logger.info("All simulations completed")
                break
            
            # Wait before next check
            time.sleep(5)
        
        # Fill in any remaining None results with failed results
        for i, result in enumerate(results):
            if result is None:
                results[i] = AlphaResult(
                    alpha_id="",
                    expression=expressions[i],
                    settings=settings,
                    sharpe=0, fitness=0, turnover=0, returns=0, drawdown=0, margin=0,
                    longCount=0, shortCount=0, timestamp=time.time(), success=False,
                    error_message="Simulation timed out"
                )
        
        return results
    
    def mine_adaptive_batch(self, batch_size: int = 5) -> List[AlphaResult]:
        """Mine a batch of alphas using adaptive strategies with multi-simulate."""
        results = []
        
        # Generate expressions
        expressions = self.generate_alpha_expressions(batch_size)
        
        # Group expressions by settings for multi-simulate
        settings_groups = {}
        for expression in expressions:
            # Select settings using bandit
            settings = self.bandit.select_arm()
            settings_key = self.bandit._settings_to_key(settings)
            
            if settings_key not in settings_groups:
                settings_groups[settings_key] = {
                    'settings': settings,
                    'expressions': []
                }
            settings_groups[settings_key]['expressions'].append(expression)
        
        # Process each settings group with multi-simulate
        for settings_key, group in settings_groups.items():
            settings = group['settings']
            expressions_batch = group['expressions']
            
            logger.info(f"Multi-simulating {len(expressions_batch)} expressions with settings: {settings.region}/{settings.universe}/{settings.neutralization}")
            
            # Use multi-simulate for this batch
            batch_results = self.multi_simulate_alpha_batch(expressions_batch, settings)
            
            # Process results
            for i, result in enumerate(batch_results):
                if result and result.success:
                    # Calculate reward and update bandit
                    reward = self.calculate_reward(result)
                    self.bandit.update_reward(settings, reward)
                    
                    # Update best alpha
                    score = result.sharpe * result.fitness
                    if score > self.best_score:
                        self.best_score = score
                        self.best_alpha = result
                        logger.info(f"New best alpha! Score: {score:.3f}, Sharpe: {result.sharpe:.3f}, Fitness: {result.fitness:.3f}")
                    
                    # Add to history
                    self.results_history.append(result)
                    results.append(result)
                    
                    logger.info(f"Alpha {i+1} completed - Sharpe: {result.sharpe:.3f}, Fitness: {result.fitness:.3f}, Reward: {reward:.3f}")
                else:
                    logger.warning(f"Alpha {i+1} simulation failed: {expressions_batch[i][:50]}...")
        
        return results
    
    def lateral_movement(self, base_alpha: AlphaResult, movement_count: int = 5) -> List[AlphaResult]:
        """Perform lateral movement to find variations of a good alpha using multi-simulate."""
        results = []
        base_expression = base_alpha.expression
        
        # Generate lateral variations
        variations = self._generate_lateral_variations(base_expression, movement_count)
        
        # Test variations with same settings using multi-simulate
        settings = base_alpha.settings
        
        logger.info(f"Lateral movement: multi-simulating {len(variations)} variations...")
        
        # Use multi-simulate for lateral variations
        batch_results = self.multi_simulate_alpha_batch(variations, settings)
        
        for i, result in enumerate(batch_results):
            if result and result.success:
                results.append(result)
                
                # Update best if better
                score = result.sharpe * result.fitness
                if score > self.best_score:
                    self.best_score = score
                    self.best_alpha = result
                    logger.info(f"Lateral movement found better alpha! Score: {score:.3f}")
        
        return results
    
    def _generate_lateral_variations(self, base_expression: str, count: int) -> List[str]:
        """Generate lateral variations of an expression."""
        variations = []
        
        # Parameter variations
        if 'std=4' in base_expression:
            for std in [2, 3, 5, 6]:
                variations.append(base_expression.replace('std=4', f'std={std}'))
        
        if '60' in base_expression:
            for days in [30, 90, 120]:
                variations.append(base_expression.replace('60', str(days)))
        
        # Operator variations
        if 'vec_max' in base_expression:
            variations.append(base_expression.replace('vec_max', 'vec_avg'))
            variations.append(base_expression.replace('vec_max', 'vec_sum'))
        
        if 'ts_delta' in base_expression:
            variations.append(base_expression.replace('ts_delta', 'ts_rank'))
            variations.append(base_expression.replace('ts_delta', 'ts_zscore'))
        
        # Add winsorize if not present
        if 'winsorize' not in base_expression:
            variations.append(f"winsorize({base_expression}, std=4)")
        
        # Add ts_backfill if not present
        if 'ts_backfill' not in base_expression:
            variations.append(f"ts_backfill({base_expression}, 60)")
        
        return variations[:count]
    
    def process_hopeful_alphas(self, hopeful_alphas_file: str = 'hopeful_alphas.json', count: int = 10) -> List[AlphaResult]:
        """Process hopeful alphas from external file using consistent universe settings."""
        if not os.path.exists(hopeful_alphas_file):
            logger.warning(f"Hopeful alphas file {hopeful_alphas_file} not found")
            return []
        
        try:
            with open(hopeful_alphas_file, 'r') as f:
                hopeful_data = json.load(f)
            
            # Sort by fitness and take top alphas
            hopeful_data.sort(key=lambda x: x.get('fitness', 0), reverse=True)
            selected_alphas = hopeful_data[:count]
            
            logger.info(f"Processing {len(selected_alphas)} hopeful alphas with region {self.selected_region}")
            
            results = []
            for alpha_data in selected_alphas:
                expression = alpha_data.get('expression', '')
                if not expression:
                    continue
                
                # Use bandit to select optimal settings for this alpha
                settings = self.bandit.select_arm()
                
                # Simulate the alpha with selected settings
                result = self.simulate_alpha(expression, settings)
                if result and result.success:
                    # Update bandit with reward
                    reward = self.calculate_reward(result)
                    self.bandit.update_reward(settings, reward)
                    
                    # Update best alpha if better
                    score = result.sharpe * result.fitness
                    if score > self.best_score:
                        self.best_score = score
                        self.best_alpha = result
                        logger.info(f"Hopeful alpha became new best! Score: {score:.3f}")
                    
                    results.append(result)
                    logger.info(f"Processed hopeful alpha - Sharpe: {result.sharpe:.3f}, Fitness: {result.fitness:.3f}")
                else:
                    logger.warning(f"Failed to simulate hopeful alpha: {expression[:50]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing hopeful alphas: {e}")
            return []
    
    def submit_best_alpha(self) -> bool:
        """Submit the best alpha found."""
        if not self.best_alpha:
            logger.warning("No best alpha to submit")
            return False
        
        try:
            # Get alpha details
            alpha_id = self.best_alpha.alpha_id
            if not alpha_id:
                logger.error("No alpha ID for best alpha")
                return False
            
            # Submit alpha
            submit_data = {
                "status": "SUBMITTED"
            }
            
            response = self.sess.patch(f"https://api.worldquantbrain.com/alphas/{alpha_id}", 
                                     json=submit_data)
            
            if response.status_code == 200:
                logger.info(f"Successfully submitted alpha {alpha_id}")
                logger.info(f"Best alpha - Sharpe: {self.best_alpha.sharpe:.3f}, Fitness: {self.best_alpha.fitness:.3f}")
                return True
            else:
                logger.error(f"Failed to submit alpha: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting alpha: {str(e)}")
            return False
    
    def save_state(self):
        """Save current state."""
        self.bandit.save_state('bandit_state.pkl')
        
        state = {
            'best_alpha': asdict(self.best_alpha) if self.best_alpha else None,
            'best_score': self.best_score,
            'results_history': [asdict(r) for r in self.results_history[-100:]],  # Keep last 100
            'genetic_algo': {
                'population': self.genetic_algo.population,
                'generation': self.genetic_algo.generation
            },
            'selected_region': self.selected_region,
            'selected_universe': self.selected_universe
        }
        
        with open('adaptive_miner_state.json', 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self):
        """Load saved state."""
        try:
            self.bandit.load_state('bandit_state.pkl')
            
            if os.path.exists('adaptive_miner_state.json'):
                with open('adaptive_miner_state.json', 'r') as f:
                    state = json.load(f)
                
                if state.get('best_alpha'):
                    self.best_alpha = AlphaResult(**state['best_alpha'])
                    self.best_score = state.get('best_score', 0)
                
                # Load results history
                self.results_history = []
                for r_data in state.get('results_history', []):
                    self.results_history.append(AlphaResult(**r_data))
                
                # Load genetic algorithm state
                ga_state = state.get('genetic_algo', {})
                self.genetic_algo.population = ga_state.get('population', [])
                self.genetic_algo.generation = ga_state.get('generation', 0)
                
                # Load universe information if available
                if 'selected_region' in state and 'selected_universe' in state:
                    self.selected_region = state['selected_region']
                    self.selected_universe = state['selected_universe']
                    logger.info(f"Restored region: {self.selected_region} with universe: {self.selected_universe}")
                
                logger.info(f"Loaded state - Best score: {self.best_score}")
                
        except Exception as e:
            logger.warning(f"Could not load state: {e}")

def main():
    parser = argparse.ArgumentParser(description='Adaptive Alpha Miner with Multi-Arm Bandit and Genetic Algorithm')
    parser.add_argument('--credentials', type=str, default='./credential.txt',
                      help='Path to credentials file (default: ./credential.txt)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                      help='Ollama API URL (default: http://localhost:11434)')
    parser.add_argument('--ollama-model', type=str, default='deepseek-r1:8b',
                      help='Ollama model to use (default: deepseek-r1:8b)')
    parser.add_argument('--mode', type=str, choices=['mine', 'submit', 'lateral', 'hopeful'],
                      default='mine', help='Operation mode (default: mine)')
    parser.add_argument('--batch-size', type=int, default=5,
                      help='Batch size for mining (default: 5)')
    parser.add_argument('--iterations', type=int, default=10,
                      help='Number of mining iterations (default: 10)')
    parser.add_argument('--lateral-count', type=int, default=5,
                      help='Number of lateral movements (default: 5)')
    parser.add_argument('--region', type=str, default=None,
                      help='Specify region to use (e.g., USA, GLB, EUR, ASI, CHN). If not specified, will randomly select one.')
    parser.add_argument('--hopeful-file', type=str, default='hopeful_alphas.json',
                      help='Path to hopeful alphas file for processing (default: hopeful_alphas.json)')
    parser.add_argument('--hopeful-count', type=int, default=10,
                      help='Number of hopeful alphas to process (default: 10)')
    
    args = parser.parse_args()
    
    try:
        miner = AdaptiveAlphaMiner(args.credentials, args.ollama_url, args.ollama_model, args.region)
        
        if args.mode == 'mine':
            logger.info(f"Starting adaptive mining for {args.iterations} iterations")
            
            for i in range(args.iterations):
                logger.info(f"\n=== Iteration {i+1}/{args.iterations} ===")
                
                # Mine batch
                results = miner.mine_adaptive_batch(args.batch_size)
                
                # Perform lateral movement on best result
                if results:
                    best_result = max(results, key=lambda r: r.sharpe * r.fitness)
                    lateral_results = miner.lateral_movement(best_result, args.lateral_count)
                    results.extend(lateral_results)
                
                # Save state
                miner.save_state()
                
                logger.info(f"Iteration {i+1} complete - {len(results)} alphas tested")
                
                # Continue immediately to next iteration
                if i < args.iterations - 1:
                    logger.info("Continuing to next iteration immediately...")
        
        elif args.mode == 'submit':
            logger.info("Submitting best alpha...")
            success = miner.submit_best_alpha()
            if success:
                logger.info("Alpha submitted successfully!")
            else:
                logger.error("Failed to submit alpha")
        
        elif args.mode == 'lateral':
            if miner.best_alpha:
                logger.info("Performing lateral movement on best alpha...")
                results = miner.lateral_movement(miner.best_alpha, args.lateral_count)
                logger.info(f"Lateral movement complete - {len(results)} variations tested")
            else:
                logger.warning("No best alpha found for lateral movement")
        
        elif args.mode == 'hopeful':
            logger.info(f"Processing hopeful alphas from {args.hopeful_file}...")
            results = miner.process_hopeful_alphas(args.hopeful_file, args.hopeful_count)
            logger.info(f"Hopeful alpha processing complete - {len(results)} alphas tested")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
