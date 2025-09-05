#!/usr/bin/env python3
"""
Enhanced Template Generator v2 with Multi-Simulation Testing
- Improved template quality validation
- Progress saving and resume functionality
- Dynamic result display
- Better error handling and field validation
"""

import argparse
import requests
import json
import os
import random
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from requests.auth import HTTPBasicAuth
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime
import threading
import sys
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_template_generator_v2.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RegionConfig:
    """Configuration for different regions"""
    region: str
    universe: str
    delay: int
    max_trade: bool = False

@dataclass
class SimulationSettings:
    """Configuration for simulation parameters."""
    region: str = "USA"
    universe: str = "TOP3000"
    instrumentType: str = "EQUITY"
    delay: int = 1
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
class TemplateResult:
    """Result of a template simulation."""
    template: str
    region: str
    settings: SimulationSettings
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    returns: float = 0.0
    drawdown: float = 0.0
    margin: float = 0.0
    longCount: int = 0
    shortCount: int = 0
    success: bool = False
    error_message: str = ""
    timestamp: float = 0.0

class MultiArmBandit:
    """Multi-arm bandit for explore vs exploit decisions"""
    
    def __init__(self, exploration_rate: float = 0.3, confidence_level: float = 0.95):
        self.exploration_rate = exploration_rate
        self.confidence_level = confidence_level
        self.arm_stats = {}  # {arm_id: {'pulls': int, 'rewards': list, 'avg_reward': float}}
    
    def add_arm(self, arm_id: str):
        """Add a new arm to the bandit"""
        if arm_id not in self.arm_stats:
            self.arm_stats[arm_id] = {
                'pulls': 0,
                'rewards': [],
                'avg_reward': 0.0,
                'confidence_interval': (0.0, 1.0)
            }
    
    def update_arm(self, arm_id: str, reward: float):
        """Update arm statistics with new reward"""
        if arm_id not in self.arm_stats:
            self.add_arm(arm_id)
        
        stats = self.arm_stats[arm_id]
        stats['pulls'] += 1
        stats['rewards'].append(reward)
        stats['avg_reward'] = np.mean(stats['rewards'])
        
        # Calculate confidence interval
        if len(stats['rewards']) > 1:
            std_err = np.std(stats['rewards']) / math.sqrt(len(stats['rewards']))
            z_score = 1.96  # 95% confidence
            margin = z_score * std_err
            stats['confidence_interval'] = (
                max(0, stats['avg_reward'] - margin),
                min(1, stats['avg_reward'] + margin)
            )
    
    def choose_action(self, available_arms: List[str]) -> Tuple[str, str]:
        """
        Choose between explore (new template) or exploit (existing template)
        Returns: (action, arm_id)
        """
        if not available_arms:
            return "explore", "new_template"
        
        # Add any new arms
        for arm in available_arms:
            self.add_arm(arm)
        
        # Calculate upper confidence bounds
        ucb_values = {}
        for arm_id in available_arms:
            stats = self.arm_stats[arm_id]
            if stats['pulls'] == 0:
                ucb_values[arm_id] = float('inf')  # Prioritize unexplored arms
            else:
                # UCB1 formula with confidence interval
                exploration_bonus = math.sqrt(2 * math.log(sum(s['pulls'] for s in self.arm_stats.values())) / stats['pulls'])
                ucb_values[arm_id] = stats['avg_reward'] + exploration_bonus
        
        # Choose best arm based on UCB
        best_arm = max(ucb_values.keys(), key=lambda x: ucb_values[x])
        
        # Decide explore vs exploit based on exploration rate and arm performance
        if random.random() < self.exploration_rate or self.arm_stats[best_arm]['pulls'] < 3:
            return "explore", "new_template"
        else:
            return "exploit", best_arm
    
    def get_arm_performance(self, arm_id: str) -> Dict:
        """Get performance statistics for an arm"""
        if arm_id not in self.arm_stats:
            return {'pulls': 0, 'avg_reward': 0.0, 'confidence_interval': (0.0, 1.0)}
        return self.arm_stats[arm_id].copy()


class ProgressTracker:
    """Track and display progress with dynamic updates"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.total_regions = 0
        self.completed_regions = 0
        self.total_templates = 0
        self.completed_templates = 0
        self.total_simulations = 0
        self.completed_simulations = 0
        self.successful_simulations = 0
        self.failed_simulations = 0
        self.current_region = ""
        self.current_phase = ""
        self.best_sharpe = 0.0
        self.best_template = ""
        
    def update_region_progress(self, region: str, phase: str, templates: int = 0, simulations: int = 0):
        with self.lock:
            self.current_region = region
            self.current_phase = phase
            if templates > 0:
                self.total_templates += templates
            if simulations > 0:
                self.total_simulations += simulations
            self._display_progress()
    
    def update_simulation_progress(self, success: bool, sharpe: float = 0.0, template: str = ""):
        with self.lock:
            self.completed_simulations += 1
            if success:
                self.successful_simulations += 1
                if sharpe > self.best_sharpe:
                    self.best_sharpe = sharpe
                    self.best_template = template[:50] + "..." if len(template) > 50 else template
            else:
                self.failed_simulations += 1
            self._display_progress()
    
    def complete_region(self):
        with self.lock:
            self.completed_regions += 1
            self._display_progress()
    
    def _display_progress(self):
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        
        # Clear line and display progress
        print(f"\r{' ' * 100}\r", end="")
        
        if self.total_simulations > 0:
            sim_progress = (self.completed_simulations / self.total_simulations) * 100
            success_rate = (self.successful_simulations / self.completed_simulations * 100) if self.completed_simulations > 0 else 0
            
            print(f"⏱️  {elapsed_str} | 🌍 {self.current_region} ({self.completed_regions}/{self.total_regions}) | "
                  f"📊 {self.current_phase} | 🎯 Sims: {self.completed_simulations}/{self.total_simulations} "
                  f"({sim_progress:.1f}%) | ✅ {success_rate:.1f}% | 🏆 Best: {self.best_sharpe:.3f}", end="")
        else:
            print(f"⏱️  {elapsed_str} | 🌍 {self.current_region} ({self.completed_regions}/{self.total_regions}) | "
                  f"📊 {self.current_phase}", end="")
        
        sys.stdout.flush()

class EnhancedTemplateGeneratorV2:
    def __init__(self, credentials_path: str, deepseek_api_key: str, max_concurrent: int = 5, 
                 progress_file: str = "template_progress.json", results_file: str = "enhanced_results.json"):
        """Initialize the enhanced template generator with multi-simulation capabilities"""
        self.sess = requests.Session()
        self.credentials_path = credentials_path
        self.deepseek_api_key = deepseek_api_key
        self.max_concurrent = max_concurrent
        self.progress_file = progress_file
        self.results_file = results_file
        self.progress_tracker = ProgressTracker()
        self.bandit = MultiArmBandit(exploration_rate=0.3)
        self.setup_auth()
        
        # Region configurations with pyramid multipliers
        self.region_configs = {
            "USA": RegionConfig("USA", "TOP3000", 1),
            "GLB": RegionConfig("GLB", "TOP3000", 1),
            "EUR": RegionConfig("EUR", "TOP2500", 1),
            "ASI": RegionConfig("ASI", "MINVOL1M", 1, max_trade=True),
            "CHN": RegionConfig("CHN", "TOP2000U", 1, max_trade=True)
        }
        
        # Pyramid theme multipliers (delay=0, delay=1) for each region
        self.pyramid_multipliers = {
            "USA": {"0": 1.8, "1": 1.2},  # delay=0 has higher multiplier
            "GLB": {"0": 1.0, "1": 1.5},  # delay=1 has higher multiplier
            "EUR": {"0": 1.7, "1": 1.4},  # delay=0 has higher multiplier
            "ASI": {"0": 1.0, "1": 1.5},  # delay=1 has higher multiplier (delay=0 not available)
            "CHN": {"0": 1.0, "1": 1.8}   # delay=1 has higher multiplier (delay=0 not available)
        }
        
        # Load operators and data fields
        self.operators = self.load_operators()
        self.data_fields = {}
        
        # Results storage
        self.template_results = []
        self.all_results = {
            'metadata': {
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_operators': len(self.operators),
                'regions': [],
                'templates_per_region': 0,
                'version': '2.0'
            },
            'templates': {},
            'simulation_results': {}
        }
    
    def select_optimal_delay(self, region: str) -> int:
        """Select delay based on pyramid multipliers and region constraints"""
        multipliers = self.pyramid_multipliers.get(region, {"0": 1.0, "1": 1.0})
        
        # For ASI and CHN, only delay=1 is available
        if region in ["ASI", "CHN"]:
            return 1
        
        # For other regions, use weighted selection based on multipliers
        delay_0_mult = multipliers.get("0", 1.0)
        delay_1_mult = multipliers.get("1", 1.0)
        
        # Calculate probabilities based on multipliers
        total_weight = delay_0_mult + delay_1_mult
        prob_delay_0 = delay_0_mult / total_weight
        prob_delay_1 = delay_1_mult / total_weight
        
        # Weighted random selection
        if random.random() < prob_delay_0:
            selected_delay = 0
        else:
            selected_delay = 1
        
        logger.info(f"Selected delay {selected_delay} for {region} (multipliers: 0={delay_0_mult}, 1={delay_1_mult}, prob_0={prob_delay_0:.2f})")
        return selected_delay
    
    def _collect_failure_patterns(self, failed_results: List[TemplateResult], region: str):
        """Collect failure patterns to help LLM learn from mistakes"""
        if not hasattr(self, 'failure_patterns'):
            self.failure_patterns = {}
        
        if region not in self.failure_patterns:
            self.failure_patterns[region] = []
        
        for result in failed_results:
            failure_info = {
                'template': result.template,
                'error': result.error_message,
                'timestamp': result.timestamp
            }
            self.failure_patterns[region].append(failure_info)
        
        logger.info(f"Collected {len(failed_results)} failure patterns for {region}")
    
    def _remove_failed_templates_from_progress(self, region: str, failed_templates: List[str]):
        """Remove failed templates from progress JSON"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                # Remove failed templates from the templates section
                if 'templates' in progress_data and region in progress_data['templates']:
                    original_templates = progress_data['templates'][region]
                    # Filter out failed templates
                    remaining_templates = [
                        template for template in original_templates 
                        if template.get('template', '') not in failed_templates
                    ]
                    progress_data['templates'][region] = remaining_templates
                    
                    logger.info(f"Removed {len(original_templates) - len(remaining_templates)} failed templates from progress for {region}")
                
                # Save updated progress
                with open(self.progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to remove failed templates from progress: {e}")
        
    def setup_auth(self):
        """Setup authentication for WorldQuant Brain API"""
        try:
            with open(self.credentials_path, 'r') as f:
                credentials = json.load(f)
            
            username = credentials[0]
            password = credentials[1]
            
            # Authenticate with WorldQuant Brain
            auth_response = self.sess.post(
                'https://api.worldquantbrain.com/authentication',
                auth=HTTPBasicAuth(username, password)
            )
            
            if auth_response.status_code == 201:
                logger.info("Authentication successful")
            else:
                logger.error(f"Authentication failed: {auth_response.status_code}")
                raise Exception("Authentication failed")
                
        except Exception as e:
            logger.error(f"Failed to setup authentication: {e}")
            raise
    
    def load_operators(self) -> List[Dict]:
        """Load operators from operatorRAW.json"""
        try:
            with open('operatorRAW.json', 'r') as f:
                operators = json.load(f)
            logger.info(f"Loaded {len(operators)} operators")
            return operators
        except Exception as e:
            logger.error(f"Failed to load operators: {e}")
            return []
    
    def get_data_fields_for_region(self, region: str, delay: int = 1) -> List[Dict]:
        """Get data fields for a specific region and delay"""
        try:
            config = self.region_configs[region]
            
            # First get available datasets
            datasets_params = {
                'category': 'fundamental',
                'delay': delay,
                'instrumentType': 'EQUITY',
                'region': region,
                'universe': config.universe,
                'limit': 50
            }
            
            logger.info(f"Getting datasets for region {region}")
            datasets_response = self.sess.get('https://api.worldquantbrain.com/data-sets', params=datasets_params)
            
            if datasets_response.status_code == 200:
                datasets_data = datasets_response.json()
                available_datasets = datasets_data.get('results', [])
                dataset_ids = [ds.get('id') for ds in available_datasets if ds.get('id')]
                logger.info(f"Found {len(dataset_ids)} datasets for region {region}")
            else:
                logger.warning(f"Failed to get datasets for region {region}")
                dataset_ids = ['fundamental6', 'fundamental2', 'analyst4', 'model16', 'model51', 'news12']
            
            # Get fields from datasets
            all_fields = []
            for dataset in dataset_ids[:5]:  # Limit to first 5 datasets
                params = {
                    'dataset.id': dataset,
                    'delay': delay,
                    'instrumentType': 'EQUITY',
                    'region': region,
                    'universe': config.universe,
                    'limit': 20
                }
                
                response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params)
                if response.status_code == 200:
                    data = response.json()
                    fields = data.get('results', [])
                    all_fields.extend(fields)
                    logger.info(f"Found {len(fields)} fields in dataset {dataset}")
            
            # Remove duplicates
            unique_fields = {field['id']: field for field in all_fields}.values()
            logger.info(f"Total unique fields for region {region}: {len(unique_fields)}")
            return list(unique_fields)
            
        except Exception as e:
            logger.error(f"Failed to get data fields for region {region}: {e}")
            return []
    
    def validate_template_syntax(self, template: str, valid_fields: List[str]) -> Tuple[bool, str]:
        """Validate template syntax and field usage - more lenient approach"""
        try:
            # Check for invalid operators that cause syntax errors
            invalid_ops = ['%', '==', '!=', '&&', '||']
            for op in invalid_ops:
                if op in template:
                    return False, f"Invalid operator: {op}"
            
            # Check for balanced parentheses
            if template.count('(') != template.count(')'):
                return False, "Unbalanced parentheses"
            
            # Check for missing commas between parameters
            if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(', template):
                return False, "Missing comma between parameters"
            
            # Basic syntax check - ensure it looks like a function call
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\(', template):
                return False, "Invalid function call syntax"
            
            # Check for obvious field name issues - only check for very obvious problems
            # Look for field names that are clearly invalid (too long, weird characters)
            field_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            identifiers = re.findall(field_pattern, template)
            
            for identifier in identifiers:
                # Skip if it's a number
                try:
                    float(identifier)
                    continue
                except ValueError:
                    pass
                
                # Skip common keywords
                if identifier.lower() in ['true', 'false', 'if', 'else', 'and', 'or', 'not', 'std']:
                    continue
                
                # Check for obviously invalid identifiers (too long, weird patterns)
                if len(identifier) > 50:
                    return False, f"Identifier too long: {identifier}"
                
                # Check if this is a valid operator first
                valid_operators = [op['name'] for op in self.operators]
                if identifier in valid_operators:
                    # It's a valid operator, continue
                    continue
                
                # Check if this is a field name (should be in valid_fields)
                # Field names typically start with 'fnd', 'fn_', or are common field names
                is_likely_field = (identifier.startswith('fnd') or 
                                 identifier.startswith('fn_') or 
                                 identifier in ['close', 'open', 'high', 'low', 'volume', 'returns', 'industry', 'sector', 'cap'])
                
                if is_likely_field and identifier not in valid_fields:
                    return False, f"Unknown field: {identifier}"
                # If it's not a field and not an operator, it might be a number or other valid token
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def call_deepseek_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call DeepSeek API to generate templates"""
        headers = {
            'Authorization': f'Bearer {self.deepseek_api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in quantitative finance and WorldQuant Brain alpha expressions. Generate valid, creative alpha expression templates with proper syntax."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.9
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"DeepSeek API call attempt {attempt + 1}/{max_retries}")
                response = requests.post(
                    'https://api.deepseek.com/v1/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    logger.info("DeepSeek API call successful")
                    return content
                else:
                    logger.warning(f"DeepSeek API error: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None
                    
            except Exception as e:
                logger.error(f"DeepSeek API call failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        
        return None
    
    def generate_templates_for_region(self, region: str, num_templates: int = 10) -> List[Dict]:
        """Generate templates for a specific region with validation"""
        logger.info(f"Generating {num_templates} templates for region: {region}")
        
        # Get data fields for this region with optimal delay based on pyramid multipliers
        config = self.region_configs[region]
        optimal_delay = self.select_optimal_delay(region)
        data_fields = self.get_data_fields_for_region(region, optimal_delay)
        if not data_fields:
            logger.warning(f"No data fields found for region {region}")
            return []
        
        # Create field name list for validation
        valid_fields = [field['id'] for field in data_fields]
        logger.info(f"Available fields for {region} (delay={optimal_delay}): {len(valid_fields)} fields")
        logger.info(f"Sample fields: {valid_fields[:5]}")
        
        # Select a subset of operators and fields for template generation
        selected_operators = random.sample(self.operators, min(20, len(self.operators)))
        selected_fields = random.sample(data_fields, min(15, len(data_fields)))
        
        logger.info(f"Selected {len(selected_fields)} fields for template generation")
        
        # Create prompt for DeepSeek with better instructions
        operators_desc = []
        for op in selected_operators:
            operators_desc.append(f"- {op['name']}: {op['description']} (Definition: {op['definition']})")
        
        fields_desc = []
        for field in selected_fields:
            fields_desc.append(f"- {field['id']}: {field.get('description', 'No description')}")
        
        # Add parameter guidelines based on operator definitions
        parameter_guidelines = []
        for op in selected_operators:
            if 'd' in op['definition'] and 'd' not in parameter_guidelines:
                parameter_guidelines.append("- 'd' parameters must be positive integers (e.g., 20, 60, 120)")
            if 'constant' in op['definition'] and 'constant' not in parameter_guidelines:
                parameter_guidelines.append("- 'constant' parameters can be numbers (e.g., 0, 1, 0.5)")
            if 'std' in op['definition'] and 'std' not in parameter_guidelines:
                parameter_guidelines.append("- 'std' parameters should be positive numbers (e.g., 3, 4)")
            if 'filter' in op['definition'] and 'filter' not in parameter_guidelines:
                parameter_guidelines.append("- 'filter' parameters should be true/false")
        
        # Add failure patterns to help LLM learn
        failure_guidance = ""
        if hasattr(self, 'failure_patterns') and region in self.failure_patterns:
            recent_failures = self.failure_patterns[region][-5:]  # Last 5 failures
            if recent_failures:
                failure_guidance = f"""

PREVIOUS FAILURES TO AVOID:
{chr(10).join([f"- FAILED: {failure['template'][:60]}... ERROR: {failure['error']}" for failure in recent_failures])}

LEARN FROM THESE MISTAKES:
- Do NOT repeat the same error patterns
- Check operator parameter requirements carefully
- Ensure proper syntax and field names
- Avoid invalid parameter combinations
"""

        prompt = f"""Generate {num_templates} diverse and creative WorldQuant Brain alpha expression templates for the {region} region.

Region Configuration:
- Region: {region}
- Universe: {config.universe}
- Delay: {optimal_delay} (selected based on pyramid multiplier: {self.pyramid_multipliers[region].get(str(optimal_delay), 1.0)})
- Max Trade: {config.max_trade}

Available Operators (USE ONLY THESE):
{chr(10).join(operators_desc)}

Available Data Fields (USE ONLY THESE - These are the EXACT field names available for delay={optimal_delay}):
{chr(10).join(fields_desc)}{failure_guidance}

PARAMETER GUIDELINES:
{chr(10).join(parameter_guidelines) if parameter_guidelines else "- All parameters should be positive integers or valid numbers"}

CRITICAL REQUIREMENTS:
1. Use ONLY the provided operator names exactly as shown
2. Use ONLY the provided field names exactly as shown (these are verified for delay={optimal_delay})
3. Use proper syntax: operator(field_name, parameter) or operator(field1, field2, parameter)
4. Follow parameter guidelines above - NO decimal parameters like 4.0, 0.5 unless specifically allowed
5. NO special characters like %, ==, !=, &&, ||
6. NO missing commas between parameters
7. Balanced parentheses
8. Each template on a separate line
9. NO explanations or comments
10. NO custom operators or fields not in the lists above
11. Field names must match EXACTLY as shown in the Available Data Fields list
12. Read operator definitions carefully to understand parameter requirements
13. AVOID the failure patterns shown above - learn from previous mistakes
14. Double-check parameter counts and types for each operator

VALID EXAMPLES:
ts_rank(ts_delta(close, 1), 20)
group_normalize(ts_zscore(volume, 60), industry)
winsorize(ts_regression(returns, volume, 120), std=3)

Generate {num_templates} templates:"""

        # Call DeepSeek API
        response = self.call_deepseek_api(prompt)
        if not response:
            logger.error(f"Failed to get response from DeepSeek for region {region}")
            return []
        
        # Parse and validate the response
        templates = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Clean up the template
                template = re.sub(r'^\d+\.\s*', '', line)  # Remove numbering
                template = template.strip()
                if template:
                    # Validate template
                    is_valid, error_msg = self.validate_template_syntax(template, valid_fields)
                    if is_valid:
                        fields_used = self.extract_fields_from_template(template, data_fields)
                        templates.append({
                            'region': region,
                            'template': template,
                            'operators_used': self.extract_operators_from_template(template),
                            'fields_used': fields_used
                        })
                        logger.info(f"Valid template: {template[:50]}... (fields: {fields_used})")
                    else:
                        logger.warning(f"Invalid template rejected: {template[:50]}... - {error_msg}")
        
        logger.info(f"Generated {len(templates)} valid templates for region {region}")
        return templates
    
    def decide_next_action(self, region: str, existing_templates: List[Dict], simulation_results: List[TemplateResult]) -> Tuple[str, str]:
        """
        Use multi-arm bandit to decide whether to explore new templates or exploit existing ones
        Returns: (action, arm_id)
        """
        # Create arm IDs based on template patterns
        available_arms = []
        for template in existing_templates:
            # Create arm ID based on main operator pattern
            main_ops = template.get('operators_used', [])
            if main_ops:
                arm_id = f"{region}_{main_ops[0]}"  # Use first operator as arm identifier
                available_arms.append(arm_id)
        
        # Update bandit with successful simulation results only
        for result in simulation_results:
            if result.success:
                # Create arm ID for this result
                template_ops = self.extract_operators_from_template(result.template)
                if template_ops:
                    arm_id = f"{region}_{template_ops[0]}"
                    # Use fitness as reward (normalized to 0-1)
                    reward = min(1.0, max(0.0, result.fitness))
                    self.bandit.update_arm(arm_id, reward)
                    logger.info(f"Bandit updated: {arm_id} -> reward: {reward:.3f}")
        
        # Choose action
        action, arm_id = self.bandit.choose_action(available_arms)
        logger.info(f"Bandit decision for {region}: {action} (arm: {arm_id})")
        return action, arm_id
    
    def extract_operators_from_template(self, template: str) -> List[str]:
        """Extract operator names from a template"""
        operators_found = []
        for op in self.operators:
            if op['name'] in template:
                operators_found.append(op['name'])
        return operators_found
    
    def extract_fields_from_template(self, template: str, data_fields: List[Dict]) -> List[str]:
        """Extract field names from a template"""
        fields_found = []
        for field in data_fields:
            if field['id'] in template:
                fields_found.append(field['id'])
        return fields_found
    
    def save_progress(self):
        """Save current progress to file"""
        try:
            progress_data = {
                'timestamp': time.time(),
                'total_regions': self.progress_tracker.total_regions,
                'completed_regions': self.progress_tracker.completed_regions,
                'total_templates': self.progress_tracker.total_templates,
                'completed_templates': self.progress_tracker.completed_templates,
                'total_simulations': self.progress_tracker.total_simulations,
                'completed_simulations': self.progress_tracker.completed_simulations,
                'successful_simulations': self.progress_tracker.successful_simulations,
                'failed_simulations': self.progress_tracker.failed_simulations,
                'current_region': self.progress_tracker.current_region,
                'current_phase': self.progress_tracker.current_phase,
                'best_sharpe': self.progress_tracker.best_sharpe,
                'best_template': self.progress_tracker.best_template,
                'results': self.all_results
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            logger.info(f"Progress saved to {self.progress_file}")
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def load_progress(self) -> bool:
        """Load progress from file"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                # Restore progress tracker state
                self.progress_tracker.total_regions = progress_data.get('total_regions', 0)
                self.progress_tracker.completed_regions = progress_data.get('completed_regions', 0)
                self.progress_tracker.total_templates = progress_data.get('total_templates', 0)
                self.progress_tracker.completed_templates = progress_data.get('completed_templates', 0)
                self.progress_tracker.total_simulations = progress_data.get('total_simulations', 0)
                self.progress_tracker.completed_simulations = progress_data.get('completed_simulations', 0)
                self.progress_tracker.successful_simulations = progress_data.get('successful_simulations', 0)
                self.progress_tracker.failed_simulations = progress_data.get('failed_simulations', 0)
                self.progress_tracker.best_sharpe = progress_data.get('best_sharpe', 0.0)
                self.progress_tracker.best_template = progress_data.get('best_template', "")
                
                # Restore results
                self.all_results = progress_data.get('results', self.all_results)
                
                logger.info(f"Progress loaded from {self.progress_file}")
                return True
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
        return False
    
    def multi_simulate_templates(self, templates: List[Dict], region: str, delay: int = None) -> List[TemplateResult]:
        """Multi-simulate a batch of templates using the powerhouse approach"""
        logger.info(f"Multi-simulating {len(templates)} templates for region {region} with delay={delay}")
        if delay is not None:
            multiplier = self.pyramid_multipliers[region].get(str(delay), 1.0)
            logger.info(f"Using pyramid multiplier: {multiplier} for {region} delay={delay}")
        
        # Create simulation settings for the region
        config = self.region_configs[region]
        if delay is None:
            delay = config.delay
        settings = SimulationSettings(
            region=region,
            universe=config.universe,
            delay=delay,
            maxTrade="ON" if config.max_trade else "OFF"
        )
        
        # Group templates into pools for better management
        pool_size = 10
        template_pools = []
        for i in range(0, len(templates), pool_size):
            pool = templates[i:i + pool_size]
            template_pools.append(pool)
        
        logger.info(f"Created {len(template_pools)} pools of size {pool_size}")
        
        all_results = []
        
        for pool_idx, pool in enumerate(template_pools):
            logger.info(f"Processing pool {pool_idx + 1}/{len(template_pools)} with {len(pool)} templates")
            
            # Submit all templates in this pool
            progress_urls = []
            template_mapping = {}  # Map progress URLs to templates
            
            for template_idx, template_data in enumerate(pool):
                template = template_data['template']
                logger.info(f"Submitting template {template_idx + 1}/{len(pool)} in pool {pool_idx + 1}")
                
                try:
                    # Generate simulation data
                    simulation_data = {
                        'type': 'REGULAR',
                        'settings': asdict(settings),
                        'regular': template
                    }
                    
                    # Submit simulation
                    simulation_response = self.sess.post('https://api.worldquantbrain.com/simulations', 
                                                       json=simulation_data)
                    
                    # Handle authentication errors
                    if simulation_response.status_code == 401:
                        logger.info("Session expired, re-authenticating...")
                        self.setup_auth()
                        simulation_response = self.sess.post('https://api.worldquantbrain.com/simulations', 
                                                           json=simulation_data)
                    
                    if simulation_response.status_code != 201:
                        logger.error(f"Simulation API error for template {template}: {simulation_response.text}")
                        continue
                    
                    simulation_progress_url = simulation_response.headers.get('Location')
                    if not simulation_progress_url:
                        logger.error(f"No Location header in response for template {template}")
                        continue
                    
                    progress_urls.append(simulation_progress_url)
                    template_mapping[simulation_progress_url] = template_data
                    logger.info(f"Successfully submitted template {template_idx + 1}, got progress URL: {simulation_progress_url}")
                    
                except Exception as e:
                    logger.error(f"Error submitting template {template}: {str(e)}")
                    continue
            
            # Monitor progress for this pool
            if progress_urls:
                pool_results = self._monitor_pool_progress(progress_urls, template_mapping, settings)
                all_results.extend(pool_results)
                logger.info(f"Pool {pool_idx + 1} completed with {len(pool_results)} results")
                
                # Save progress after each pool
                self.save_progress()
            
            # Wait between pools to avoid overwhelming the API
            if pool_idx + 1 < len(template_pools):
                logger.info(f"Waiting 30 seconds before next pool...")
                time.sleep(30)
        
        logger.info(f"Multi-simulation complete: {len(all_results)} results")
        return all_results
    
    def _monitor_pool_progress(self, progress_urls: List[str], template_mapping: Dict[str, Dict], settings: SimulationSettings) -> List[TemplateResult]:
        """Monitor progress for a pool of simulations"""
        results = []
        max_wait_time = 3600  # 1 hour maximum wait time
        start_time = time.time()
        
        while progress_urls and (time.time() - start_time) < max_wait_time:
            logger.info(f"Monitoring {len(progress_urls)} simulations in pool...")
            
            completed_urls = []
            
            for progress_url in progress_urls:
                try:
                    response = self.sess.get(progress_url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        status = data.get('status')
                        
                        if status == 'COMPLETE':
                            template_data = template_mapping[progress_url]
                            is_data = data.get('is', {})
                            
                            result = TemplateResult(
                                template=template_data['template'],
                                region=template_data['region'],
                                settings=settings,
                                sharpe=is_data.get('sharpe', 0),
                                fitness=is_data.get('fitness', 0),
                                turnover=is_data.get('turnover', 0),
                                returns=is_data.get('returns', 0),
                                drawdown=is_data.get('drawdown', 0),
                                margin=is_data.get('margin', 0),
                                longCount=is_data.get('longCount', 0),
                                shortCount=is_data.get('shortCount', 0),
                                success=True,
                                timestamp=time.time()
                            )
                            results.append(result)
                            completed_urls.append(progress_url)
                            
                            # Update progress tracker
                            self.progress_tracker.update_simulation_progress(True, result.sharpe, result.template)
                            
                            logger.info(f"Template simulation completed successfully: {template_data['template'][:50]}...")
                            
                        elif status in ['FAILED', 'ERROR']:
                            template_data = template_mapping[progress_url]
                            result = TemplateResult(
                                template=template_data['template'],
                                region=template_data['region'],
                                settings=settings,
                                success=False,
                                error_message=data.get('message', 'Unknown error'),
                                timestamp=time.time()
                            )
                            results.append(result)
                            completed_urls.append(progress_url)
                            
                            # Update progress tracker
                            self.progress_tracker.update_simulation_progress(False)
                            
                            logger.error(f"Template simulation failed: {template_data['template'][:50]}... - {data.get('message', 'Unknown error')}")
                    
                    elif response.status_code == 401:
                        logger.info("Session expired, re-authenticating...")
                        self.setup_auth()
                        continue
                    
                except Exception as e:
                    logger.error(f"Error monitoring progress URL {progress_url}: {str(e)}")
                    continue
            
            # Remove completed URLs
            for url in completed_urls:
                progress_urls.remove(url)
            
            if not progress_urls:
                break
            
            # Wait before next check
            time.sleep(10)
        
        return results
    
    def generate_and_test_templates(self, regions: List[str] = None, templates_per_region: int = 10, resume: bool = False) -> Dict:
        """Generate templates and test them with multi-simulation"""
        if regions is None:
            regions = list(self.region_configs.keys())
        
        # Initialize progress tracker
        self.progress_tracker.total_regions = len(regions)
        
        # Try to resume from previous progress
        if resume and self.load_progress():
            logger.info("Resuming from previous progress...")
            # Filter out already completed regions
            completed_regions = set()
            for region in regions:
                if region in self.all_results.get('templates', {}):
                    completed_regions.add(region)
            
            remaining_regions = [r for r in regions if r not in completed_regions]
            if not remaining_regions:
                logger.info("All regions already completed!")
                return self.all_results
            
            logger.info(f"Resuming with remaining regions: {remaining_regions}")
            regions = remaining_regions
        
        # Update metadata
        self.all_results['metadata']['regions'] = list(self.region_configs.keys())
        self.all_results['metadata']['templates_per_region'] = templates_per_region
        
        for region in regions:
            logger.info(f"Processing region: {region}")
            self.progress_tracker.update_region_progress(region, "Generating templates")
            
            # Generate templates
            templates = self.generate_templates_for_region(region, templates_per_region)
            self.all_results['templates'][region] = templates
            
            if templates:
                self.progress_tracker.update_region_progress(region, "Simulating templates", 
                                                           templates=len(templates), simulations=len(templates))
                
                # Multi-simulate the templates with optimal delay
                optimal_delay = self.select_optimal_delay(region)
                simulation_results = self.multi_simulate_templates(templates, region, optimal_delay)
                
                # Only save successful simulations and collect failure information
                successful_results = [result for result in simulation_results if result.success]
                failed_results = [result for result in simulation_results if not result.success]
                failed_count = len(failed_results)
                
                if failed_count > 0:
                    logger.info(f"Discarding {failed_count} failed templates for {region}")
                    # Collect failure patterns for LLM learning
                    self._collect_failure_patterns(failed_results, region)
                    # Remove failed templates from progress JSON
                    failed_templates = [result.template for result in failed_results]
                    self._remove_failed_templates_from_progress(region, failed_templates)
                
                if successful_results:
                    logger.info(f"Saving {len(successful_results)} successful templates for {region}")
                    self.all_results['simulation_results'][region] = [
                        {
                            'template': result.template,
                            'region': result.region,
                            'sharpe': result.sharpe,
                            'fitness': result.fitness,
                            'turnover': result.turnover,
                            'returns': result.returns,
                            'drawdown': result.drawdown,
                            'margin': result.margin,
                            'longCount': result.longCount,
                            'shortCount': result.shortCount,
                            'success': result.success,
                            'error_message': result.error_message,
                            'timestamp': result.timestamp
                        }
                        for result in successful_results
                    ]
                else:
                    logger.warning(f"No successful templates for {region} - all {len(simulation_results)} templates failed")
                    self.all_results['simulation_results'][region] = []
                
                # Store results for analysis
                self.template_results.extend(simulation_results)
                
                # Generate HTML visualization for this batch (only successful results)
                if successful_results:
                    batch_results = {
                        'metadata': {
                            'generated_at': datetime.now().isoformat(),
                            'batch_number': self.progress_tracker.completed_regions,
                            'region': region,
                            'delay': optimal_delay,
                            'pyramid_multiplier': self.pyramid_multipliers[region].get(str(optimal_delay), 1.0),
                            'total_generated': len(simulation_results),
                            'successful': len(successful_results),
                            'failed': failed_count
                        },
                        'simulation_results': {region: [
                            {
                                'template': result.template,
                                'region': result.region,
                                'sharpe': result.sharpe,
                                'fitness': result.fitness,
                                'turnover': result.turnover,
                                'returns': result.returns,
                                'drawdown': result.drawdown,
                                'margin': result.margin,
                                'longCount': result.longCount,
                                'shortCount': result.shortCount,
                                'success': result.success,
                                'error_message': result.error_message,
                                'timestamp': result.timestamp
                            }
                            for result in successful_results
                        ]},
                        'analysis': self.analyze_results()
                    }
                else:
                    # No successful results, create empty batch
                    batch_results = {
                        'metadata': {
                            'generated_at': datetime.now().isoformat(),
                            'batch_number': self.progress_tracker.completed_regions,
                            'region': region,
                            'delay': optimal_delay,
                            'pyramid_multiplier': self.pyramid_multipliers[region].get(str(optimal_delay), 1.0),
                            'total_generated': len(simulation_results),
                            'successful': 0,
                            'failed': failed_count
                        },
                        'simulation_results': {region: []},
                        'analysis': self.analyze_results()
                    }
                html_file = self.generate_html_visualization(batch_results, self.progress_tracker.completed_regions + 1)
                logger.info(f"📊 HTML visualization generated: {html_file}")
            
            # Complete region
            self.progress_tracker.complete_region()
            
            # Save progress after each region
            self.save_progress()
            
            # Add delay between regions
            time.sleep(2)
        
        return self.all_results
    
    def analyze_results(self) -> Dict:
        """Analyze the simulation results"""
        if not self.template_results:
            return {}
        
        successful_results = [r for r in self.template_results if r.success]
        failed_results = [r for r in self.template_results if not r.success]
        
        analysis = {
            'total_templates': len(self.template_results),
            'successful_simulations': len(successful_results),
            'failed_simulations': len(failed_results),
            'success_rate': len(successful_results) / len(self.template_results) if self.template_results else 0,
            'performance_metrics': {}
        }
        
        if successful_results:
            sharpe_values = [r.sharpe for r in successful_results]
            fitness_values = [r.fitness for r in successful_results]
            turnover_values = [r.turnover for r in successful_results]
            
            analysis['performance_metrics'] = {
                'sharpe': {
                    'mean': np.mean(sharpe_values),
                    'std': np.std(sharpe_values),
                    'min': np.min(sharpe_values),
                    'max': np.max(sharpe_values)
                },
                'fitness': {
                    'mean': np.mean(fitness_values),
                    'std': np.std(fitness_values),
                    'min': np.min(fitness_values),
                    'max': np.max(fitness_values)
                },
                'turnover': {
                    'mean': np.mean(turnover_values),
                    'std': np.std(turnover_values),
                    'min': np.min(turnover_values),
                    'max': np.max(turnover_values)
                }
            }
        
        return analysis
    
    def generate_html_visualization(self, results: Dict, batch_number: int = 0):
        """Generate HTML visualization of progress and results"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Template Generator Progress - Batch {batch_number}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
        }}
        .stat-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #3498db;
        }}
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #3498db;
            margin: 0;
        }}
        .bandit-stats {{
            background: #e8f5e8;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }}
        .arm-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .arm-card {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            border: 2px solid #27ae60;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Template Generator Progress</h1>
            <p>Batch {batch_number} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Region: {results.get('metadata', {}).get('region', 'Unknown')} | 
               Delay: {results.get('metadata', {}).get('delay', 'N/A')} | 
               Pyramid Multiplier: {results.get('metadata', {}).get('pyramid_multiplier', 'N/A')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Generated</h3>
                <p class="value">{results.get('metadata', {}).get('total_generated', 0)}</p>
            </div>
            <div class="stat-card">
                <h3>Successful</h3>
                <p class="value">{results.get('metadata', {}).get('successful', 0)}</p>
            </div>
            <div class="stat-card">
                <h3>Failed</h3>
                <p class="value">{results.get('metadata', {}).get('failed', 0)}</p>
            </div>
            <div class="stat-card">
                <h3>Success Rate</h3>
                <p class="value">{results.get('analysis', {}).get('success_rate', 0):.1%}</p>
            </div>
        </div>
        
        <div class="bandit-stats">
            <h3>🎯 Multi-Arm Bandit Statistics</h3>
            <div class="arm-stats">
"""
        
        # Add bandit statistics
        for arm_id, stats in self.bandit.arm_stats.items():
            html_content += f"""
                <div class="arm-card">
                    <h4>{arm_id}</h4>
                    <p><strong>Pulls:</strong> {stats['pulls']}</p>
                    <p><strong>Avg Reward:</strong> {stats['avg_reward']:.3f}</p>
                </div>
"""
        
        html_content += """
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML file
        html_filename = f"template_progress_batch_{batch_number}.html"
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML visualization saved to {html_filename}")
        return html_filename

    def save_results(self, results: Dict, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            filename = self.results_file
            
        try:
            # Add analysis to results
            results['analysis'] = self.analyze_results()
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced template generator v2 with multi-simulation testing')
    parser.add_argument('--credentials', default='credential.txt', help='Path to credentials file')
    parser.add_argument('--deepseek-key', required=True, help='DeepSeek API key')
    parser.add_argument('--output', default='enhanced_results_v2.json', help='Output filename')
    parser.add_argument('--progress-file', default='template_progress_v2.json', help='Progress file')
    parser.add_argument('--regions', nargs='+', help='Regions to process (default: all)')
    parser.add_argument('--templates-per-region', type=int, default=10, help='Number of templates per region')
    parser.add_argument('--max-concurrent', type=int, default=5, help='Maximum concurrent simulations')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = EnhancedTemplateGeneratorV2(
            args.credentials, 
            args.deepseek_key, 
            args.max_concurrent,
            args.progress_file,
            args.output
        )
        
        # Generate and test templates
        results = generator.generate_and_test_templates(args.regions, args.templates_per_region, args.resume)
        
        # Save final results
        generator.save_results(results, args.output)
        
        # Print final summary
        print(f"\n{'='*70}")
        print("🎉 ENHANCED TEMPLATE GENERATION COMPLETE!")
        print(f"{'='*70}")
        
        total_templates = sum(len(templates) for templates in results['templates'].values())
        total_simulations = sum(len(sims) for sims in results['simulation_results'].values())
        successful_sims = sum(len([s for s in sims if s.get('success', False)]) for sims in results['simulation_results'].values())
        
        print(f"📊 Final Statistics:")
        print(f"   Total templates generated: {total_templates}")
        print(f"   Total simulations: {total_simulations}")
        print(f"   Successful simulations: {successful_sims}")
        print(f"   Success rate: {successful_sims/total_simulations*100:.1f}%" if total_simulations > 0 else "   Success rate: N/A")
        print(f"   Best Sharpe ratio: {generator.progress_tracker.best_sharpe:.3f}")
        print(f"   Results saved to: {args.output}")
        print(f"   Progress saved to: {args.progress_file}")
        
    except Exception as e:
        logger.error(f"Enhanced template generation failed: {e}")
        raise

if __name__ == '__main__': 
    main()
