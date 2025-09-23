#!/usr/bin/env python3
"""
Enhanced Template Generator v2 for MAPC2025 Competition
- Optimized for MAPC2025: GLB region, RAM neutralization, delay=1
- TRUE concurrent subprocess execution using ThreadPoolExecutor
- Smart plan for 8 concurrent slots: [explore, exploit, explore, exploit, explore, exploit, explore, exploit]
- Only save successful simulations, discard failures
- Continuous operation with multi-arm bandit
- Focus on RAM neutralization for risk-adjusted market strategies
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
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import numpy as np
from datetime import datetime
import threading
import sys
import math
import subprocess

# Configure logging with UTF-8 encoding to handle Unicode characters
import io
import codecs

# Create a safe stream handler that handles Unicode errors gracefully
class SafeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            # Try to write the message
            self.stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # If Unicode error, replace problematic characters
            try:
                msg = self.format(record)
                # Replace Unicode emojis with ASCII equivalents
                msg = msg.replace('📊', '[CHART]')
                msg = msg.replace('🔄', '[REFRESH]')
                msg = msg.replace('❌', '[FAIL]')
                msg = msg.replace('✅', '[SUCCESS]')
                msg = msg.replace('💡', '[INFO]')
                msg = msg.replace('🎯', '[TARGET]')
                msg = msg.replace('📈', '[TREND]')
                msg = msg.replace('🏆', '[TROPHY]')
                msg = msg.replace('⚠️', '[WARNING]')
                msg = msg.replace('💾', '[SAVE]')
                msg = msg.replace('🛑', '[STOP]')
                msg = msg.replace('🔍', '[SEARCH]')
                msg = msg.replace('🗑️', '[DELETE]')
                msg = msg.replace('🚀', '[ROCKET]')
                msg = msg.replace('🌍', '[GLOBE]')
                self.stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                # If all else fails, just write a simple message
                self.stream.write(f"Log message: {record.getMessage()}\n")
                self.flush()
        except Exception:
            self.handleError(record)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        SafeStreamHandler(sys.stdout),
        logging.FileHandler('enhanced_template_generator_v2.log', encoding='utf-8')
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
    neutralization_options: List[str] = None  # Available neutralization options for this region
    
    def __post_init__(self):
        if self.neutralization_options is None:
            # Default neutralization options by region
            if self.region == "GLB":
                # MAPC2025 competition focuses on GLB with REVERSION_AND_MOMENTUM neutralization
                self.neutralization_options = ["REVERSION_AND_MOMENTUM", "INDUSTRY", "SUBINDUSTRY", "SECTOR", "COUNTRY", "NONE"]
            elif self.region == "USA":
                self.neutralization_options = ["INDUSTRY", "SUBINDUSTRY", "SECTOR", "COUNTRY", "NONE"]
            elif self.region == "EUR":
                self.neutralization_options = ["INDUSTRY", "SUBINDUSTRY", "SECTOR", "COUNTRY", "NONE"]
            elif self.region == "CHN":
                self.neutralization_options = ["INDUSTRY", "SUBINDUSTRY", "SECTOR", "COUNTRY", "NONE"]
            elif self.region == "ASI":
                self.neutralization_options = ["INDUSTRY", "SUBINDUSTRY", "SECTOR", "COUNTRY", "NONE"]
            else:
                self.neutralization_options = ["INDUSTRY", "NONE"]

@dataclass
class SimulationSettings:
    """Configuration for simulation parameters - MAPC2025 optimized."""
    region: str = "GLB"  # MAPC2025 competition region
    universe: str = "TOP3000"
    instrumentType: str = "EQUITY"
    delay: int = 1  # MAPC2025 competition delay
    decay: int = 0
    neutralization: str = "REVERSION_AND_MOMENTUM"  # MAPC2025 competition neutralization
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
    neutralization: str = "REVERSION_AND_MOMENTUM"  # Track neutralization used - MAPC2025 default
    timestamp: float = 0.0

class MultiArmBandit:
    """Multi-arm bandit for explore vs exploit decisions with time decay"""
    
    def __init__(self, exploration_rate: float = 0.3, confidence_level: float = 0.95, 
                 decay_rate: float = 0.001, decay_interval: int = 100):
        self.exploration_rate = exploration_rate
        self.confidence_level = confidence_level
        self.arm_stats = {}  # {arm_id: {'pulls': int, 'rewards': list, 'avg_reward': float}}
        self.decay_rate = decay_rate  # How much to decay rewards per interval
        self.decay_interval = decay_interval  # Apply decay every N pulls
        self.total_pulls = 0  # Track total pulls for decay timing
    
    def add_arm(self, arm_id: str):
        """Add a new arm to the bandit"""
        if arm_id not in self.arm_stats:
            self.arm_stats[arm_id] = {
                'pulls': 0,
                'rewards': [],
                'avg_reward': 0.0,
                'confidence_interval': (0.0, 1.0)
            }
    
    def calculate_time_decay_factor(self) -> float:
        """Calculate time decay factor based on total pulls"""
        # Apply exponential decay: factor = e^(-decay_rate * (total_pulls / decay_interval))
        # This ensures rewards gradually decrease over time to prevent overfitting
        decay_steps = self.total_pulls / self.decay_interval
        decay_factor = math.exp(-self.decay_rate * decay_steps)
        return max(0.1, decay_factor)  # Minimum decay factor of 0.1 to prevent complete decay
    
    def update_arm(self, arm_id: str, reward: float):
        """Update arm statistics with new reward and apply time decay"""
        if arm_id not in self.arm_stats:
            self.add_arm(arm_id)
        
        # Increment total pulls for decay calculation
        self.total_pulls += 1
        
        # Calculate time decay factor
        time_decay_factor = self.calculate_time_decay_factor()
        
        # Apply time decay to the reward
        decayed_reward = reward * time_decay_factor
        
        stats = self.arm_stats[arm_id]
        stats['pulls'] += 1
        stats['rewards'].append(decayed_reward)
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
        
        # Log decay information periodically
        if self.total_pulls % self.decay_interval == 0:
            logger.info(f"🕒 Time decay applied: factor={time_decay_factor:.4f}, total_pulls={self.total_pulls}")
            logger.info(f"   Original reward: {reward:.3f} -> Decayed reward: {decayed_reward:.3f}")
    
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

def calculate_enhanced_reward(result: TemplateResult, time_decay_factor: float = 1.0) -> float:
    """
    Calculate enhanced reward based on multiple criteria with time decay:
    - Margin: >5bps good, >20bps excellent
    - Turnover: <30 very good, <50 acceptable
    - Return/Drawdown ratio: should be >1
    - Sharpe ratio: base reward
    - Time decay: gradually reduces reward over time to prevent overfitting
    """
    if not result.success:
        return 0.0
    
    # Base reward from Sharpe ratio
    base_reward = max(0, result.sharpe)
    
    # Margin bonus (in basis points)
    margin_bps = result.margin * 10000  # Convert to basis points
    margin_bonus = 0.0
    if margin_bps >= 20:
        margin_bonus = 0.5  # Excellent margin
    elif margin_bps >= 5:
        margin_bonus = 0.2  # Good margin
    elif margin_bps > 0:
        margin_bonus = 0.1  # Some margin
    
    # Turnover penalty/bonus
    turnover_bonus = 0.0
    if result.turnover <= 30:
        turnover_bonus = 0.3  # Very good turnover
    elif result.turnover <= 50:
        turnover_bonus = 0.1  # Acceptable turnover
    else:
        turnover_bonus = -0.2  # Penalty for high turnover
    
    # Return/Drawdown ratio bonus
    return_drawdown_bonus = 0.0
    if result.drawdown > 0 and result.returns > result.drawdown:
        ratio = result.returns / result.drawdown
        if ratio >= 2.0:
            return_drawdown_bonus = 0.4  # Excellent ratio
        elif ratio >= 1.5:
            return_drawdown_bonus = 0.2  # Good ratio
        elif ratio >= 1.0:
            return_drawdown_bonus = 0.1  # Acceptable ratio
    
    # Fitness bonus (if available)
    fitness_bonus = 0.0
    if result.fitness > 0:
        fitness_bonus = min(0.2, result.fitness * 0.1)  # Cap fitness bonus
    
    # Calculate total reward before time decay
    total_reward = base_reward + margin_bonus + turnover_bonus + return_drawdown_bonus + fitness_bonus
    
    # Apply time decay factor (gradually reduces reward over time)
    decayed_reward = total_reward * time_decay_factor
    
    # Ensure non-negative reward
    return max(0, decayed_reward)


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
    def __init__(self, credentials_path: str, deepseek_api_key: str, max_concurrent: int = 8, 
                 progress_file: str = "template_progress_v2.json", results_file: str = "enhanced_results_v2.json"):
        """Initialize the enhanced template generator with TRUE CONCURRENT subprocess execution"""
        self.sess = requests.Session()
        self.credentials_path = credentials_path
        self.deepseek_api_key = deepseek_api_key
        self.max_concurrent = min(max_concurrent, 8)  # WorldQuant Brain limit is 8
        self.progress_file = progress_file
        self.results_file = results_file
        self.progress_tracker = ProgressTracker()
        self.bandit = MultiArmBandit(exploration_rate=0.3, decay_rate=0.001, decay_interval=100)
        
        # TRUE CONCURRENT execution using ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
        self.active_futures = {}  # Track active Future objects
        self.completed_count = 0
        self.successful_count = 0
        self.failed_count = 0
        
        # Smart plan for 8 concurrent slots: [explore, exploit, explore, exploit, explore, exploit, explore, exploit]
        self.slot_plans = ['explore', 'exploit', 'explore', 'exploit', 'explore', 'exploit', 'explore', 'exploit']
        self.slot_plan_index = 0
        
        self.setup_auth()
        
        # Region configurations with pyramid multipliers - MAPC2025 focuses on GLB only
        self.region_configs = {
            "GLB": RegionConfig("GLB", "TOP3000", 1)  # MAPC2025 competition region
        }
        
        # Define regions list - only GLB for MAPC2025
        self.regions = list(self.region_configs.keys())
        
        # Pyramid theme multipliers - MAPC2025 focuses on GLB delay=1 only
        self.pyramid_multipliers = {
            "GLB": {"1": 2.0}  # MAPC2025 competition: GLB delay=1 only with high priority
        }
        
        # Load operators and data fields
        self.operators = self.load_operators()
        self.data_fields = {}
        
        # Error learning system - store failure patterns per region
        self.failure_patterns = {}  # {region: [{'template': str, 'error': str, 'timestamp': float}]}
        self.max_failures_per_region = 10  # Keep last 10 failures per region
        
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
        
        # Hopeful alphas storage for negation exploitation
        self.hopeful_alphas = []
        
        # Template quality tracking for PnL data quality
        self.template_quality_tracker = {}  # {template_hash: {'zero_pnl_count': int, 'total_attempts': int}}
        self.max_zero_pnl_attempts = 3  # Delete template after 3 zero PnL occurrences
        
        # Load previous progress if it exists (for exploit data)
        self.load_progress()
    
    def select_optimal_delay(self, region: str) -> int:
        """Select delay based on pyramid multipliers and region constraints - MAPC2025 optimized"""
        # MAPC2025 competition: GLB region only uses delay=1
        if region == "GLB":
            logger.info(f"MAPC2025 competition: Using delay=1 for GLB region")
            return 1
        
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
    
    def record_failure(self, region: str, template: str, error_message: str):
        """Record a failed template attempt for learning purposes"""
        if region not in self.failure_patterns:
            self.failure_patterns[region] = []
        
        failure_record = {
            'template': template,
            'error': error_message,
            'timestamp': time.time()
        }
        
        self.failure_patterns[region].append(failure_record)
        
        # Keep only the most recent failures
        if len(self.failure_patterns[region]) > self.max_failures_per_region:
            self.failure_patterns[region] = self.failure_patterns[region][-self.max_failures_per_region:]
        
        logger.info(f"📚 Recorded failure for {region}: {template[:50]}... - {error_message}")
    
    def get_failure_guidance(self, region: str) -> str:
        """Get failure guidance text for LLM prompts"""
        if region not in self.failure_patterns or not self.failure_patterns[region]:
            return ""
        
        recent_failures = self.failure_patterns[region][-5:]  # Last 5 failures
        if not recent_failures:
            return ""
        
        failure_guidance = f"""

PREVIOUS FAILURES TO AVOID:
{chr(10).join([f"- FAILED: {failure['template'][:60]}... ERROR: {failure['error']}" for failure in recent_failures])}

LEARN FROM THESE MISTAKES:
- Do NOT repeat the same error patterns
- Check operator parameter requirements carefully
- Ensure proper syntax and field names
- Avoid invalid parameter combinations
- Pay attention to the specific error messages above
"""
        return failure_guidance
    
    def get_data_fields_for_region(self, region: str, delay: int = 1) -> List[Dict]:
        """Get data fields for a specific region and delay with local caching"""
        try:
            # Check if we have cached data fields
            cache_key = f"{region}_{delay}"
            cache_file = f"data_fields_cache_{cache_key}.json"
            
            if os.path.exists(cache_file):
                logger.info(f"Loading cached data fields for {region} delay={delay}")
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    logger.info(f"Loaded {len(cached_data)} cached fields for {region} delay={delay}")
                    return cached_data
            
            logger.info(f"No cache found for {region} delay={delay}, fetching from API...")
            config = self.region_configs[region]
            
            # First get available datasets from multiple categories
            categories = ['fundamental', 'analyst', 'model', 'news', 'alternative']
            all_dataset_ids = []
            
            for category in categories:
                datasets_params = {
                    'category': category,
                    'delay': delay,
                    'instrumentType': 'EQUITY',
                    'region': region,
                    'universe': config.universe,
                    'limit': 20
                }
                
                logger.info(f"Getting {category} datasets for region {region}")
                datasets_response = self.sess.get('https://api.worldquantbrain.com/data-sets', params=datasets_params)
                
                if datasets_response.status_code == 200:
                    datasets_data = datasets_response.json()
                    available_datasets = datasets_data.get('results', [])
                    category_dataset_ids = [ds.get('id') for ds in available_datasets if ds.get('id')]
                    all_dataset_ids.extend(category_dataset_ids)
                    logger.info(f"Found {len(category_dataset_ids)} {category} datasets for region {region}")
                else:
                    logger.warning(f"Failed to get {category} datasets for region {region}")
            
            # Remove duplicates and use the combined list
            dataset_ids = list(set(all_dataset_ids))
            
            if not dataset_ids:
                logger.warning(f"No datasets found for region {region}, using fallback datasets")
                dataset_ids = ['fundamental6', 'fundamental2', 'analyst4', 'model16', 'model51', 'news12']
            
            logger.info(f"Total unique datasets for region {region}: {len(dataset_ids)}")
            
            # Get fields from datasets with pagination
            all_fields = []
            max_datasets = min(10, len(dataset_ids))  # Use up to 10 datasets
            
            for dataset in dataset_ids[:max_datasets]:
                dataset_fields = []
                page = 1
                max_pages = 5  # Get up to 5 pages per dataset
                
                while page <= max_pages:
                    params = {
                        'dataset.id': dataset,
                        'delay': delay,
                        'instrumentType': 'EQUITY',
                        'region': region,
                        'universe': config.universe,
                        'limit': 50,  # Increased from 20 to 50 per page
                        'page': page
                    }
                    
                    response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params)
                    if response.status_code == 200:
                        data = response.json()
                        fields = data.get('results', [])
                        if not fields:  # No more fields on this page
                            break
                        dataset_fields.extend(fields)
                        logger.info(f"Found {len(fields)} fields in dataset {dataset} page {page}")
                        page += 1
                    else:
                        logger.warning(f"Failed to get fields from dataset {dataset} page {page}")
                        break
                
                all_fields.extend(dataset_fields)
                logger.info(f"Total fields from dataset {dataset}: {len(dataset_fields)}")
            
            # Remove duplicates
            unique_fields = {field['id']: field for field in all_fields}.values()
            field_list = list(unique_fields)
            logger.info(f"Total unique fields for region {region}: {len(field_list)} (from {max_datasets} datasets)")
            
            # Cache the fetched data
            try:
                with open(cache_file, 'w') as f:
                    json.dump(field_list, f, indent=2)
                logger.info(f"Cached {len(field_list)} fields to {cache_file}")
            except Exception as cache_error:
                logger.warning(f"Failed to cache data fields: {cache_error}")
            
            return field_list
            
        except Exception as e:
            logger.error(f"Failed to get data fields for region {region}: {e}")
            return []
    
    def clear_data_fields_cache(self, region: str = None, delay: int = None):
        """Clear cached data fields for a specific region/delay or all caches"""
        import glob
        
        if region and delay is not None:
            # Clear specific cache
            cache_file = f"data_fields_cache_{region}_{delay}.json"
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.info(f"Cleared cache file: {cache_file}")
            else:
                logger.info(f"Cache file not found: {cache_file}")
        else:
            # Clear all cache files
            cache_files = glob.glob("data_fields_cache_*.json")
            for cache_file in cache_files:
                os.remove(cache_file)
                logger.info(f"Cleared cache file: {cache_file}")
            logger.info(f"Cleared {len(cache_files)} cache files")
    
    def get_cache_info(self):
        """Get information about cached data fields"""
        import glob
        
        cache_files = glob.glob("data_fields_cache_*.json")
        cache_info = {}
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    cache_info[cache_file] = len(data)
            except Exception as e:
                cache_info[cache_file] = f"Error: {e}"
        
        return cache_info
    
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
    
    def generate_templates_for_region_with_retry(self, region: str, num_templates: int = 1, max_retries: int = 5) -> List[Dict]:
        """Generate templates with retry logic and error learning"""
        for attempt in range(max_retries):
            logger.info(f"🔄 Template generation attempt {attempt + 1}/{max_retries} for {region}")
            
            templates = self.generate_templates_for_region(region, num_templates)
            
            if templates:
                logger.info(f"✅ Successfully generated {len(templates)} templates for {region} on attempt {attempt + 1}")
                return templates
            else:
                logger.warning(f"❌ Template generation failed for {region} on attempt {attempt + 1}")
                
                if attempt < max_retries - 1:
                    # Record the failure for learning (we don't have a specific template, so record a generic failure)
                    self.record_failure(region, "Template generation failed", f"Attempt {attempt + 1} - No valid templates generated")
                    logger.info(f"📚 Recorded failure for learning. Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    logger.error(f"🚫 All {max_retries} attempts failed for {region}. Discarding this attempt.")
                    self.record_failure(region, "All attempts failed", f"Failed after {max_retries} attempts")
        
        return []  # Return empty list if all attempts failed
    
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
        failure_guidance = self.get_failure_guidance(region)

        prompt = f"""Generate {num_templates} diverse and creative WorldQuant Brain alpha expression templates for the {region} region.

MAPC2025 COMPETITION FOCUS:
- This is for the MAPC2025 competition
- Region: {region} (GLB - Global)
- Universe: {config.universe}
- Delay: {optimal_delay} (MAPC2025 competition requirement)
- Max Trade: {config.max_trade}
- Primary Neutralization: REVERSION_AND_MOMENTUM (Risk-Adjusted Market)

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
5. NO special characters like %, ==, !=, &&, ||, >, <, >=, <=, NO COMPARISON OPERATORS LIKE >, <, >=, <=
6. NO missing commas between parameters
7. Balanced parentheses
8. Each template on a separate line
9. NO explanations or comments
10. NO custom operators or fields not in the lists above
11. Field names must match EXACTLY as shown in the Available Data Fields list
12. Read operator definitions carefully to understand parameter requirements
13. AVOID the failure patterns shown above - learn from previous mistakes
14. Double-check parameter counts and types for each operator
15. mdl307_sales_pct_eu_7 > mdl307_cusip_3, filter=true is NOT a valid template, refrain from using comparison operators like >, <, >=, <= in the templates 

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
        
        # Note: Templates are NOT saved to templates section here
        # They will only be saved after successful simulation in _add_to_results()
        
        return templates
    
    def decide_next_action(self):
        """
        Use multi-arm bandit to decide next action: explore new template or exploit existing one
        Returns: dict with action details
        """
        # Get all successful templates from all regions
        all_successful_templates = []
        for region, results in self.all_results.get('simulation_results', {}).items():
            for result in results:
                if result.get('success', False):
                    all_successful_templates.append({
                        'template': result,
                        'region': region,
                        'sharpe': result.get('sharpe', 0)
                    })
        
        # Filter out blacklisted templates (those with poor PnL quality)
        all_successful_templates = self.filter_blacklisted_templates(all_successful_templates)
        
        # Decide between explore and exploit
        if len(all_successful_templates) < 3:  # Need at least 3 successful templates to start exploiting
            # Explore: generate new template
            region = self.select_region_by_pyramid()
            delay = self.select_optimal_delay(region)
            return {
                'type': 'explore_new_template',
                'region': region,
                'delay': delay,
                'reason': 'insufficient_successful_templates'
            }
        else:
            # Use bandit to decide between explore and exploit
            explore_prob = 0.3  # 30% chance to explore, 70% to exploit
            
            if random.random() < explore_prob:
                # Explore: generate new template
                region = self.select_region_by_pyramid()
                delay = self.select_optimal_delay(region)
                return {
                    'type': 'explore_new_template',
                    'region': region,
                    'delay': delay,
                    'reason': 'bandit_exploration'
                }
            else:
                # Exploit: use existing successful template
                # Select best template based on sharpe ratio
                best_template = max(all_successful_templates, key=lambda x: x['sharpe'])
                region = best_template['region']
                delay = self.select_optimal_delay(region)
                return {
                    'type': 'exploit_existing_template',
                    'template': best_template['template'],
                    'region': region,
                    'delay': delay,
                    'reason': 'bandit_exploitation'
                }
    
    def select_region_by_pyramid(self):
        """Select region based on pyramid multipliers"""
        # Calculate weights based on pyramid multipliers
        region_weights = {}
        for region in self.regions:
            delay = self.select_optimal_delay(region)
            multiplier = self.pyramid_multipliers.get(region, {}).get(delay, 1.0)
            region_weights[region] = multiplier
        
        # Weighted random selection
        total_weight = sum(region_weights.values())
        if total_weight == 0:
            return random.choice(self.regions)
        
        rand = random.random() * total_weight
        cumulative = 0
        for region, weight in region_weights.items():
            cumulative += weight
            if rand <= cumulative:
                return region
        
        return random.choice(self.regions)
    
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
                # Save the all_results structure directly (not wrapped in 'results')
                'metadata': self.all_results.get('metadata', {}),
                'templates': self.all_results.get('templates', {}),
                'simulation_results': self.all_results.get('simulation_results', {})
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
                
                # Restore results - handle both old and new format
                if 'results' in progress_data:
                    # Old format: results wrapped in 'results' key
                    self.all_results = progress_data.get('results', self.all_results)
                else:
                    # New format: direct structure
                    self.all_results = {
                        'metadata': progress_data.get('metadata', {}),
                        'templates': progress_data.get('templates', {}),
                        'simulation_results': progress_data.get('simulation_results', {})
                    }
                
                # Debug: Check if results were loaded
                total_simulations = 0
                successful_simulations = 0
                for region, results in self.all_results.get('simulation_results', {}).items():
                    total_simulations += len(results)
                    successful_simulations += len([r for r in results if r.get('success', False)])
                
                logger.info(f"Progress loaded from {self.progress_file}")
                logger.info(f"📊 Loaded {total_simulations} total simulations, {successful_simulations} successful")
                return True
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
                            
                            # Get the alphaId from the simulation response
                            alpha_id = data.get('alpha')
                            if not alpha_id:
                                logger.error(f"No alphaId in completed simulation response for {template_data['template'][:50]}...")
                                result = TemplateResult(
                                    template=template_data['template'],
                                    region=template_data['region'],
                                    settings=settings,
                                    success=False,
                                    error_message="No alphaId in simulation response",
                                    timestamp=time.time()
                                )
                                results.append(result)
                                completed_urls.append(progress_url)
                                continue
                            
                            # Fetch the alpha data using the alphaId
                            logger.info(f"Simulation complete, fetching alpha {alpha_id} for {template_data['template'][:50]}...")
                            alpha_response = self.sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}')
                            
                            if alpha_response.status_code != 200:
                                logger.error(f"Failed to fetch alpha {alpha_id}: {alpha_response.status_code}")
                                result = TemplateResult(
                                    template=template_data['template'],
                                    region=template_data['region'],
                                    settings=settings,
                                    success=False,
                                    error_message=f"Failed to fetch alpha: {alpha_response.status_code}",
                                    timestamp=time.time()
                                )
                                results.append(result)
                                completed_urls.append(progress_url)
                                continue
                            
                            alpha_data = alpha_response.json()
                            is_data = alpha_data.get('is', {})
                            
                            # Extract metrics from the alpha data
                            sharpe = is_data.get('sharpe', 0)
                            fitness = is_data.get('fitness', 0)
                            turnover = is_data.get('turnover', 0)
                            returns = is_data.get('returns', 0)
                            drawdown = is_data.get('drawdown', 0)
                            margin = is_data.get('margin', 0)
                            longCount = is_data.get('longCount', 0)
                            shortCount = is_data.get('shortCount', 0)
                            
                            # A simulation is successful if it completed and has meaningful metrics
                            # Check if we have at least some non-zero performance indicators
                            has_meaningful_metrics = (
                                sharpe != 0 or  # Non-zero Sharpe ratio
                                (fitness is not None and fitness != 0) or  # Non-zero fitness
                                turnover != 0 or  # Non-zero turnover
                                returns != 0 or  # Non-zero returns
                                longCount > 0 or  # Has long positions
                                shortCount > 0  # Has short positions
                            )
                            
                            # Check PnL data quality for successful simulations
                            pnl_quality_ok = True
                            if has_meaningful_metrics:
                                pnl_quality_ok = self.track_template_quality(template_data['template'], alpha_id)
                            
                            # Only consider truly successful if both metrics and PnL quality are good
                            is_truly_successful = has_meaningful_metrics and pnl_quality_ok
                            
                            result = TemplateResult(
                                template=template_data['template'],
                                region=template_data['region'],
                                settings=settings,
                                sharpe=sharpe,
                                fitness=fitness if fitness is not None else 0,
                                turnover=turnover,
                                returns=returns,
                                drawdown=drawdown,
                                margin=margin,
                                longCount=longCount,
                                shortCount=shortCount,
                                success=is_truly_successful,
                                neutralization=settings.neutralization,
                                timestamp=time.time()
                            )
                            results.append(result)
                            completed_urls.append(progress_url)
                            
                            # Update progress tracker
                            self.progress_tracker.update_simulation_progress(is_truly_successful, result.sharpe, result.template)
                            
                            if is_truly_successful:
                                logger.info(f"✅ Template simulation completed successfully: {template_data['template'][:50]}...")
                                logger.info(f"📊 Alpha {alpha_id} Performance: Sharpe={sharpe}, Fitness={fitness}, Turnover={turnover}, Returns={returns}")
                                logger.info(f"📊 Alpha {alpha_id} Positions: Long={longCount}, Short={shortCount}")
                                logger.info(f"📊 Alpha {alpha_id} PnL Quality: Good")
                            else:
                                if has_meaningful_metrics and not pnl_quality_ok:
                                    logger.info(f"⚠️ Template simulation completed with good metrics but poor PnL quality: {template_data['template'][:50]}...")
                                    logger.info(f"📊 Alpha {alpha_id} Values: Sharpe={sharpe}, Fitness={fitness}, Turnover={turnover}, Returns={returns}")
                                    logger.info(f"📊 Alpha {alpha_id} PnL Quality: Poor - No reward given")
                                else:
                                    logger.info(f"⚠️ Template simulation completed but with zero/meaningless values: {template_data['template'][:50]}...")
                                    logger.info(f"📊 Alpha {alpha_id} Values: Sharpe={sharpe}, Fitness={fitness}, Turnover={turnover}, Returns={returns}")
                                    logger.info(f"📊 Alpha {alpha_id} Positions: Long={longCount}, Short={shortCount}")
                                    logger.info(f"📊 Alpha {alpha_id} Success criteria: has_meaningful_metrics={has_meaningful_metrics}")
                            
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
    
    def generate_and_test_templates(self, regions: List[str] = None, templates_per_region: int = 10, resume: bool = False, max_iterations: int = None) -> Dict:
        """Generate templates and test them with TRUE CONCURRENT subprocess execution"""
        if regions is None:
            regions = list(self.region_configs.keys())
        
        # Initialize progress tracker
        self.progress_tracker.total_regions = len(regions)
        
        # Try to load previous progress (always load if exists, regardless of resume flag)
        if self.load_progress():
            if resume:
                logger.info("Resuming from previous progress...")
            else:
                logger.info("Loaded previous progress for exploit data...")
        
        # Update metadata
        self.all_results['metadata']['regions'] = list(self.region_configs.keys())
        self.all_results['metadata']['templates_per_region'] = templates_per_region
        
        iteration = 0
        logger.info("🚀 Starting TRUE CONCURRENT template generation with subprocess execution...")
        logger.info("💡 Use Ctrl+C to stop gracefully")
        logger.info(f"🎯 Target: Maintain {self.max_concurrent} concurrent simulations for maximum efficiency")
        logger.info(f"🎯 Smart Plan: {self.slot_plans}")
        
        try:
            while True:
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"🛑 Reached maximum iterations ({max_iterations})")
                    break
                    
                iteration += 1
                logger.info(f"\n🔄 === ITERATION {iteration} ===")
                logger.info(f"📊 Active futures: {len(self.active_futures)}/{self.max_concurrent}")
                logger.info(f"📊 Completed: {self.completed_count}, Successful: {self.successful_count}, Failed: {self.failed_count}")
                
                # Process completed futures
                self._process_completed_futures()
                
                # Fill available slots with new concurrent tasks
                self._fill_available_slots_concurrent()
                
                # Save progress every iteration
                self.save_progress()
                
                # Wait a bit before next iteration
                time.sleep(2)
                    
        except KeyboardInterrupt:
            logger.info("\n🛑 Received interrupt signal. Stopping gracefully...")
            # Wait for active futures to complete
            self._wait_for_futures_completion()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        return self.all_results
    
    def process_simulation_results(self, simulation_results, region, delay, iteration):
        """Process simulation results and update bandit"""
        successful_results = [r for r in simulation_results if r.success]
        failed_count = len(simulation_results) - len(successful_results)
        
        if failed_count > 0:
            logger.info(f"🗑️ Discarding {failed_count} failed templates")
        
        if successful_results:
            logger.info(f"💾 Found {len(successful_results)} successful templates")
            
            # Update bandit with rewards using enhanced calculation with time decay
            for result in successful_results:
                # Extract main operator from template
                main_operator = self.extract_main_operator(result.template)
                if main_operator:
                    # Calculate time decay factor
                    time_decay_factor = self.bandit.calculate_time_decay_factor()
                    
                    # Use enhanced reward calculation with time decay
                    reward = calculate_enhanced_reward(result, time_decay_factor)
                    self.bandit.update_arm(main_operator, reward)
                    logger.info(f"Updated bandit: {main_operator} -> enhanced_reward={reward:.3f} (decay_factor={time_decay_factor:.4f})")
            
            # Add to results
            if region not in self.all_results['simulation_results']:
                self.all_results['simulation_results'][region] = []
            self.all_results['simulation_results'][region].extend(successful_results)
            
            # Update progress tracker
            for result in successful_results:
                self.progress_tracker.update_simulation_progress(True, result.sharpe, result.template)
        else:
            logger.warning(f"⚠️ No successful simulations in this batch")
    
    def extract_main_operator(self, template):
        """Extract the main operator from a template"""
        # Simple heuristic: find the outermost operator
        template = template.strip()
        if '(' in template:
            # Find the first operator before the first parenthesis
            paren_pos = template.find('(')
            operator_part = template[:paren_pos].strip()
            if operator_part:
                return operator_part
        return None
    
    def generate_template_variations(self, base_template, region, delay):
        """Generate variations of a successful template with different data fields"""
        # Get available data fields for this region/delay
        data_fields = self.get_data_fields_for_region(region, delay)
        if not data_fields:
            return []
        
        # Extract the base template structure
        base_code = base_template['template']
        
        # Find all field names in the base template
        import re
        # More flexible pattern to match various field types
        field_patterns = [
            r'fnd\d+_[a-zA-Z0-9_]+',  # fnd28_field_name
            r'[a-zA-Z_][a-zA-Z0-9_]*',  # cash_st, volume, etc.
        ]
        
        existing_fields = []
        for pattern in field_patterns:
            fields = re.findall(pattern, base_code)
            existing_fields.extend(fields)
        
        # Remove duplicates and filter out common words that aren't fields
        common_words = {'max', 'min', 'log', 'abs', 'scale', 'rank', 'ts_rank', 'ts_mean', 'ts_std', 'ts_delta', 'ts_av_diff', 'divide', 'multiply', 'add', 'subtract', 'if_else', 'winsorize', 'group_neutralize', 'longscale', 'shortscale', 'scale'}
        existing_fields = list(set([f for f in existing_fields if f not in common_words and len(f) > 2]))
        
        if not existing_fields:
            logger.warning(f"No field patterns found in template: {base_code[:50]}...")
            return []
        
        # Generate variations by randomly selecting new data fields
        import random
        variations = []
        field_names = [field['id'] for field in data_fields]
        
        # Generate multiple variations with random field selections
        num_variations = min(50, len(field_names))  # Generate up to 50 variations or all available fields
        
        # Get existing field positions in the template for replacement
        existing_field_positions = []
        for field in existing_fields:
            start_pos = base_code.find(field)
            if start_pos != -1:
                existing_field_positions.append((field, start_pos, start_pos + len(field)))
        
        # Sort by position to replace from right to left (to maintain positions)
        existing_field_positions.sort(key=lambda x: x[1], reverse=True)
        
        # Generate variations
        used_combinations = set()  # Track used field combinations to avoid duplicates
        
        for i in range(num_variations):
            # Randomly select fields to use in this variation
            num_fields_to_use = random.randint(1, min(3, len(existing_fields)))  # Use 1-3 fields
            selected_fields = random.sample(field_names, num_fields_to_use)
            
            # Create field combination signature to avoid duplicates
            field_signature = tuple(sorted(selected_fields))
            if field_signature in used_combinations:
                continue  # Skip duplicate combination
            used_combinations.add(field_signature)
            
            # Create variation by replacing existing fields with randomly selected ones
            variation_code = base_code
            fields_used = []
            
            # Replace existing fields with randomly selected ones
            for j, (old_field, start_pos, end_pos) in enumerate(existing_field_positions):
                if j < len(selected_fields):
                    new_field = selected_fields[j]
                    variation_code = variation_code[:start_pos] + new_field + variation_code[end_pos:]
                    fields_used.append(new_field)
            
            # Only add if the variation is different from the original
            if variation_code != base_code:
                variations.append({
                    'template': variation_code,
                    'region': region,
                    'operators_used': base_template.get('operators_used', []),
                    'fields_used': fields_used
                })
        
        logger.info(f"Generated {len(variations)} variations for template: {base_code[:50]}... (from {len(field_names)} available fields)")
        return variations
    
    def generate_neutralization_variations(self, base_template, region, delay):
        """Generate variations of a successful template with different neutralization settings"""
        # Get region-specific neutralization options
        region_config = self.region_configs.get(region)
        if not region_config or not region_config.neutralization_options:
            logger.warning(f"No neutralization options available for region {region}")
            return []
        
        neutralization_options = region_config.neutralization_options
        variations = []
        
        # Create variations with different neutralization settings
        for neutralization in neutralization_options:
            if neutralization != base_template.get('neutralization', 'REVERSION_AND_MOMENTUM'):
                variation = {
                    'template': base_template['template'],
                    'region': region,
                    'operators_used': base_template.get('operators_used', []),
                    'fields_used': base_template.get('fields_used', []),
                    'neutralization': neutralization,
                    'variation_type': 'neutralization'
                }
                variations.append(variation)
        
        logger.info(f"Generated {len(variations)} neutralization variations for region {region}: {neutralization_options}")
        return variations
    
    def generate_negation_variations(self, base_template, region, delay):
        """Generate negated variations of a successful template to test inverse strategies"""
        base_code = base_template['template']
        variations = []
        
        # Check if the template is already negated (starts with minus)
        if base_code.strip().startswith('-'):
            logger.info(f"Template already negated, skipping negation variation: {base_code[:50]}...")
            return []
        
        # Create negated version by adding minus sign
        negated_template = f"-({base_code})"
        
        # Get valid fields for validation
        data_fields = self.get_data_fields_for_region(region, delay)
        valid_fields = [field['id'] for field in data_fields] if data_fields else []
        
        # Validate the negated template syntax
        is_valid, error_msg = self.validate_template_syntax(negated_template, valid_fields)
        if is_valid:
            variation = {
                'template': negated_template,
                'region': region,
                'operators_used': base_template.get('operators_used', []),
                'fields_used': base_template.get('fields_used', []),
                'neutralization': base_template.get('neutralization', 'REVERSION_AND_MOMENTUM'),
                'variation_type': 'negation',
                'original_template': base_code
            }
            variations.append(variation)
            logger.info(f"Generated negation variation: {negated_template[:50]}...")
        else:
            logger.warning(f"Negated template failed syntax validation: {negated_template[:50]}... - {error_msg}")
        
        return variations
    
    def is_hopeful_alpha(self, result: TemplateResult) -> bool:
        """
        Check if an alpha is 'hopeful' - has negative metrics but absolute values above threshold
        These are candidates for negation exploitation
        """
        if not result.success:
            return False
        
        # Check if any key metrics are negative but have good absolute values
        hopeful_conditions = []
        
        # Sharpe ratio: negative but absolute value > 0.5
        if result.sharpe < 0 and abs(result.sharpe) > 1.25:
            hopeful_conditions.append(f"Sharpe={result.sharpe:.3f} (abs={abs(result.sharpe):.3f})")
        
        # Fitness: negative but absolute value > 0.3
        if result.fitness < 0 and abs(result.fitness) > 1:
            hopeful_conditions.append(f"Fitness={result.fitness:.3f} (abs={abs(result.fitness):.3f})")
        
        # Returns: negative but absolute value > 0.1
        if result.returns < 0 and abs(result.returns) > 0.2:
            hopeful_conditions.append(f"Returns={result.returns:.3f} (abs={abs(result.returns):.3f})")
        
        # Margin: negative but absolute value > 0.002 (20bps)
        if result.margin < 0 and abs(result.margin) > 0.002:
            hopeful_conditions.append(f"Margin={result.margin:.4f} (abs={abs(result.margin):.4f})")
        
        if hopeful_conditions:
            logger.info(f"🎯 HOPEFUL ALPHA detected: {', '.join(hopeful_conditions)}")
            logger.info(f"  Template: {result.template[:50]}...")
            return True
        
        return False
    
    def check_pnl_data_quality(self, alpha_id: str) -> Tuple[bool, str]:
        """
        Check PnL data quality for an alpha
        Returns: (is_good_quality, reason)
        """
        try:
            # Fetch PnL data from WorldQuant API
            pnl_url = f'https://api.worldquantbrain.com/alphas/{alpha_id}/recordsets/pnl'
            response = self.sess.get(pnl_url)
            
            if response.status_code != 200:
                return False, f"Failed to fetch PnL data: {response.status_code}"
            
            pnl_data = response.json()
            records = pnl_data.get('records', [])
            
            if not records:
                return False, "No PnL records found"
            
            # Analyze PnL data quality
            total_records = len(records)
            zero_pnl_count = 0
            non_zero_pnl_count = 0
            total_pnl_sum = 0.0
            
            for record in records:
                if len(record) >= 2:  # Ensure we have at least date and pnl
                    pnl_value = record[1]  # PnL is the second element
                    if pnl_value == 0.0:
                        zero_pnl_count += 1
                    else:
                        non_zero_pnl_count += 1
                        total_pnl_sum += abs(pnl_value)
            
            # Calculate quality metrics
            zero_pnl_ratio = zero_pnl_count / total_records if total_records > 0 else 1.0
            avg_non_zero_pnl = total_pnl_sum / non_zero_pnl_count if non_zero_pnl_count > 0 else 0.0
            
            # Quality criteria
            if zero_pnl_ratio > 0.8:  # More than 80% zeros
                return False, f"Too many zero PnL values: {zero_pnl_ratio:.1%} ({zero_pnl_count}/{total_records})"
            
            if non_zero_pnl_count < 10:  # Less than 10 non-zero values
                return False, f"Insufficient non-zero PnL data: {non_zero_pnl_count} values"
            
            if avg_non_zero_pnl < 0.001:  # Very small PnL values
                return False, f"PnL values too small: avg={avg_non_zero_pnl:.6f}"
            
            # Good quality
            return True, f"Good PnL quality: {non_zero_pnl_count}/{total_records} non-zero values, avg={avg_non_zero_pnl:.4f}"
            
        except Exception as e:
            return False, f"Error checking PnL quality: {str(e)}"
    
    def track_template_quality(self, template: str, alpha_id: str) -> bool:
        """
        Track template quality based on PnL data
        Returns: True if template should be kept, False if it should be deleted
        """
        # Create template hash for tracking
        template_hash = hash(template)
        
        # Check PnL data quality
        is_good_quality, reason = self.check_pnl_data_quality(alpha_id)
        
        # Initialize tracking if not exists
        if template_hash not in self.template_quality_tracker:
            self.template_quality_tracker[template_hash] = {
                'zero_pnl_count': 0,
                'total_attempts': 0,
                'template': template
            }
        
        tracker = self.template_quality_tracker[template_hash]
        tracker['total_attempts'] += 1
        
        if not is_good_quality:
            tracker['zero_pnl_count'] += 1
            logger.warning(f"⚠️ Poor PnL quality for template: {template[:50]}...")
            logger.warning(f"   Reason: {reason}")
            logger.warning(f"   Zero PnL count: {tracker['zero_pnl_count']}/{self.max_zero_pnl_attempts}")
            
            # Check if template should be deleted
            if tracker['zero_pnl_count'] >= self.max_zero_pnl_attempts:
                logger.error(f"🗑️ DELETING template due to poor PnL quality: {template[:50]}...")
                logger.error(f"   Total attempts: {tracker['total_attempts']}, Zero PnL: {tracker['zero_pnl_count']}")
                return False  # Delete template
        else:
            logger.info(f"✅ Good PnL quality for template: {template[:50]}...")
            logger.info(f"   {reason}")
        
        return True  # Keep template
    
    def is_template_blacklisted(self, template: str) -> bool:
        """Check if a template is blacklisted due to poor PnL quality"""
        template_hash = hash(template)
        if template_hash in self.template_quality_tracker:
            tracker = self.template_quality_tracker[template_hash]
            return tracker['zero_pnl_count'] >= self.max_zero_pnl_attempts
        return False
    
    def filter_blacklisted_templates(self, templates: List[Dict]) -> List[Dict]:
        """Filter out blacklisted templates from a list"""
        filtered_templates = []
        blacklisted_count = 0
        
        for template in templates:
            if not self.is_template_blacklisted(template.get('template', '')):
                filtered_templates.append(template)
            else:
                blacklisted_count += 1
        
        if blacklisted_count > 0:
            logger.info(f"🚫 Filtered out {blacklisted_count} blacklisted templates due to poor PnL quality")
        
        return filtered_templates
    
    def _process_completed_futures(self):
        """Process completed futures and update bandit"""
        completed_futures = []
        
        for future_id, future in self.active_futures.items():
            if future.done():
                completed_futures.append(future_id)
                try:
                    result = future.result()
                    if result and result.success:
                        self.successful_count += 1
                        self._update_bandit_with_result(result)
                        self._add_to_results(result)
                        logger.info(f"✅ CONCURRENT simulation SUCCESS: {result.template[:50]}... (Sharpe: {result.sharpe:.3f})")
                    elif result and not result.success:
                        self.failed_count += 1
                        error_msg = getattr(result, 'error_message', 'Simulation failed')
                        logger.info(f"❌ CONCURRENT simulation FAILED: {result.template[:50]}... - {error_msg}")
                    else:
                        # result is None - this means the concurrent task failed to return a proper result
                        self.failed_count += 1
                        logger.info(f"❌ CONCURRENT simulation FAILED: Task returned no result (likely template generation or API error)")
                except Exception as e:
                    self.failed_count += 1
                    logger.error(f"❌ CONCURRENT simulation ERROR: {e}")
        
        # Remove completed futures
        for future_id in completed_futures:
            del self.active_futures[future_id]
            self.completed_count += 1
    
    def _fill_available_slots_concurrent(self):
        """Fill available slots with TRUE CONCURRENT subprocess execution"""
        available_slots = self.max_concurrent - len(self.active_futures)
        
        if available_slots > 0:
            logger.info(f"🎯 Filling {available_slots} available slots with CONCURRENT tasks...")
            
            for _ in range(available_slots):
                # Get next action from smart plan
                plan_type = self.slot_plans[self.slot_plan_index % len(self.slot_plans)]
                self.slot_plan_index += 1
                
                if plan_type == 'explore':
                    # Explore: generate new template and simulate CONCURRENTLY
                    future = self.executor.submit(self._explore_and_simulate_concurrent)
                    future_id = f"explore_{int(time.time() * 1000)}"
                    self.active_futures[future_id] = future
                    logger.info(f"🚀 Started CONCURRENT EXPLORE task: {future_id}")
                
                elif plan_type == 'exploit':
                    # Exploit: try to use existing successful template
                    logger.info(f"🎯 EXPLOIT mode: Looking for successful templates...")
                    successful_templates = self._get_successful_templates()
                    if successful_templates:
                        # Use best performing template
                        best_template = max(successful_templates, key=lambda x: x.get('sharpe', 0))
                        logger.info(f"🎯 EXPLOIT: Using best template with Sharpe={best_template.get('sharpe', 0):.3f}")
                        future = self.executor.submit(self._exploit_and_simulate_concurrent, best_template)
                        future_id = f"exploit_{int(time.time() * 1000)}"
                        self.active_futures[future_id] = future
                        logger.info(f"🚀 Started CONCURRENT EXPLOIT task: {future_id}")
                    else:
                        # No successful templates yet, fallback to explore
                        logger.info(f"🎯 EXPLOIT: No successful templates found, falling back to EXPLORE")
                        future = self.executor.submit(self._explore_and_simulate_concurrent)
                        future_id = f"explore_fallback_{int(time.time() * 1000)}"
                        self.active_futures[future_id] = future
                        logger.info(f"🚀 Started CONCURRENT EXPLORE (fallback) task: {future_id}")
    
    def _explore_and_simulate_concurrent(self) -> Optional[TemplateResult]:
        """CONCURRENTLY explore new template and simulate it"""
        try:
            # Generate new template with retry logic
            region = self.select_region_by_pyramid()
            delay = self.select_optimal_delay(region)
            templates = self.generate_templates_for_region_with_retry(region, 1, 5)
            
            if not templates:
                logger.warning(f"No templates generated for {region}")
                return TemplateResult(
                    template="",
                    region=region,
                    settings={'region': region, 'delay': delay},
                    success=False,
                    error_message="No templates generated",
                    timestamp=time.time()
                )
            
            template = templates[0]
            logger.info(f"🔍 EXPLORING new template: {template['template'][:50]}...")
            
            # Simulate the template CONCURRENTLY
            return self._simulate_template_concurrent(template, region, delay)
            
        except Exception as e:
            logger.error(f"Error in explore_and_simulate_concurrent: {e}")
            return TemplateResult(
                template="",
                region="",
                settings={},
                success=False,
                error_message=f"Explore error: {str(e)}",
                timestamp=time.time()
            )
    
    def _exploit_and_simulate_concurrent(self, best_template: Dict) -> Optional[TemplateResult]:
        """CONCURRENTLY exploit existing template and simulate it with enhanced variations"""
        try:
            region = best_template['region']
            delay = self.select_optimal_delay(region)
            
            # Generate all types of variations
            field_variations = self.generate_template_variations(best_template, region, delay)
            neutralization_variations = self.generate_neutralization_variations(best_template, region, delay)
            negation_variations = self.generate_negation_variations(best_template, region, delay)
            
            # Also check for hopeful alphas in the same region for negation exploitation
            hopeful_negation_variations = []
            for hopeful_alpha in self.hopeful_alphas:
                if hopeful_alpha['region'] == region:
                    # Create negation variation from hopeful alpha
                    hopeful_template = {
                        'template': hopeful_alpha['template'],
                        'region': hopeful_alpha['region'],
                        'operators_used': [],
                        'fields_used': [],
                        'neutralization': hopeful_alpha['neutralization']
                    }
                    hopeful_negations = self.generate_negation_variations(hopeful_template, region, delay)
                    hopeful_negation_variations.extend(hopeful_negations)
            
            # Combine all variations
            all_variations = field_variations + neutralization_variations + negation_variations + hopeful_negation_variations
            
            if not all_variations:
                logger.warning(f"No variations generated for {region}")
                return TemplateResult(
                    template=best_template.get('template', ''),
                    region=region,
                    settings={'region': region, 'delay': delay},
                    success=False,
                    error_message="No variations generated",
                    timestamp=time.time()
                )
            
            # Randomly select a variation type
            variation = random.choice(all_variations)
            variation_type = variation.get('variation_type', 'field')
            
            logger.info(f"🎯 EXPLOITING {variation_type} variation: {variation['template'][:50]}...")
            if variation_type == 'neutralization':
                logger.info(f"  Using neutralization: {variation.get('neutralization', 'INDUSTRY')}")
            elif variation_type == 'negation':
                original_template = variation.get('original_template', '')
                if original_template:
                    logger.info(f"  Testing negated version of: {original_template[:50]}...")
                else:
                    logger.info(f"  Testing negation variation from hopeful alpha")
            
            # Simulate the variation CONCURRENTLY
            return self._simulate_template_concurrent(variation, region, delay)
            
        except Exception as e:
            logger.error(f"Error in exploit_and_simulate_concurrent: {e}")
            return TemplateResult(
                template=best_template.get('template', ''),
                region=best_template.get('region', ''),
                settings={},
                success=False,
                error_message=f"Exploit error: {str(e)}",
                timestamp=time.time()
            )
    
    def _simulate_template_concurrent(self, template: Dict, region: str, delay: int) -> Optional[TemplateResult]:
        """CONCURRENTLY simulate a single template"""
        try:
            # Create simulation data with all required fields
            # Use neutralization from template variation if available, default to REVERSION_AND_MOMENTUM for MAPC2025
            neutralization = template.get('neutralization', 'REVERSION_AND_MOMENTUM')
            simulation_data = {
                'type': 'REGULAR',
                'settings': {
                    'instrumentType': 'EQUITY',
                    'region': region,
                    'universe': self.region_configs[region].universe,
                    'delay': delay,
                    'decay': 0,
                    'neutralization': neutralization,
                    'truncation': 0.08,
                    'pasteurization': 'ON',
                    'unitHandling': 'VERIFY',
                    'nanHandling': 'OFF',
                    'maxTrade': 'ON' if self.region_configs[region].max_trade else 'OFF',
                    'language': 'FASTEXPR',
                    'visualization': False,
                    'startDate': '2013-01-20',
                    'endDate': '2023-01-20',
                    'testPeriod': 'P5Y0M0D'
                },
                'regular': template['template']
            }
            
            # Submit simulation
            response = self.sess.post('https://api.worldquantbrain.com/simulations', json=simulation_data)
            
            if response.status_code != 201:
                error_message = f"Failed to submit simulation: {response.status_code}"
                logger.error(f"Failed to submit simulation: {response.status_code} - {response.text}")
                # Record the failure for learning
                self.record_failure(region, template['template'], error_message)
                
                return TemplateResult(
                    template=template['template'],
                    region=region,
                    settings={'region': region, 'delay': delay},
                    success=False,
                    error_message=error_message,
                    timestamp=time.time()
                )
            
            progress_url = response.headers.get('Location')
            if not progress_url:
                error_message = "No Location header in response"
                logger.error(f"No Location header in response")
                # Record the failure for learning
                self.record_failure(region, template['template'], error_message)
                
                return TemplateResult(
                    template=template['template'],
                    region=region,
                    settings={'region': region, 'delay': delay},
                    success=False,
                    error_message=error_message,
                    timestamp=time.time()
                )
            
            # Monitor simulation progress CONCURRENTLY
            return self._monitor_simulation_concurrent(progress_url, template, region, delay)
            
        except Exception as e:
            error_message = f"Simulation error: {str(e)}"
            logger.error(f"Error in simulate_template_concurrent: {e}")
            # Record the failure for learning
            self.record_failure(region, template['template'], error_message)
            
            return TemplateResult(
                template=template['template'],
                region=region,
                settings={'region': region, 'delay': delay},
                success=False,
                error_message=error_message,
                timestamp=time.time()
            )
    
    def _monitor_simulation_concurrent(self, progress_url: str, template: Dict, region: str, delay: int) -> Optional[TemplateResult]:
        """CONCURRENTLY monitor simulation progress"""
        max_wait_time = 3600  # 1 hour maximum wait time
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait_time:
            try:
                response = self.sess.get(progress_url)
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status')
                    
                    if status == 'COMPLETE':
                        # Get the alphaId from the simulation response
                        alpha_id = data.get('alpha')
                        if not alpha_id:
                            logger.error(f"No alphaId in completed simulation response")
                            return TemplateResult(
                                template=template['template'],
                                region=region,
                                settings={'region': region, 'delay': delay},
                                success=False,
                                error_message="No alphaId in simulation response",
                                timestamp=time.time()
                            )
                        
                        # Fetch the alpha data using the alphaId
                        logger.info(f"Simulation complete, fetching alpha {alpha_id}")
                        alpha_response = self.sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}')
                        
                        if alpha_response.status_code != 200:
                            logger.error(f"Failed to fetch alpha {alpha_id}: {alpha_response.status_code}")
                            return TemplateResult(
                                template=template['template'],
                                region=region,
                                settings={'region': region, 'delay': delay},
                                success=False,
                                error_message=f"Failed to fetch alpha: {alpha_response.status_code}",
                                timestamp=time.time()
                            )
                        
                        alpha_data = alpha_response.json()
                        is_data = alpha_data.get('is', {})
                        
                        # Extract metrics from the alpha data
                        sharpe = is_data.get('sharpe', 0)
                        fitness = is_data.get('fitness', 0)
                        turnover = is_data.get('turnover', 0)
                        returns = is_data.get('returns', 0)
                        drawdown = is_data.get('drawdown', 0)
                        margin = is_data.get('margin', 0)
                        longCount = is_data.get('longCount', 0)
                        shortCount = is_data.get('shortCount', 0)
                        
                        # A simulation is successful if it completed and has meaningful metrics
                        # Check if we have at least some non-zero performance indicators
                        has_meaningful_metrics = (
                            sharpe != 0 or  # Non-zero Sharpe ratio
                            (fitness is not None and fitness != 0) or  # Non-zero fitness
                            turnover != 0 or  # Non-zero turnover
                            returns != 0 or  # Non-zero returns
                            longCount > 0 or  # Has long positions
                            shortCount > 0  # Has short positions
                        )
                        
                        is_truly_successful = has_meaningful_metrics
                        
                        logger.info(f"Alpha {alpha_id} metrics: Sharpe={sharpe}, Fitness={fitness}, Turnover={turnover}, Returns={returns}")
                        logger.info(f"Alpha {alpha_id} positions: Long={longCount}, Short={shortCount}")
                        logger.info(f"Alpha {alpha_id} success: {is_truly_successful}")
                        
                        return TemplateResult(
                            template=template['template'],
                            region=region,
                            settings={'region': region, 'delay': delay},
                            sharpe=sharpe,
                            fitness=fitness if fitness is not None else 0,
                            turnover=turnover,
                            returns=returns,
                            drawdown=drawdown,
                            margin=margin,
                            longCount=longCount,
                            shortCount=shortCount,
                            success=is_truly_successful,
                            timestamp=time.time()
                        )
                    
                    elif status in ['FAILED', 'ERROR']:
                        error_message = data.get('message', 'Unknown error')
                        # Record the failure for learning
                        self.record_failure(region, template['template'], error_message)
                        
                        return TemplateResult(
                            template=template['template'],
                            region=region,
                            settings={'region': region, 'delay': delay},
                            success=False,
                            error_message=error_message,
                            timestamp=time.time()
                        )
                
                elif response.status_code == 401:
                    logger.info("Session expired, re-authenticating...")
                    self.setup_auth()
                    continue
                
            except Exception as e:
                logger.error(f"Error monitoring simulation: {e}")
                continue
            
            # Wait before next check
            time.sleep(10)
        
        # Timeout
        error_message = "Simulation timeout"
        # Record the failure for learning
        self.record_failure(region, template['template'], error_message)
        
        return TemplateResult(
            template=template['template'],
            region=region,
            settings={'region': region, 'delay': delay},
            success=False,
            error_message=error_message,
            timestamp=time.time()
        )
    
    def _wait_for_futures_completion(self):
        """Wait for all active futures to complete"""
        logger.info(f"Waiting for {len(self.active_futures)} active futures to complete...")
        
        while self.active_futures:
            self._process_completed_futures()
            if self.active_futures:
                time.sleep(5)  # Check every 5 seconds
        
        logger.info("All futures completed")
    
    def _update_bandit_with_result(self, result):
        """Update the bandit with simulation result using enhanced reward calculation with time decay"""
        if result.success:
            main_operator = self.extract_main_operator(result.template)
            if main_operator:
                # Calculate time decay factor
                time_decay_factor = self.bandit.calculate_time_decay_factor()
                
                # Use enhanced reward calculation with time decay
                reward = calculate_enhanced_reward(result, time_decay_factor)
                self.bandit.update_arm(main_operator, reward)
                
                # Log detailed reward breakdown
                margin_bps = result.margin * 10000
                turnover_bonus = 0.3 if result.turnover <= 30 else (0.1 if result.turnover <= 50 else -0.2)
                return_drawdown_ratio = result.returns / result.drawdown if result.drawdown > 0 else 0
                
                logger.info(f"Updated bandit: {main_operator} -> enhanced_reward={reward:.3f} (decay_factor={time_decay_factor:.4f})")
                logger.info(f"  Breakdown: Sharpe={result.sharpe:.3f}, Margin={margin_bps:.1f}bps, "
                          f"Turnover={result.turnover:.1f}, R/D={return_drawdown_ratio:.2f}")
                
                # Check if this is a hopeful alpha (negative metrics with good absolute values)
                if self.is_hopeful_alpha(result):
                    # Store this as a candidate for negation exploitation
                    self._store_hopeful_alpha(result)
        else:
            # Even for failed results, check if they might be hopeful
            if self.is_hopeful_alpha(result):
                logger.info(f"🎯 Failed but hopeful alpha detected: {result.template[:50]}...")
                self._store_hopeful_alpha(result)
    
    def _store_hopeful_alpha(self, result: TemplateResult):
        """Store a hopeful alpha for potential negation exploitation"""
        hopeful_alpha = {
            'template': result.template,
            'region': result.region,
            'sharpe': result.sharpe,
            'fitness': result.fitness,
            'returns': result.returns,
            'margin': result.margin,
            'turnover': result.turnover,
            'drawdown': result.drawdown,
            'neutralization': result.neutralization,
            'timestamp': time.time(),
            'original_success': result.success
        }
        
        # Add to hopeful alphas list (keep last 20)
        self.hopeful_alphas.append(hopeful_alpha)
        if len(self.hopeful_alphas) > 20:
            self.hopeful_alphas.pop(0)  # Remove oldest
        
        logger.info(f"💾 Stored hopeful alpha for negation exploitation: {result.template[:50]}...")
        logger.info(f"  Metrics: Sharpe={result.sharpe:.3f}, Fitness={result.fitness:.3f}, "
                   f"Returns={result.returns:.3f}, Margin={result.margin:.4f}")
    
    def _add_to_results(self, result):
        """Add result to the results collection"""
        if result.success:
            region = result.region
            if region not in self.all_results['simulation_results']:
                self.all_results['simulation_results'][region] = []
            
            # Add to simulation results
            self.all_results['simulation_results'][region].append({
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
            })
            
            # Also add to templates section (only successful templates)
            if region not in self.all_results['templates']:
                self.all_results['templates'][region] = []
            
            # Check if template already exists in templates section to avoid duplicates
            template_exists = any(t.get('template') == result.template for t in self.all_results['templates'][region])
            if not template_exists:
                self.all_results['templates'][region].append({
                    'region': result.region,
                    'template': result.template,
                    'operators_used': self.extract_operators_from_template(result.template),
                    'fields_used': self.extract_fields_from_template(result.template, [])
                })
                logger.info(f"Added successful template to templates section: {result.template[:50]}...")
            
            # Update progress tracker
            self.progress_tracker.update_simulation_progress(True, result.sharpe, result.template)
        else:
            # Failed simulation - remove from results if it was previously saved
            self._remove_failed_template_from_results(result.template)
            logger.info(f"Failed template NOT saved to results: {result.template[:50]}...")
    
    def _wait_for_completion(self):
        """Wait for all active simulations to complete"""
        logger.info(f"Waiting for {self.active_simulations} active simulations to complete...")
        
        while self.active_simulations > 0:
            self._process_completed_simulations()
            if self.active_simulations > 0:
                time.sleep(5)  # Check every 5 seconds
        
        logger.info("All simulations completed")
    
    def _get_successful_templates(self):
        """Get all successful templates from results"""
        successful_templates = []
        total_results = 0
        for region, results in self.all_results.get('simulation_results', {}).items():
            total_results += len(results)
            for result in results:
                if result.get('success', False):
                    successful_templates.append(result)
        
        logger.info(f"📊 Found {len(successful_templates)} successful templates out of {total_results} total results")
        if successful_templates:
            best_sharpe = max(successful_templates, key=lambda x: x.get('sharpe', 0))
            logger.info(f"🏆 Best successful template: Sharpe={best_sharpe.get('sharpe', 0):.3f}, Region={best_sharpe.get('region', 'N/A')}")
        
        return successful_templates
    
    def _remove_failed_template_from_results(self, template_text):
        """Remove a failed template from results if it was previously saved"""
        removed_from_simulation_results = False
        removed_from_templates = False
        
        # Remove from simulation_results
        for region, results in self.all_results.get('simulation_results', {}).items():
            for i, result in enumerate(results):
                if result.get('template') == template_text:
                    logger.info(f"Removing failed template from simulation_results: {template_text[:50]}...")
                    results.pop(i)
                    removed_from_simulation_results = True
                    break
        
        # Remove from templates section
        for region, templates in self.all_results.get('templates', {}).items():
            for i, template in enumerate(templates):
                if template.get('template') == template_text:
                    logger.info(f"Removing failed template from templates section: {template_text[:50]}...")
                    templates.pop(i)
                    removed_from_templates = True
                    break
        
        return removed_from_simulation_results or removed_from_templates
    
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
    parser = argparse.ArgumentParser(description='Enhanced template generator v2 for MAPC2025 competition - GLB region, REVERSION_AND_MOMENTUM neutralization, delay=1')
    parser.add_argument('--credentials', default='credential.txt', help='Path to credentials file')
    parser.add_argument('--deepseek-key', required=True, help='DeepSeek API key')
    parser.add_argument('--output', default='enhanced_results_v2.json', help='Output filename')
    parser.add_argument('--progress-file', default='template_progress_v2.json', help='Progress file')
    parser.add_argument('--regions', nargs='+', help='Regions to process (default: all)')
    parser.add_argument('--templates-per-region', type=int, default=10, help='Number of templates per region')
    parser.add_argument('--max-concurrent', type=int, default=8, help='Maximum concurrent simulations (default: 8)')
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
        print("🎉 MAPC2025 TEMPLATE GENERATION COMPLETE!")
        print("🏆 GLB Region, REVERSION_AND_MOMENTUM Neutralization, Delay=1")
        print(f"{'='*70}")
        
        total_simulations = sum(len(sims) for sims in results['simulation_results'].values())
        successful_sims = sum(len([s for s in sims if s.get('success', False)]) for sims in results['simulation_results'].values())
        
        print(f"📊 Final Statistics:")
        print(f"   Total concurrent simulations: {total_simulations}")
        print(f"   Successful simulations: {successful_sims}")
        print(f"   Failed simulations: {total_simulations - successful_sims}")
        print(f"   Success rate: {successful_sims/total_simulations*100:.1f}%" if total_simulations > 0 else "   Success rate: N/A")
        print(f"   Best Sharpe ratio: {generator.progress_tracker.best_sharpe:.3f}")
        print(f"   Results saved to: {args.output}")
        print(f"   Progress saved to: {args.progress_file}")
        print(f"   Smart Plan Used: {generator.slot_plans}")
        print(f"   Max Concurrent: {generator.max_concurrent}")
        
    except Exception as e:
        logger.error(f"Enhanced template generation failed: {e}")
        raise

if __name__ == '__main__': 
    main()
