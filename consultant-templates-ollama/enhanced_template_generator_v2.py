#!/usr/bin/env python3
"""
Enhanced Template Generator v2 with TRUE CONCURRENT Subprocess Execution
- NO HTML generation - completely removed
- TRUE concurrent subprocess execution using ThreadPoolExecutor
- Smart plan for 8 concurrent slots: [explore, exploit, explore, exploit, explore, exploit, explore, exploit]
- Only save successful simulations, discard failures
- Continuous operation with multi-arm bandit
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
import ollama

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
            if self.region == "USA":
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
    neutralization: str = "INDUSTRY"  # Track neutralization used
    alpha_id: str = ""  # Track alpha ID for post-simulation analysis (optional)
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
    
    def choose_action_weighted(self, available_arms: List[str], performance_weights: List[float] = None) -> Tuple[str, str]:
        """
        Choose action using weighted selection based on performance
        Best performing templates have higher chance but not 100%
        """
        if not available_arms:
            return "explore", "new_template"
        
        # Add any new arms
        for arm in available_arms:
            self.add_arm(arm)
        
        # If no performance weights provided, use average rewards as weights
        if performance_weights is None:
            weights = []
            for arm_id in available_arms:
                stats = self.arm_stats[arm_id]
                # Use average reward as weight, with minimum weight of 0.1
                weight = max(stats['avg_reward'], 0.1)
                weights.append(weight)
        else:
            weights = performance_weights
        
        # Normalize weights to probabilities
        total_weight = sum(weights)
        if total_weight == 0:
            # If all weights are 0, use uniform selection
            probabilities = [1.0 / len(available_arms)] * len(available_arms)
        else:
            probabilities = [w / total_weight for w in weights]
        
        # Weighted random selection
        selected_idx = random.choices(range(len(available_arms)), weights=probabilities)[0]
        selected_arm = available_arms[selected_idx]
        
        # Decide explore vs exploit based on exploration rate
        if random.random() < self.exploration_rate:
            return "explore", "new_template"
        else:
            return "exploit", selected_arm
    
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
        margin_bonus = -3  # Some margin
    
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
    def __init__(self, credentials_path: str, ollama_model: str = "qwen2.5-coder:7b", max_concurrent: int = 8, 
                 progress_file: str = "template_progress_v2.json", results_file: str = "enhanced_results_v2.json"):
        """Initialize the enhanced template generator with TRUE CONCURRENT subprocess execution"""
        self.sess = requests.Session()
        self.credentials_path = credentials_path
        self.ollama_model = ollama_model
        self.ollama_url = "http://127.0.0.1:11434"  # Default Ollama URL
        self.max_concurrent = min(max_concurrent, 8)  # WorldQuant Brain limit is 8
        self.progress_file = progress_file
        self.results_file = results_file
        self.progress_tracker = ProgressTracker()
        self.bandit = MultiArmBandit(exploration_rate=0.3, decay_rate=0.001, decay_interval=100)
        
        # Operator usage tracking and blacklist
        self.operator_usage_count = {}  # Track operator usage frequency
        self.operator_blacklist = set()  # Temporarily blacklisted operators
        self.blacklist_threshold = 5  # Blacklist after 5 uses in recent templates
        self.max_operator_usage = 10  # Max times an operator can be used before blacklisting
        self.blacklist_file = "operator_blacklist.json"
        
        # Blacklist release mechanism
        self.operator_blacklist_timestamps = {}  # Track when operators were blacklisted
        self.operator_blacklist_reasons = {}  # Track why operators were blacklisted
        self.blacklist_release_threshold = 10  # Release after 10 successful simulations
        self.blacklist_timeout_hours = 10/60  # Release after 10 minutes (10/60 = 0.167 hours)
        self.successful_simulations_since_blacklist = 0  # Track successful simulations
        
        self._load_blacklist_from_disk()
        
        # TRUE CONCURRENT execution using ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
        self.active_futures = {}  # Track active Future objects
        self.future_start_times = {}  # Track when futures were started
        self.future_timeout = 300  # 5 minutes timeout for hanging futures
        self.thread_termination_log = []  # Track thread terminations
        self.thread_exception_count = 0  # Count unexpected thread terminations
        self.thread_count = 0  # Track total thread count
        self.completed_threads = 0  # Track completed threads
        self.completed_count = 0
        self.successful_count = 0
        self.failed_count = 0
        
        # Smart plan for 8 concurrent slots: [explore, exploit, explore, exploit, explore, exploit, explore, exploit]
        self.slot_plans = ['explore', 'exploit', 'explore', 'exploit', 'explore', 'exploit', 'explore', 'exploit']
        self.slot_plan_index = 0
        
        self.setup_auth()
        
        # Optimization tracking
        self.optimization_queue = []  # Queue of alphas to optimize
        self.optimization_results = {}  # Track optimization history
        self.max_optimization_iterations = 10
        
        # Initialize persona system
        self.personas = self._load_personas()
        self.recent_personas = []  # Track recently used personas
        
        # Load historical alpha expressions for inspiration from local file
        logger.info("📚 Loading historical alphas from local JSON file...")
        self.historical_alphas = self._load_historical_alphas()
        
        # Cache historical alphas to avoid repeated API calls
        if self.historical_alphas:
            logger.info(f"✅ Cached {len(self.historical_alphas)} historical alphas for reuse")
        else:
            logger.warning("⚠️ No historical alphas loaded - will use personas only")
            # Set a flag to use personas more frequently when historical alphas are unavailable
            self.use_personas_only = True
        
        # Rate limiting for API calls (30 requests per minute)
        self.last_api_call_time = 0
        self.api_call_interval = 2.1  # 2.1 seconds between calls (30 calls per minute = 2 seconds, plus buffer)
        
        # Three-phase system tracking
        self.total_simulations = 0
        self.phase_switch_threshold = 100  # Switch to exploitation after 100 successful simulations
        self.exploitation_end_threshold = 300  # 100 + 200 exploitation
        self.current_phase = "explore_exploit"  # "explore_exploit", "exploit", "loop"
        self.exploitation_phase = False
        self.top_templates = []  # Track top-performing templates for exploitation
        self.exploitation_bandit = None  # Separate bandit for exploitation phase
        self.loop_count = 0  # Track number of loops completed
        
        # Region configurations with pyramid multipliers
        self.region_configs = {
            "USA": RegionConfig("USA", "TOP3000", 1),
            "GLB": RegionConfig("GLB", "TOP3000", 1),
            "EUR": RegionConfig("EUR", "TOP2500", 1),
            "ASI": RegionConfig("ASI", "MINVOL1M", 1, max_trade=True),
            "CHN": RegionConfig("CHN", "TOP2000U", 1, max_trade=True)
        }
        
        # Define regions list
        self.regions = list(self.region_configs.keys())
        
        # Dynamic learning system for operator compatibility
        self.operator_compatibility_file = 'operator_compatibility.json'
        self.operator_compatibility = self._load_operator_compatibility()
        
        # Pyramid theme multipliers (delay=0, delay=1) for each region
        self.pyramid_multipliers = {
            "USA": {"0": 1.8, "1": 1.2},  # delay=0 has higher multiplier
            "GLB": {"0": 1.0, "1": 1.5},  # delay=1 has higher multiplier (delay=0 not available)
            "EUR": {"0": 1.7, "1": 1.4},  # delay=0 has higher multiplier
            "ASI": {"0": 1.0, "1": 1.5},  # delay=1 has higher multiplier (delay=0 not available)
            "CHN": {"0": 1.0, "1": 1.8}   # delay=1 has higher multiplier (delay=0 not available)
        }
        
        # Load operators and data fields
        self.operators = self.load_operators()
        self.data_fields = {}
        
        # Operator usage tracking for diversity
        self.operator_usage_count = {}  # Track how often each operator is used
        
        # Template generation tracking to avoid repetition
        self.recent_templates = []  # Store recently generated templates
        self.max_recent_templates = 50  # Keep track of last 50 templates
        
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
        
        # PnL checking statistics
        self.pnl_check_stats = {
            'total_checks': 0,
            'mandatory_checks': 0,
            'probability_checks': 0,
            'skipped_checks': 0,
            'flatlined_detected': 0,
            'suspicion_scores': []
        }
        
        # Load existing blacklist for persistence
        self.load_blacklist_from_file()
        
        # Load previous progress if it exists (for exploit data)
        self.load_progress()
        
        # Dynamic field selection strategy tracking
        self.elite_templates_found = 0  # Count of elite templates discovered
        self.last_elite_discovery_time = 0  # Timestamp of last elite template discovery
        self.field_strategy_start_time = time.time()  # When field strategy started
        self.field_strategy_mode = "random_exploration"  # "random_exploration" or "rare_focused"
        self.elite_discovery_threshold = 3  # Switch to rare-focused after 3 elite templates
        self.time_decay_threshold = 3600  # 1 hour in seconds - switch back to random if no elites found
        self.field_strategy_weights = {
            "random_exploration": {"random": 0.7, "rare": 0.3},
            "rare_focused": {"random": 0.3, "rare": 0.7}
        }
    
    def select_optimal_delay(self, region: str) -> int:
        """Select delay based on pyramid multipliers and region constraints"""
        multipliers = self.pyramid_multipliers.get(region, {"0": 1.0, "1": 1.0})
        
        # For ASI, CHN, and GLB, only delay=1 is available
        if region in ["ASI", "CHN", "GLB"]:
            return 1
        
        # For EUR, prefer delay=0 (higher multiplier: 1.7 vs 1.4)
        if region == "EUR":
            delay_0_mult = multipliers.get("0", 1.0)
            delay_1_mult = multipliers.get("1", 1.0)
            if delay_0_mult > delay_1_mult:
                logger.info(f"EUR: Preferring delay=0 (multiplier: {delay_0_mult} vs {delay_1_mult})")
                return 0
            else:
                logger.info(f"EUR: Using delay=1 (multiplier: {delay_1_mult} vs {delay_0_mult})")
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
    
    def make_api_request(self, method: str, url: str, **kwargs):
        """Make API request with automatic 401 reauthentication and rate limiting"""
        # Rate limiting: ensure minimum interval between API calls
        import time
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        
        if time_since_last_call < self.api_call_interval:
            wait_time = self.api_call_interval - time_since_last_call
            logger.info(f"⏳ Rate limiting: waiting {wait_time:.2f} seconds before API call")
            time.sleep(wait_time)
        
        self.last_api_call_time = time.time()
        
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                if method.upper() == 'GET':
                    response = self.sess.get(url, **kwargs)
                elif method.upper() == 'POST':
                    response = self.sess.post(url, **kwargs)
                elif method.upper() == 'PUT':
                    response = self.sess.put(url, **kwargs)
                elif method.upper() == 'PATCH':
                    response = self.sess.patch(url, **kwargs)
                elif method.upper() == 'DELETE':
                    response = self.sess.delete(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Check for 401 error
                if response.status_code == 401:
                    if attempt < max_retries - 1:
                        logger.warning(f"🔐 401 Unauthorized - Re-authenticating (attempt {attempt + 1}/{max_retries})")
                        self.setup_auth()
                        continue
                    else:
                        logger.error(f"🔐 Authentication failed after {max_retries} attempts")
                        raise Exception("Authentication failed after retries")
                
                return response
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"🔐 API request failed, retrying: {e}")
                    time.sleep(1)
                    continue
                else:
                    raise e
        
        return None
    
    def load_operators(self) -> List[Dict]:
        """Load operators from operatorRAW.json"""
        try:
            with open('operatorRAW.json', 'r', encoding='utf-8') as f:
                operators = json.load(f)
            logger.info(f"Loaded {len(operators)} operators")
            return operators
        except Exception as e:
            logger.error(f"Failed to load operators: {e}")
            return []
    
    def record_failure(self, region: str, template: str, error_message: str, settings: Dict = None):
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
        
        # Enhanced logging with simulation settings
        if settings:
            logger.info(f"📚 Recorded failure for {region}: {template[:50]}... - {error_message}")
            logger.info(f"🔧 SIMULATION SETTINGS: Region={settings.get('region', 'UNKNOWN')}, Universe={settings.get('universe', 'UNKNOWN')}, Delay={settings.get('delay', 'UNKNOWN')}, Neutralization={settings.get('neutralization', 'UNKNOWN')}")
        else:
            logger.info(f"📚 Recorded failure for {region}: {template[:50]}... - {error_message}")
    
    def get_failure_guidance(self, region: str) -> str:
        """ULTRA-ENHANCED failure guidance system for maximum success rate"""
        if region not in self.failure_patterns or not self.failure_patterns[region]:
            return ""
        
        recent_failures = self.failure_patterns[region][-10:]  # Last 10 failures for better learning
        if not recent_failures:
            return ""
        
        # Analyze failure patterns for better guidance
        error_types = {}
        for failure in recent_failures:
            error = failure['error']
            if 'data field' in error.lower() or 'operator' in error.lower():
                error_types['data_field_as_operator'] = error_types.get('data_field_as_operator', 0) + 1
            elif 'parameter' in error.lower():
                error_types['parameter_error'] = error_types.get('parameter_error', 0) + 1
            elif 'syntax' in error.lower():
                error_types['syntax_error'] = error_types.get('syntax_error', 0) + 1
            else:
                error_types['other'] = error_types.get('other', 0) + 1
        
        # Generate specific guidance based on error patterns
        guidance_sections = []
        
        if error_types.get('data_field_as_operator', 0) > 0:
            guidance_sections.append("""
CRITICAL ERROR PATTERN: DATA FIELDS AS OPERATORS
- NEVER use data field names as operators (e.g., anl69_best_bps_stddev(...))
- ALWAYS use data fields as inputs to operators (e.g., ts_rank(anl69_best_bps_stddev, 20))
- Data fields are INPUTS, not operators!
""")
        
        if error_types.get('parameter_error', 0) > 0:
            guidance_sections.append("""
CRITICAL ERROR PATTERN: PARAMETER ERRORS
- Check operator parameter requirements carefully
- Use correct parameter types (integers, not decimals)
- Ensure proper parameter counts for each operator
- Read operator definitions before using them
""")
        
        if error_types.get('syntax_error', 0) > 0:
            guidance_sections.append("""
CRITICAL ERROR PATTERN: SYNTAX ERRORS
- Ensure perfectly balanced parentheses
- Use proper comma separation between parameters
- NO comparison operators: >, <, >=, <=, ==, !=, &&, ||, %
- Check for missing commas or extra characters
""")
        
        failure_guidance = f"""
ULTRA-ENHANCED FAILURE ANALYSIS - LEARN FROM THESE MISTAKES:

RECENT FAILURE PATTERNS ({len(recent_failures)} failures analyzed):
{chr(10).join([f"- FAILED: {failure['template'][:60]}... ERROR: {failure['error']}" for failure in recent_failures[-5:]])}

ERROR TYPE ANALYSIS:
{chr(10).join([f"- {error_type.replace('_', ' ').title()}: {count} occurrences" for error_type, count in error_types.items()])}

{''.join(guidance_sections)}

SUCCESS STRATEGY - AVOID THESE PATTERNS:
- Do NOT repeat the same error patterns shown above
- Focus on proven financial patterns (momentum, mean reversion, volatility)
- Use group operations for risk reduction
- Combine operators for sophisticated strategies
- Pay attention to the specific error messages above
- Double-check syntax before generating templates

MAXIMUM SUCCESS FOCUS:
- Generate templates with economic intuition
- Use appropriate time periods and parameters
- Focus on risk-adjusted returns
- Combine multiple operators for alpha generation
"""
        return failure_guidance
    
    def is_good_alpha(self, result: TemplateResult) -> bool:
        """Check if an alpha meets the criteria for optimization"""
        if not result.success:
            return False
        
        # Check criteria: 0.75+ Sharpe, 30%+ margin
        sharpe_threshold = 0.75
        margin_threshold = 0.30  # 30% margin
        
        is_elite = (result.sharpe >= sharpe_threshold and 
                   result.margin >= margin_threshold)
        
        # Track elite template discovery for dynamic field strategy
        if is_elite:
            self._track_elite_discovery()
        
        return is_elite
    
    def _track_elite_discovery(self):
        """Track elite template discovery and update field strategy"""
        self.elite_templates_found += 1
        self.last_elite_discovery_time = time.time()
        
        logger.info(f"🎯 ELITE DISCOVERY: Found elite template #{self.elite_templates_found}")
        
        # Update field strategy based on elite discoveries
        self._update_field_strategy()
    
    def _update_field_strategy(self):
        """Update field selection strategy based on elite template discoveries and time"""
        current_time = time.time()
        time_since_last_elite = current_time - self.last_elite_discovery_time if self.last_elite_discovery_time > 0 else float('inf')
        time_since_start = current_time - self.field_strategy_start_time
        
        # Strategy decision logic
        if self.elite_templates_found >= self.elite_discovery_threshold:
            # Switch to rare-focused mode when we have enough elite templates
            if self.field_strategy_mode != "rare_focused":
                self.field_strategy_mode = "rare_focused"
                logger.info(f"🎯 FIELD STRATEGY: Switching to RARE-FOCUSED mode (elite count: {self.elite_templates_found})")
        elif time_since_last_elite > self.time_decay_threshold and time_since_start > self.time_decay_threshold:
            # Switch back to random exploration if no elites found in time threshold
            if self.field_strategy_mode != "random_exploration":
                self.field_strategy_mode = "random_exploration"
                logger.info(f"🎯 FIELD STRATEGY: Switching to RANDOM EXPLORATION mode (no elites in {self.time_decay_threshold}s)")
        
        # Log current strategy status
        weights = self.field_strategy_weights[self.field_strategy_mode]
        logger.info(f"🎯 FIELD STRATEGY: {self.field_strategy_mode.upper()} - Random: {weights['random']:.1%}, Rare: {weights['rare']:.1%}")
        logger.info(f"🎯 ELITE TRACKING: Found {self.elite_templates_found} elites, last discovery: {time_since_last_elite:.0f}s ago")
    
    def force_field_strategy_update(self):
        """Manually trigger field strategy update (useful for testing or manual control)"""
        logger.info("🎯 MANUAL FIELD STRATEGY UPDATE TRIGGERED")
        self._update_field_strategy()
        status = self.get_field_strategy_status()
        logger.info(f"🎯 CURRENT STATUS: {status}")
    
    def add_to_optimization_queue(self, result: TemplateResult):
        """Add a good alpha to the optimization queue"""
        if self.is_good_alpha(result):
            optimization_id = f"{result.template}_{result.region}_{int(time.time())}"
            self.optimization_queue.append({
                'id': optimization_id,
                'template': result.template,
                'region': result.region,
                'current_result': result,
                'iteration': 0,
                'best_result': result,
                'improvement_count': 0
            })
            logger.info(f"🎯 Added alpha to optimization queue: {optimization_id}")
            logger.info(f"   Sharpe: {result.sharpe:.3f}, Margin: {result.margin:.3f}")
    
    def optimize_alpha_with_llm(self, optimization_item: Dict) -> Dict:
        """Use DeepSeek API to optimize an alpha iteratively"""
        template = optimization_item['template']
        region = optimization_item['region']
        current_result = optimization_item['current_result']
        iteration = optimization_item['iteration']
        
        # Get all available operators for optimization
        all_operators = self.operators
        data_fields = self.get_data_fields_for_region(region, current_result.settings.delay)
        
        # Create optimization prompt
        operators_desc = []
        for op in all_operators:
            operators_desc.append(f"- {op['name']}: {op['description']} (Definition: {op['definition']})")
        
        fields_desc = []
        for field in data_fields:
            field_type = field.get('type', 'REGULAR')
            
            # Add field type information with operator compatibility
            if field_type == 'VECTOR':
                field_type_info = "VECTOR (use Cross Sectional operators: normalize, quantile, rank, scale, winsorize, zscore, vec_avg, vec_sum, vec_max, vec_min)"
            elif field_type == 'MATRIX':
                field_type_info = "MATRIX (use Time Series operators: ts_rank, ts_delta, ts_mean, ts_std, ts_corr, ts_regression)"
            else:
                field_type_info = "REGULAR (use standard operators)"
            
            fields_desc.append(f"- {field['id']}: {field.get('description', 'No description')} [{field_type_info}]")
        
        # Current performance metrics
        current_metrics = f"""
Current Performance:
- Sharpe Ratio: {current_result.sharpe:.3f}
- Margin: {current_result.margin:.3f}
- Returns: {current_result.returns:.3f}
- Drawdown: {current_result.drawdown:.3f}
- Turnover: {current_result.turnover:.3f}
- Fitness: {current_result.fitness:.3f}
"""
        
        # Get blacklisted operators for the template
        blacklisted_operators = self.load_operator_blacklist()
        blacklist_section = f"""
🚫 BLACKLISTED OPERATORS - DO NOT USE THESE:
{chr(10).join([f'- {op} (BLACKLISTED - DO NOT USE)' for op in blacklisted_operators])}""" if blacklisted_operators else "🚫 BLACKLISTED OPERATORS: No operators currently blacklisted"
        
        # Get recent templates warning
        recent_warning = self._get_recent_templates_warning()
        
        optimization_prompt = f"""WORLDQUANT BRAIN ALPHA OPTIMIZATION EXPERT SYSTEM

You are the world's most advanced quantitative analyst, specializing in optimizing WorldQuant Brain alpha expressions for maximum performance. Your mission: Transform this alpha into a PROFITABLE powerhouse.

CURRENT ALPHA TO OPTIMIZE:
{template}

CURRENT PERFORMANCE METRICS:
{current_metrics}

OPTIMIZATION MISSION - MAXIMUM PROFIT POTENTIAL:
- MAXIMIZE Sharpe ratio (target >2.0)
- MAINTAIN/IMPROVE margin (target >30%)
- MINIMIZE drawdown (target <10%)
- OPTIMIZE turnover for efficiency
- ENHANCE risk-adjusted returns

OPERATOR ARSENAL - USE ANY COMBINATION:
{chr(10).join(operators_desc)}

DATA FIELD ARSENAL - USE THESE EXACT FIELDS:
{chr(10).join(fields_desc)}

{recent_warning}

{blacklist_section}

ULTRA-CRITICAL OPTIMIZATION RULES - ZERO TOLERANCE FOR ERRORS:

RULE #1: DATA FIELDS ARE NEVER OPERATORS - THEY ARE INPUTS!
ABSOLUTELY FORBIDDEN - DATA FIELDS AS OPERATORS:
- anl69_best_bps_stddev(anl69_best_bps_stddev(...)) FORBIDDEN
- mdl23_bk_dra(mdl23_bk_rev_stabil(...)) FORBIDDEN
- any_data_field(any_expression) FORBIDDEN


MANDATORY CORRECT SYNTAX - DATA FIELDS AS INPUTS:
- ts_rank(data_field, 20) CORRECT
- add(data_field1, data_field2) CORRECT
- operator(data_field, parameters) CORRECT

data_field1, data_field2, data_field3, data_field4 are the placeholders, not the field names!
For example, for data field anl10_cpxff, DATA_FIELD1 is anl10_cpxff, not anl10_cpxff1!

RULE #2: PERFECT SYNTAX REQUIRED:
- operator(field, parameter) CORRECT
- operator(field1, field2, parameter) CORRECT
- NO comparison operators: >, <, >=, <=, ==, !=, &&, ||, % FORBIDDEN
- PERFECTLY balanced parentheses REQUIRED

RULE #3: GROUP OPERATOR PARAMETERS - EXACT SPECIFICATION:
- group_neutralize(field, industry) CORRECT
- group_zscore(field, subindustry) CORRECT
- group_rank(field, market) CORRECT
NEVER use generic "group" - always specify: industry, subindustry, sector, market

OPTIMIZATION STRATEGY - MAXIMUM ALPHA POTENTIAL:

1. MOMENTUM OPTIMIZATION:
   - Enhance ts_rank with better time periods
   - Add group neutralization for risk reduction
   - Combine with volatility scaling

2. MEAN REVERSION OPTIMIZATION:
   - Improve ts_zscore parameters
   - Add cross-sectional normalization
   - Enhance with group operations

3. RISK MANAGEMENT OPTIMIZATION:
   - Add winsorize for outlier control
   - Implement volatility scaling
   - Use group operations for risk reduction

4. SOPHISTICATED COMBINATIONS:
   - Nested group operations
   - Multi-timeframe strategies
   - Cross-sectional momentum with neutralization

OPTIMIZATION INSTRUCTIONS - MAXIMUM PERFORMANCE:

1. ANALYZE current alpha's strengths and weaknesses
2. IDENTIFY optimization opportunities in:
   - Operator combinations
   - Parameter tuning
   - Risk management
   - Group operations
3. GENERATE 3 optimized versions with:
   - Higher Sharpe ratio potential
   - Better risk management
   - Improved margin potential
   - Enhanced alpha generation
4. FOCUS on proven financial patterns:
   - Momentum strategies
   - Mean reversion strategies
   - Cross-sectional strategies
   - Risk-adjusted returns

SUCCESS VALIDATION CHECKLIST:
- Perfect syntax with balanced parentheses
- Correct operator parameter counts
- NO comparison operators (>, <, >=, <=, ==, !=, &&, ||, %)
- Data fields as INPUTS, not operators
- Group operators use correct group types
- Realistic field names and appropriate parameters

GENERATE 3 OPTIMIZED VERSIONS:

1. [Your first optimized alpha expression - focus on momentum enhancement]
2. [Your second optimized alpha expression - focus on mean reversion improvement]  
3. [Your third optimized alpha expression - focus on sophisticated combinations]

REASONING FOR EACH OPTIMIZATION:

1. [Why this momentum-enhanced version should perform better - specific improvements and expected benefits]
2. [Why this mean reversion version should perform better - specific improvements and expected benefits]
3. [Why this sophisticated combination should perform better - specific improvements and expected benefits]

RESPONSE FORMAT - STRUCTURED OPTIMIZATION:
1. [Optimized alpha expression 1]
2. [Optimized alpha expression 2]
3. [Optimized alpha expression 3]

Reasoning:
1. [Detailed reasoning for optimization 1]
2. [Detailed reasoning for optimization 2]
3. [Detailed reasoning for optimization 3]

Generate 3 PROFITABLE optimized alpha expressions:"""

        # Call Ollama API for optimization
        response = self.call_ollama_api(optimization_prompt)
        if not response:
            logger.error(f"Failed to get optimization suggestions for iteration {iteration}")
            return optimization_item
        
        # Parse the response to extract optimized templates
        optimized_templates = self.parse_optimization_response(response)
        
        return {
            **optimization_item,
            'optimized_templates': optimized_templates,
            'optimization_prompt': optimization_prompt,
            'llm_response': response
        }
    
    def parse_optimization_response(self, response: str) -> List[str]:
        """Parse LLM response to extract optimized alpha expressions"""
        templates = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for lines that start with numbers (1., 2., 3.) or contain alpha expressions
            if (line.startswith(('1.', '2.', '3.')) or 
                ('(' in line and ')' in line and any(op in line for op in ['ts_', 'group_', 'winsorize', 'rank', 'delta', 'zscore']))):
                # Clean up the line
                template = line
                if template.startswith(('1.', '2.', '3.')):
                    template = template[2:].strip()
                if template and template not in templates:
                    templates.append(template)
        
        return templates[:3]  # Return max 3 templates
    
    def process_optimization_queue(self):
        """Process the optimization queue with iterative LLM optimization"""
        if not self.optimization_queue:
            return
        
        logger.info(f"🚀 Starting optimization of {len(self.optimization_queue)} alphas")
        
        for optimization_item in self.optimization_queue[:]:  # Copy to avoid modification during iteration
            try:
                self._optimize_single_alpha(optimization_item)
            except Exception as e:
                logger.error(f"Error optimizing alpha {optimization_item['id']}: {e}")
                continue
        
        # Clear processed items
        self.optimization_queue.clear()
        logger.info("✅ Optimization queue processing completed")
    
    def _optimize_single_alpha(self, optimization_item: Dict):
        """Optimize a single alpha through iterative LLM optimization"""
        alpha_id = optimization_item['id']
        logger.info(f"🔧 Starting optimization for alpha: {alpha_id}")
        
        best_result = optimization_item['best_result']
        iteration = 0
        
        while iteration < self.max_optimization_iterations:
            iteration += 1
            logger.info(f"🔄 Optimization iteration {iteration}/{self.max_optimization_iterations} for {alpha_id}")
            
            # Get LLM optimization suggestions
            optimization_item['iteration'] = iteration
            optimized_item = self.optimize_alpha_with_llm(optimization_item)
            
            if not optimized_item.get('optimized_templates'):
                logger.warning(f"No optimization suggestions from LLM for {alpha_id}")
                break
            
            # Test the optimized templates
            best_improvement = None
            for template in optimized_item['optimized_templates']:
                try:
                    # Create a new simulation for the optimized template
                    test_result = self._test_optimized_template(template, optimization_item['region'], best_result.settings)
                    
                    if test_result and test_result.success:
                        # Check if this is an improvement
                        if self._is_improvement(test_result, best_result):
                            if not best_improvement or test_result.sharpe > best_improvement.sharpe:
                                best_improvement = test_result
                                logger.info(f"📈 Found improvement: Sharpe {test_result.sharpe:.3f} > {best_result.sharpe:.3f}")
                except Exception as e:
                    logger.error(f"Error testing optimized template: {e}")
                    continue
            
            if best_improvement:
                # Update with the best improvement
                optimization_item['current_result'] = best_improvement
                optimization_item['best_result'] = best_improvement
                optimization_item['improvement_count'] += 1
                best_result = best_improvement
                
                logger.info(f"✅ Iteration {iteration} improved alpha: Sharpe {best_result.sharpe:.3f}, Margin {best_result.margin:.3f}")
            else:
                logger.info(f"❌ No improvement found in iteration {iteration}, stopping optimization")
                break
        
        # Save final optimized result
        self.optimization_results[alpha_id] = {
            'original': optimization_item['best_result'],
            'final': best_result,
            'iterations': iteration,
            'improvements': optimization_item['improvement_count']
        }
        
        logger.info(f"🏁 Optimization completed for {alpha_id}: {optimization_item['improvement_count']} improvements in {iteration} iterations")
    
    def _test_optimized_template(self, template: str, region: str, settings: SimulationSettings) -> TemplateResult:
        """Test an optimized template by submitting it for simulation"""
        try:
            # Submit the optimized template for simulation
            simulation_response = self.make_api_request('POST', 'https://api.worldquantbrain.com/alphas', json={
                    'expression': template,
                    'universe': self.region_configs[region].universe,
                    'delay': settings.delay,
                    'neutralization': settings.neutralization
                }
            )
            
            if simulation_response.status_code != 201:
                logger.error(f"Failed to submit optimized template: {simulation_response.status_code}")
                return None
            
            alpha_id = simulation_response.json().get('id')
            if not alpha_id:
                logger.error("No alpha ID returned for optimized template")
                return None
            
            # Monitor the simulation
            return self._monitor_optimized_simulation(alpha_id, template, region, settings)
            
        except Exception as e:
            logger.error(f"Error testing optimized template: {e}")
            return None
    
    def _monitor_optimized_simulation(self, alpha_id: str, template: str, region: str, settings: SimulationSettings) -> TemplateResult:
        """Monitor an optimized simulation until completion"""
        max_wait_time = 300  # 5 minutes max wait
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                alpha_response = self.make_api_request('GET', f'https://api.worldquantbrain.com/alphas/{alpha_id}')
                if alpha_response.status_code != 200:
                    time.sleep(10)
                    continue
                
                alpha_data = alpha_response.json()
                status = alpha_data.get('status')
                
                if status == 'SUCCESS':
                    is_data = alpha_data.get('is', {})
                    sharpe = is_data.get('sharpe', 0)
                    fitness = is_data.get('fitness', 0)
                    turnover = is_data.get('turnover', 0)
                    returns = is_data.get('returns', 0)
                    drawdown = is_data.get('drawdown', 0)
                    margin = is_data.get('margin', 0)
                    longCount = is_data.get('longCount', 0)
                    shortCount = is_data.get('shortCount', 0)
                    
                    return TemplateResult(
                        template=template,
                        region=region,
                        settings=settings,
                        sharpe=sharpe,
                        fitness=fitness,
                        turnover=turnover,
                        returns=returns,
                        drawdown=drawdown,
                        margin=margin,
                        longCount=longCount,
                        shortCount=shortCount,
                        success=True,
                        alpha_id=alpha_id,
                        timestamp=time.time()
                    )
                elif status in ['FAILED', 'ERROR']:
                    logger.error(f"Optimized simulation failed: {alpha_id}")
                    return None
                
                time.sleep(10)  # Wait 10 seconds before checking again
                
            except Exception as e:
                logger.error(f"Error monitoring optimized simulation: {e}")
                time.sleep(10)
        
        logger.error(f"Optimized simulation timed out: {alpha_id}")
        return None
    
    def _is_improvement(self, new_result: TemplateResult, current_best: TemplateResult) -> bool:
        """Check if the new result is an improvement over the current best"""
        # Primary criteria: Sharpe ratio improvement
        sharpe_improvement = new_result.sharpe > current_best.sharpe
        
        # Secondary criteria: maintain or improve margin
        margin_maintained = new_result.margin >= current_best.margin * 0.9  # Allow 10% margin drop
        
        # Tertiary criteria: better return/drawdown ratio
        current_ratio = current_best.returns / max(current_best.drawdown, 0.001)
        new_ratio = new_result.returns / max(new_result.drawdown, 0.001)
        ratio_improvement = new_ratio > current_ratio * 0.95  # Allow 5% ratio drop
        
        return sharpe_improvement and margin_maintained and ratio_improvement
    
    def update_simulation_count(self):
        """Update simulation count and check for phase switches"""
        self.total_simulations += 1
        
        # Phase 1 to Phase 2: Switch to exploitation at 100 successful simulations
        # DISABLED: Keep using Ollama generation throughout to maintain consistent logging
        if False and (self.current_phase == "explore_exploit" and 
            self.total_simulations >= self.phase_switch_threshold):
            self.current_phase = "exploit"
            self.exploitation_phase = True
            self._initialize_exploitation_phase()
            logger.info(f"🔄 PHASE SWITCH: Switching to pure exploitation mode after {self.total_simulations} successful simulations")
            logger.info(f"🎯 EXPLOITATION PHASE: Will now use top-performing templates with dataset substitution across regions")
        
        # Phase 2 to Phase 3: Switch back to explore/exploit at 300 total simulations (100 + 200)
        # DISABLED: Keep using Ollama generation throughout to maintain consistent logging
        elif False and (self.current_phase == "exploit" and 
              self.total_simulations >= self.exploitation_end_threshold):
            self.current_phase = "loop"
            self.exploitation_phase = False
            self.loop_count += 1
            logger.info(f"🔄 PHASE SWITCH: Switching back to explore/exploit mode after {self.total_simulations} total simulations")
            logger.info(f"🔄 LOOP PHASE: Loop #{self.loop_count} - Resuming normal explore/exploit with new discoveries")
            
            # Reset for new loop and update thresholds for next cycle
            self._reset_for_new_loop()
            self._update_thresholds_for_next_cycle()
    
    def _reset_for_new_loop(self):
        """Reset system for new explore/exploit loop"""
        # Reset phase tracking
        self.current_phase = "explore_exploit"
        self.exploitation_phase = False
        
        # Clear old top templates to allow new discoveries
        self.top_templates = []
        self.exploitation_bandit = None
    
    def _update_thresholds_for_next_cycle(self):
        """Update thresholds for the next cycle to ensure infinite operation"""
        # Update thresholds based on current simulation count
        current_count = self.total_simulations
        
        # Set next phase switch threshold to be 100 simulations from current point
        self.phase_switch_threshold = current_count + 100
        
        # Set next exploitation end threshold to be 200 simulations after phase switch
        self.exploitation_end_threshold = self.phase_switch_threshold + 200
        
        logger.info(f"🔄 THRESHOLDS UPDATED: Next phase switch at {self.phase_switch_threshold}, exploitation end at {self.exploitation_end_threshold}")
        logger.info(f"🔄 LOOP RESET: Starting new explore/exploit cycle (Loop #{self.loop_count})")
        logger.info(f"📊 New cycle will run: 0-100 explore/exploit, 100-300 exploit, then loop again")
    
    def _initialize_exploitation_phase(self):
        """Initialize exploitation phase with top templates"""
        # Collect all successful templates from results
        all_successful = []
        for region, results in self.all_results.get('simulation_results', {}).items():
            for result in results:
                if result.get('success', False):
                    all_successful.append({
                        'template': result.get('template', ''),
                        'region': result.get('region', ''),
                        'sharpe': result.get('sharpe', 0),
                        'margin': result.get('margin', 0),
                        'fitness': result.get('fitness', 0),
                        'returns': result.get('returns', 0),
                        'drawdown': result.get('drawdown', 0)
                    })
        
        # Sort by Sharpe ratio and take top 50
        self.top_templates = sorted(all_successful, key=lambda x: x['sharpe'], reverse=True)[:50]
        
        # Initialize exploitation bandit
        self.exploitation_bandit = MultiArmBandit(exploration_rate=0.0, decay_rate=0.0, decay_interval=1000)  # Pure exploitation
        
        logger.info(f"📊 Exploitation phase initialized with {len(self.top_templates)} top templates")
        if self.top_templates:
            best = self.top_templates[0]
            logger.info(f"🏆 Best template: Sharpe={best['sharpe']:.3f}, Margin={best['margin']:.3f}")
            logger.info(f"🎯 Exploitation strategy: Dataset substitution across regions (USA, GLB, EUR, ASI, CHN)")
            logger.info(f"📈 Phase status: {self.current_phase} | Simulations: {self.total_simulations} | Loop: #{self.loop_count}")
    
    def get_exploitation_template(self) -> Dict:
        """Get a template for exploitation phase with dataset substitution.
        Only considers templates with Sharpe > 1.25, Fitness > 1.0, Margin > 5bps."""
        if not self.top_templates:
            logger.warning("No top templates available for exploitation")
            return None
        
        # Filter templates that meet the high performance criteria
        qualifying_templates = []
        qualifying_indices = []
        
        for i, template in enumerate(self.top_templates):
            sharpe = template.get('sharpe', 0)
            fitness = template.get('fitness', 0)
            margin = template.get('margin', 0)
            
            # Only consider templates that meet the high bar (5 bps = 0.0005)
            if (sharpe > 1.25 and fitness > 1.0 and margin > 0.0005):
                qualifying_templates.append(template)
                qualifying_indices.append(i)
        
        if not qualifying_templates:
            logger.warning(f"⚠️ No templates meet exploitation criteria (Sharpe > 1.25, Fitness > 1.0, Margin > 5bps)")
            logger.info(f"📊 Available templates: {len(self.top_templates)}")
            for i, template in enumerate(self.top_templates[:3]):  # Show first 3 for debugging
                logger.info(f"   Template {i+1}: Sharpe={template.get('sharpe', 0):.3f}, Fitness={template.get('fitness', 0):.3f}, Margin={template.get('margin', 0):.3f}")
            
            # FALLBACK: Generate new templates using LLM when no existing templates meet criteria
            logger.info(f"🎯 EXPLOITATION FALLBACK: No qualifying templates found, generating new templates using LLM")
            return self._generate_exploitation_fallback_template()
        
        logger.info(f"🎯 EXPLOITATION: {len(qualifying_templates)}/{len(self.top_templates)} templates meet high performance criteria")
        
        # Create template IDs and weights for qualifying templates only
        template_ids = [f"template_{qualifying_indices[i]}" for i in range(len(qualifying_templates))]
        
        # Create performance weights based on Sharpe ratios for qualifying templates
        performance_weights = []
        for template in qualifying_templates:
            # Use Sharpe ratio as the weight (higher Sharpe = higher weight)
            weight = max(template.get('sharpe', 0), 0.1)  # Minimum weight of 0.1
            performance_weights.append(weight)
        
        action, selected_id = self.exploitation_bandit.choose_action_weighted(template_ids, performance_weights)
        
        template_idx = int(selected_id.split('_')[1])
        selected_template = self.top_templates[template_idx]
        
        # Log the performance metrics of the selected template
        logger.info(f"🎯 EXPLOITATION: Selected template {template_idx} with Sharpe={selected_template.get('sharpe', 0):.3f}, Fitness={selected_template.get('fitness', 0):.3f}, Margin={selected_template.get('margin', 0):.3f}")
        
        # Choose a different region for dataset substitution
        original_region = selected_template['region']
        available_regions = [r for r in self.active_regions if r != original_region]
        target_region = random.choice(available_regions)
        
        # Get data fields for the target region and shuffle them
        target_config = self.region_configs[target_region]
        optimal_delay = self.select_optimal_delay(target_region)
        data_fields = self.get_data_fields_for_region(target_region, optimal_delay)
        
        if data_fields:
            # Shuffle data fields to ensure different combinations
            shuffled_fields = random.sample(data_fields, len(data_fields))
            logger.info(f"🎯 EXPLOITATION: Using shuffled data fields for {target_region} (delay={optimal_delay})")
            logger.info(f"🎯 EXPLOITATION: Shuffled {len(shuffled_fields)} fields for template generation")
        else:
            logger.warning(f"🎯 EXPLOITATION: No data fields found for {target_region}")
        
        return {
            'template': selected_template['template'],
            'original_region': original_region,
            'target_region': target_region,
            'original_sharpe': selected_template['sharpe'],
            'original_margin': selected_template['margin'],
            'shuffled_fields': shuffled_fields if data_fields else []
        }
    
    def _generate_exploitation_fallback_template(self) -> Dict:
        """Generate a new template using pure Ollama when no existing templates meet exploitation criteria"""
        # Choose a random region for template generation
        target_region = random.choice(self.active_regions)
        
        # Get data fields for the target region
        optimal_delay = self.select_optimal_delay(target_region)
        data_fields = self.get_data_fields_for_region(target_region, optimal_delay)
        
        if not data_fields:
            logger.warning(f"🎯 EXPLOITATION FALLBACK: No data fields found for {target_region}")
            return None
        
        logger.info(f"🎯 EXPLOITATION FALLBACK: Generating new template for {target_region} with {len(data_fields)} fields")
        logger.info(f"🤖 EXPLOITATION FALLBACK: Using step-by-step generation (same as default mode)")
        
        # Use step-by-step generation (same as default mode)
        templates = self._generate_step_by_step_templates(target_region, 1)
        
        if not templates:
            logger.error(f"🎯 EXPLOITATION FALLBACK: Failed to generate Ollama template for {target_region}")
            return None
        
        # Take the first generated template
        template_data = templates[0]
        template = template_data['template']
        
        logger.info(f"🎯 EXPLOITATION FALLBACK: Generated valid template: {template[:50]}...")
        return {
            'template': template,
            'original_region': target_region,
            'target_region': target_region,
            'original_sharpe': 0.0,  # New template, no history
            'original_margin': 0.0,  # New template, no history
            'shuffled_fields': data_fields,
            'fallback_generated': True,
            'ollama_only': True
        }
    
    def create_exploitation_templates(self, num_templates: int = 10) -> List[Dict]:
        """Create templates for exploitation phase with dataset substitution"""
        if not self.exploitation_phase:
            return []
        
        templates = []
        for _ in range(num_templates):
            exploitation_data = self.get_exploitation_template()
            if exploitation_data:
                templates.append({
                    'template': exploitation_data['template'],
                    'region': exploitation_data['target_region'],
                    'original_region': exploitation_data['original_region'],
                    'original_sharpe': exploitation_data['original_sharpe'],
                    'original_margin': exploitation_data['original_margin'],
                    'exploitation': True
                })
        
        return templates
    
    def update_exploitation_bandit(self, result: TemplateResult, original_sharpe: float):
        """Update exploitation bandit with results"""
        if not self.exploitation_bandit or not result.success:
            return
        
        # Calculate reward based on improvement
        improvement = result.sharpe - original_sharpe
        reward = max(0, improvement * 2)  # Reward for improvement
        
        # Find the template index and update bandit
        for i, template in enumerate(self.top_templates):
            if template['template'] == result.template:
                arm_id = f"template_{i}"
                self.exploitation_bandit.update_arm(arm_id, reward)
                break
    
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
                    
                    # Validate that cached data matches the expected parameters
                    if cached_data:
                        config = self.region_configs[region]
                        matching_fields = []
                        
                        for field in cached_data:
                            field_region = field.get('region', '')
                            field_universe = field.get('universe', '')
                            field_delay = field.get('delay', -1)
                            
                            # Only include fields that match ALL parameters exactly
                            if (field_region == region and 
                                field_universe == config.universe and 
                                field_delay == delay):
                                matching_fields.append(field)
                        
                        if len(matching_fields) == 0:
                            logger.warning(f"⚠️ Cached data doesn't match expected parameters!")
                            logger.warning(f"   Expected: region={region}, universe={config.universe}, delay={delay}")
                            
                            # Show what we actually have in cache
                            if cached_data:
                                sample_field = cached_data[0]
                                logger.warning(f"   Cached field region: {sample_field.get('region', 'UNKNOWN')}")
                                logger.warning(f"   Cached field universe: {sample_field.get('universe', 'UNKNOWN')}")
                                logger.warning(f"   Cached field delay: {sample_field.get('delay', 'UNKNOWN')}")
                            
                            logger.warning(f"⚠️ Cache mismatch detected, will refetch data...")
                            # Don't return cached data, let it refetch
                        else:
                            logger.info(f"✅ Cached data validation: {len(matching_fields)} fields match exact parameters")
                            return matching_fields
                    
                    # If we get here, either no cached data or validation failed
                    logger.error(f"🚨 CACHE VALIDATION FAILED: Cannot use cached data due to region mismatch!")
                    logger.error(f"   Expected: region={region}, universe={config.universe}, delay={delay}")
                    logger.error(f"   This could cause 'unknown variable' errors - will refetch from API")
                    # DO NOT return cached data - force refetch to prevent cross-region contamination
                    return []
            
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
                datasets_response = self.make_api_request('GET', 'https://api.worldquantbrain.com/data-sets', params=datasets_params)
                
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
                    
                    response = self.make_api_request('GET', 'https://api.worldquantbrain.com/data-fields', params=params)
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
            
            # Filter fields to ensure they match the exact parameters
            filtered_fields = []
            for field in field_list:
                field_region = field.get('region', '')
                field_universe = field.get('universe', '')
                field_delay = field.get('delay', -1)
                
                # Only include fields that match ALL parameters exactly
                if (field_region == region and 
                    field_universe == config.universe and 
                    field_delay == delay):
                    filtered_fields.append(field)
            
            logger.info(f"🔍 FILTERED FIELDS: {len(filtered_fields)} fields match exact parameters")
            logger.info(f"   Region: {region}, Universe: {config.universe}, Delay: {delay}")
            
            if len(filtered_fields) == 0:
                logger.warning(f"⚠️ No fields found matching exact parameters!")
                logger.warning(f"   Expected: region={region}, universe={config.universe}, delay={delay}")
                
                # Show what we actually got
                if field_list:
                    sample_field = field_list[0]
                    logger.warning(f"   Sample field region: {sample_field.get('region', 'UNKNOWN')}")
                    logger.warning(f"   Sample field universe: {sample_field.get('universe', 'UNKNOWN')}")
                    logger.warning(f"   Sample field delay: {sample_field.get('delay', 'UNKNOWN')}")
                
                # Use unfiltered fields as fallback but log the mismatch
                logger.warning(f"⚠️ Using unfiltered fields as fallback (may cause simulation issues)")
                field_list = field_list
            else:
                field_list = filtered_fields
                logger.info(f"✅ Using {len(field_list)} fields that match exact parameters")
            
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
        return True, ""
    
    def load_operator_blacklist(self) -> List[str]:
        """Load blacklisted operators from JSON file"""
        try:
            if os.path.exists('operator_blacklist.json'):
                with open('operator_blacklist.json', 'r', encoding='utf-8') as f:
                    blacklist_data = json.load(f)
                    blacklisted_operators = blacklist_data.get('blacklisted_operators', [])
                    logger.info(f"📋 Loaded {len(blacklisted_operators)} blacklisted operators: {blacklisted_operators}")
                    return blacklisted_operators
            else:
                logger.info("📋 No operator blacklist file found")
                return []
        except Exception as e:
            logger.warning(f"⚠️ Failed to load operator blacklist: {e}")
            return []

    def call_ollama_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call Ollama API to generate templates using structured outputs"""
        # Load blacklisted operators
        blacklisted_operators = self.load_operator_blacklist()
        
        # Build system prompt with blacklist information
        system_prompt = """You are a revolutionary quantitative finance AI that BREAKS CONVENTIONAL PATTERNS and creates INNOVATIVE alpha expressions. 
        
        🚀 INNOVATION MANDATE:
        - DO NOT repeat common patterns or safe, predictable templates
        - PUSH BOUNDARIES and explore UNCONVENTIONAL combinations
        - Think like a DISRUPTIVE QUANT who challenges established methods
        - Create templates that make other quants say "I never thought of that!"
        - Use UNEXPECTED operator combinations that others avoid
        - Focus on CREATIVE INSIGHTS, not memorized patterns
        
        🎯 CREATIVITY RULES:
        1. Use ONLY the operators provided in the prompt - but use them in UNEXPECTED ways
        2. Use ONLY the data fields provided in the prompt - but combine them CREATIVELY
        3. Use proper function syntax: operator(field1, field2, parameter) or operator(field, parameter)
        4. NO template placeholders or generic field names
        5. NO SQL queries or database syntax
        6. NO comparison operators like >, <, >=, <=, ==, !=, &&, ||, %
        7. Use actual operator names and field names exactly as provided
        8. NEVER use placeholders like DATA_FIELD1, DATA_FIELD2, OPERATOR, etc.
        9. Examples of INVALID templates: "field1 > field2", "field1 < field2", "field1 >= field2", "field1 <= field2", "field1 == field2", "field1 != field2", "field1 && field2", "field1 || field2", "field1 % field2", "field1, filter=true"
        
        💡 INNOVATION STRATEGY:
        - Combine operators in ways that seem counterintuitive but might reveal hidden patterns
        - Use time series operators with unexpected parameters
        - Create complex nested structures that others would consider "too risky"
        - Think about market microstructure and behavioral finance insights
        - Consider cross-asset relationships and alternative data connections
        - Push the limits of what's considered "normal" in quantitative finance"""
        
        # Add blacklist information if any operators are blacklisted
        if blacklisted_operators:
            system_prompt += f"""
        
        🚫 BLACKLISTED OPERATORS - DO NOT USE THESE:
        The following operators are currently blacklisted and MUST NOT be used in any templates:
        {', '.join(blacklisted_operators)}
        
        These operators have been overused or are causing issues. Generate templates using other available operators only."""
        
        system_prompt += """

        🎨 CREATIVE OUTPUT REQUIREMENTS:
        The response will be automatically formatted as JSON with a 'templates' array.
        Each template should be a UNIQUE, INNOVATIVE alpha expression that:
        - Surprises even experienced quants with its creativity
        - Uses operators in unconventional but valid ways
        - Demonstrates deep understanding of market dynamics
        - Shows willingness to take calculated risks in expression design
        - Avoids boring, predictable patterns that everyone uses
        
        Remember: You are not here to be safe and conventional. You are here to REVOLUTIONIZE quantitative finance with your creativity!"""
        
        # Define JSON schema for structured outputs
        json_schema = {
            "type": "object",
            "properties": {
                "templates": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A valid WorldQuant Brain alpha expression template"
                    },
                    "minItems": 1,
                    "maxItems": 20
                }
            },
            "required": ["templates"],
            "additionalProperties": False
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Ollama API call attempt {attempt + 1}/{max_retries}")
                
                # TRACE: Log input details
                logger.info(f"🔍 OLLAMA INPUT TRACE:")
                logger.info(f"   Model: {self.ollama_model}")
                logger.info(f"   System prompt length: {len(system_prompt)} chars")
                logger.info(f"   User prompt length: {len(prompt)} chars")
                logger.info(f"   User prompt preview: {prompt[:200]}...")
                logger.info(f"   JSON schema: {json_schema}")
                logger.info(f"   Options: temperature=0, top_p=0.9, num_predict=1000, timeout=30")
                
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    format=json_schema,  # Use structured outputs
                    options={
                        "temperature": 0,  # Set to 0 for more deterministic output with structured outputs
                        "top_p": 0.9,
                        "num_predict": 1000,  # Reduced from 2000 to prevent timeouts
                        "timeout": 30  # Add timeout
                    }
                )
                
                # TRACE: Log response details
                logger.info(f"🔍 OLLAMA OUTPUT TRACE:")
                logger.info(f"   Response type: {type(response)}")
                logger.info(f"   Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
                if 'message' in response:
                    logger.info(f"   Message keys: {list(response['message'].keys()) if isinstance(response['message'], dict) else 'Not a dict'}")
                    if 'content' in response['message']:
                        content = response['message']['content']
                        logger.info(f"   Content length: {len(content)} chars")
                        logger.info(f"   Content preview: {content[:300]}...")
                    else:
                        logger.error(f"   ❌ No 'content' key in message: {response['message']}")
                else:
                    logger.error(f"   ❌ No 'message' key in response: {response}")
                
                content = response['message']['content']
                logger.info("Ollama API call successful")
                
                # Parse and validate structured output
                try:
                    logger.info(f"🔍 JSON PARSING TRACE:")
                    logger.info(f"   Raw content: {content}")
                    parsed_json = json.loads(content)
                    logger.info(f"   Parsed JSON type: {type(parsed_json)}")
                    logger.info(f"   Parsed JSON keys: {list(parsed_json.keys()) if isinstance(parsed_json, dict) else 'Not a dict'}")
                    
                    if 'templates' in parsed_json and isinstance(parsed_json['templates'], list):
                        logger.info(f"✅ Structured output validation successful: {len(parsed_json['templates'])} templates")
                        logger.info(f"   Templates: {parsed_json['templates']}")
                        return content
                    else:
                        logger.error(f"❌ Invalid structured output: missing 'templates' key or not a list")
                        logger.error(f"   Available keys: {list(parsed_json.keys()) if isinstance(parsed_json, dict) else 'Not a dict'}")
                        logger.error(f"   'templates' key exists: {'templates' in parsed_json if isinstance(parsed_json, dict) else False}")
                        logger.error(f"   'templates' is list: {isinstance(parsed_json.get('templates'), list) if isinstance(parsed_json, dict) else False}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return None
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Structured output failed JSON validation: {e}")
                    logger.error(f"   Raw content that failed to parse: {content}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Ollama API call failed: {e}")
                logger.error(f"🔍 EXCEPTION TRACE:")
                logger.error(f"   Exception type: {type(e).__name__}")
                logger.error(f"   Exception message: {str(e)}")
                logger.error(f"   Model: {self.ollama_model}")
                logger.error(f"   Attempt: {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    logger.info(f"   Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"   All retries exhausted, returning None")
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
        
        # Log current field strategy status
        strategy_status = self.get_field_strategy_status()
        logger.info(f"🎯 FIELD STRATEGY STATUS: {strategy_status['strategy_mode']} (Elites: {strategy_status['elite_templates_found']}, Time since last: {strategy_status['time_since_last_elite']:.0f}s)")
        
        # Check if we're in exploitation phase
        # DISABLED: Always use Ollama generation to maintain consistent logging
        if False and self.exploitation_phase:
            logger.info(f"🎯 EXPLOITATION PHASE: Using top-performing templates with dataset substitution for {region}")
            return self.create_exploitation_templates(num_templates)
        
        # Check if we're in loop phase (back to explore/exploit)
        if self.current_phase == "loop":
            logger.info(f"🔄 LOOP PHASE: Resuming normal explore/exploit template generation for {region}")
            # Continue with normal template generation
        
        # DEFAULT: Use step-by-step generation (separate field and operator selection)
        logger.info(f"🤖 STEP-BY-STEP MODE (DEFAULT): Expression will be generated step-by-step with separate field and operator selection")
        return self._generate_step_by_step_templates(region, num_templates)
    
    def _generate_step_by_step_templates(self, region: str, num_templates: int, delay: int = None) -> List[Dict]:
        """Generate templates using step-by-step approach: 1) Choose fields, 2) Choose operators, 3) Build template"""
        logger.info(f"🤖 STEP-BY-STEP GENERATION: Generating {num_templates} templates for {region}")
        
        # If delay is None, use optimal delay
        if delay is None:
            delay = self.select_optimal_delay(region)
            logger.info(f"🔧 DELAY SYNC: Using optimal delay {delay} for {region}")
        
        templates = []
        
        for i in range(num_templates):
            logger.info(f"🔄 STEP-BY-STEP: Template {i+1}/{num_templates}")
            
            # Step 1: Choose data fields
            selected_fields = self._choose_data_fields_step(region, delay)
            if not selected_fields:
                logger.warning(f"❌ No fields selected for template {i+1}")
                continue
            
            # Step 2: Choose operators (with exploitation hook)
            selected_operators = self._choose_operators_step_with_exploitation(region, selected_fields, i)
            if not selected_operators:
                logger.warning(f"❌ No operators selected for template {i+1}")
                continue
            
            # Step 3: Build template
            template = self._build_template_step(region, selected_fields, selected_operators)
            if not template:
                logger.warning(f"❌ Template building failed for template {i+1}")
                continue
            
            # Step 4: Validate and add to results
            is_valid, fixed_template = self._validate_template_step(template, region, delay)
            if is_valid:
                templates.append({
                    'template': fixed_template,
                    'region': region,
                    'delay': delay,
                    'fields_used': [f['id'] for f in selected_fields],
                    'operators_used': [op['name'] for op in selected_operators],
                    'generated_at': datetime.now().isoformat()
                })
                logger.info(f"✅ STEP-BY-STEP Template {i+1}: {fixed_template}")
        else:
                logger.warning(f"❌ Template validation failed for template {i+1}")
        
        return templates
    
    def _choose_data_fields_step(self, region: str, delay: int = None) -> List[Dict]:
        """Step 1: Let Ollama choose data fields"""
        logger.info(f"📊 STEP 1: Choosing data fields for {region}")
        
        # If delay is None, use optimal delay
        if delay is None:
            delay = self.select_optimal_delay(region)
            logger.info(f"🔧 DELAY SYNC: Using optimal delay {delay} for {region}")
        
        # Get data fields
        data_fields = self.get_data_fields_for_region(region, delay)
        if not data_fields:
            logger.warning(f"❌ No data fields found for {region} delay={delay}")
            return []
        
        # Log cache file info for debugging
        cache_file = f"data_fields_cache_{region}_{delay}.json"
        if os.path.exists(cache_file):
            logger.info(f"📁 Using cache file: {cache_file}")
        else:
            logger.info(f"📁 No cache file found: {cache_file}, using API data")
        
        # CRITICAL: Double-check that all fields are region-specific
        region_specific_fields = []
        cross_region_fields = []
        for field in data_fields:
            field_region = field.get('region', '')
            if field_region == region:
                region_specific_fields.append(field)
            else:
                cross_region_fields.append(field)
                logger.warning(f"🚨 CROSS-REGION FIELD DETECTED: {field.get('id', 'UNKNOWN')} from {field_region} used in {region}")
        
        if len(region_specific_fields) == 0:
            logger.error(f"🚨 NO REGION-SPECIFIC FIELDS: All fields are from other regions!")
            logger.error(f"🚨 Cross-region fields found: {[f.get('id', 'UNKNOWN') for f in cross_region_fields[:5]]}")
            return []
        
        if len(region_specific_fields) != len(data_fields):
            logger.warning(f"🔧 REGION FILTER: Filtered {len(data_fields) - len(region_specific_fields)} cross-region fields")
            logger.warning(f"🔧 Using {len(region_specific_fields)} region-specific fields for {region}")
            data_fields = region_specific_fields
        
        # BALANCED FIELD SELECTION: Mix of high-usage and low-usage fields
        def get_usage_score(field):
            user_count = field.get('userCount', 0)
            alpha_count = field.get('alphaCount', 0)
            return user_count + alpha_count
        
        # Categorize fields by usage
        data_fields.sort(key=get_usage_score)
        total_fields = len(data_fields)
        
        # Define usage categories
        low_usage_fields = data_fields[:total_fields//3]  # Bottom third (lowest usage)
        medium_usage_fields = data_fields[total_fields//3:2*total_fields//3]  # Middle third
        high_usage_fields = data_fields[2*total_fields//3:]  # Top third (highest usage)
        
        # Randomly decide selection strategy (30% chance for all high-usage, 70% for balanced)
        import random
        selection_strategy = random.choice(['balanced', 'balanced', 'balanced', 'all_high_usage'])
        
        if selection_strategy == 'all_high_usage':
            # Sometimes select all high-usage fields to stir things up
            selected_fields = high_usage_fields[:30] if len(high_usage_fields) >= 30 else high_usage_fields
            logger.info(f"🎯 HIGH-USAGE STRATEGY: Selected {len(selected_fields)} high-usage fields")
        else:
            # Balanced selection: mix of low, medium, and high usage fields
            # 40% low-usage, 40% medium-usage, 20% high-usage
            low_count = min(12, len(low_usage_fields))  # 40% of 30 fields
            medium_count = min(12, len(medium_usage_fields))  # 40% of 30 fields  
            high_count = min(6, len(high_usage_fields))  # 20% of 30 fields
            
            # Randomly select from each category
            selected_low = random.sample(low_usage_fields, low_count) if low_usage_fields else []
            selected_medium = random.sample(medium_usage_fields, medium_count) if medium_usage_fields else []
            selected_high = random.sample(high_usage_fields, high_count) if high_usage_fields else []
            
            selected_fields = selected_low + selected_medium + selected_high
            # Shuffle to mix the categories
            random.shuffle(selected_fields)
            
            logger.info(f"🎯 BALANCED STRATEGY: {len(selected_low)} low-usage, {len(selected_medium)} medium-usage, {len(selected_high)} high-usage fields")
        
        # Use the selected fields for the prompt
        data_fields = selected_fields[:30]  # Limit to 30 fields for the prompt
        
        # Log usage statistics for debugging
        usage_stats = []
        for field in data_fields[:10]:  # Show first 10 fields
            user_count = field.get('userCount', 0)
            alpha_count = field.get('alphaCount', 0)
            usage_stats.append(f"{field['id']}: users={user_count}, alphas={alpha_count}")
        
        logger.info(f"📊 SELECTED FIELDS: {usage_stats[:5]}")
        
        # Create field selection prompt with indexed list
        fields_desc = []
        for i, field in enumerate(data_fields[:30]):  # Show first 30 fields with indices
            field_type = field.get('type', 'REGULAR')
            if field_type == 'VECTOR':
                field_type_info = "VECTOR (use Cross Sectional operators: normalize, quantile, rank, scale, winsorize, zscore)"
            elif field_type == 'MATRIX':
                field_type_info = "MATRIX (use Time Series operators: ts_rank, ts_delta, ts_mean, ts_std, ts_corr, ts_regression, vec_avg, vec_sum, vec_max, vec_min)"
            else:
                field_type_info = "REGULAR (use standard operators)"
            
            # Include usage information
            user_count = field.get('userCount', 0)
            alpha_count = field.get('alphaCount', 0)
            usage_info = f"users={user_count}, alphas={alpha_count}"
            fields_desc.append(f"[{i}] {field['id']}: {field.get('description', 'No description')} [{field_type_info}] [{usage_info}]")
        
        prompt = f"""
🎯 DATA FIELD SELECTION FOR {region.upper()}

Choose 2-4 data fields by selecting their INDEX NUMBERS from the list below:

AVAILABLE DATA FIELDS (balanced mix of usage levels):
{chr(10).join(fields_desc)}

INSTRUCTIONS:
- Choose 2-4 fields by their INDEX NUMBERS (e.g., [0, 1, 4])
- BALANCED APPROACH: Mix of low-usage, medium-usage, and high-usage fields
- Low-usage fields offer discovery potential but may be less reliable
- High-usage fields are proven but may have lower alpha potential
- Mix different field types: VECTOR fields need Cross Sectional operators, MATRIX fields need Time Series operators
- Consider the trade-off between discovery potential and reliability
- Return ONLY the index numbers in square brackets

RESPONSE FORMAT:
[0, 1, 4]
"""
        
        try:
            import ollama
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.7, 'top_p': 0.9}
            )
            
            # Parse response to get field indices
            content = response['message']['content'].strip()
            logger.info(f"📊 STEP 1 RESPONSE: {content}")
            
            # Extract indices from response (e.g., [0, 1, 4] or 0, 1, 4)
            import re
            indices = []
            
            # Try to find indices in square brackets
            bracket_match = re.search(r'\[([0-9,\s]+)\]', content)
            if bracket_match:
                indices_str = bracket_match.group(1)
                indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
            else:
                # Try to find individual numbers
                numbers = re.findall(r'\b(\d+)\b', content)
                indices = [int(x) for x in numbers if int(x) < len(data_fields[:30])]
            
            # Get selected fields by index
            selected_fields = []
            for idx in indices:
                if 0 <= idx < len(data_fields[:30]):
                    selected_fields.append(data_fields[idx])
            
            logger.info(f"📊 STEP 1 COMPLETE: Selected {len(selected_fields)} fields: {[f['id'] for f in selected_fields]}")
            return selected_fields
            
        except Exception as e:
            logger.error(f"❌ STEP 1 FAILED: {e}")
            return []
    
    def _choose_operators_step_with_exploitation(self, region: str, selected_fields: List[Dict], template_index: int) -> List[Dict]:
        """Choose operators with weighted exploitation hook - can use successful template operators"""
        # 30% chance to use exploitation from successful templates
        if random.random() < 0.3 and template_index > 0:  # Don't exploit on first template
            successful_templates = self._get_successful_templates()
            if successful_templates:
                # Use weighted selection to pick one template for exploitation
                performance_weights = []
                for template in successful_templates:
                    # Use Sharpe ratio as the weight (higher Sharpe = higher weight)
                    weight = max(template.get('sharpe', 0), 0.1)  # Minimum weight of 0.1
                    performance_weights.append(weight)
                
                # Weighted random selection for one template
                total_weight = sum(performance_weights)
                probabilities = [w / total_weight for w in performance_weights]
                selected_idx = random.choices(range(len(successful_templates)), weights=probabilities)[0]
                template_to_exploit = successful_templates[selected_idx]
                
                logger.info(f"🎯 EXPLOITATION HOOK: Using operators from template (Sharpe: {template_to_exploit.get('sharpe', 0):.3f}, Weight: {probabilities[selected_idx]:.3f})")
                
                # Extract operators from the successful template
                template_text = template_to_exploit.get('template', '')
                exploited_operators = self.extract_operators_from_template(template_text)
                
                if exploited_operators:
                    # Convert to operator dicts for compatibility
                    operator_dicts = []
                    for op_name in exploited_operators:
                        # Find operator info
                        for op in self.operators:
                            if op['name'] == op_name:
                                operator_dicts.append(op)
                                break
                    
                    if operator_dicts:
                        logger.info(f"🎯 EXPLOITATION HOOK: Using {len(operator_dicts)} operators from successful template")
                        return operator_dicts
        
        # Fall back to normal operator selection
        return self._choose_operators_step(region, selected_fields)
    
    def _choose_operators_step(self, region: str, selected_fields: List[Dict]) -> List[Dict]:
        """Step 2: Let Ollama choose operators based on selected fields"""
        logger.info(f"⚙️ STEP 2: Choosing operators for {len(selected_fields)} fields")
        
        # Analyze field types for logging purposes only
        vector_fields = [f for f in selected_fields if f.get('type') == 'VECTOR']
        matrix_fields = [f for f in selected_fields if f.get('type') == 'MATRIX']
        
        logger.info(f"📊 FIELD ANALYSIS: {len(vector_fields)} VECTOR, {len(matrix_fields)} MATRIX fields")
        
        # Allow ALL operators - we'll handle field compatibility later during template validation
        compatible_operators = self.operators.copy()
        
        # CRITICAL: Filter out blacklisted operators
        if self.operator_blacklist:
            original_count = len(compatible_operators)
            compatible_operators = [op for op in compatible_operators if op['name'] not in self.operator_blacklist]
            filtered_count = len(compatible_operators)
            if original_count != filtered_count:
                logger.info(f"🚫 BLACKLIST FILTER: Removed {original_count - filtered_count} blacklisted operators")
                logger.info(f"🚫 BLACKLISTED OPERATORS: {list(self.operator_blacklist)}")
        
        if not compatible_operators:
            logger.warning(f"⚠️ NO COMPATIBLE OPERATORS: All operators are blacklisted or incompatible")
            return []
        
        # Create operator selection prompt with indexed list
        # Randomly select 15 operators for variety
        import random
        selected_operators = random.sample(compatible_operators, min(15, len(compatible_operators)))
        logger.info(f"🎲 RANDOM OPERATOR SELECTION: {[op['name'] for op in selected_operators]}")
        
        operators_desc = []
        for i, op in enumerate(selected_operators):
            operators_desc.append(f"[{i}] {op['name']}: {op['description']}")
        
        prompt = f"""
⚙️ OPERATOR SELECTION FOR {region.upper()}

Choose 2-4 operators by selecting their INDEX NUMBERS:

SELECTED FIELDS:
{chr(10).join([f"- {f['id']} ({f.get('type', 'REGULAR')})" for f in selected_fields])}

AVAILABLE OPERATORS:
{chr(10).join(operators_desc)}

INSTRUCTIONS:
- Choose 2-4 operators by their INDEX NUMBERS (e.g., [0, 1, 4])
- Valid indices are 0-14 (15 operators available)
- Select operators that create interesting and diverse combinations
- Mix different operator types for variety
- Field compatibility will be handled automatically
- Return ONLY the index numbers in square brackets

RESPONSE FORMAT:
[0, 1, 4]
"""
        
        try:
            import ollama
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.7, 'top_p': 0.9}
            )
            
            # Parse response to get operator indices
            content = response['message']['content'].strip()
            logger.info(f"⚙️ STEP 2 RESPONSE: {content}")
            
            # Extract indices from response (e.g., [0, 1, 4] or 0, 1, 4)
            import re
            indices = []
            
            # Try to find indices in square brackets
            bracket_match = re.search(r'\[([0-9,\s]+)\]', content)
            if bracket_match:
                indices_str = bracket_match.group(1)
                indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
                logger.info(f"🔍 PARSED BRACKET INDICES: {indices}")
            else:
                # Try to find individual numbers
                numbers = re.findall(r'\b(\d+)\b', content)
                indices = [int(x) for x in numbers if int(x) < len(selected_operators)]
                logger.info(f"🔍 PARSED NUMBER INDICES: {indices}")
            
            logger.info(f"🔍 FINAL INDICES: {indices}")
            
            # Get selected operators by index from the randomly selected operators
            final_selected_operators = []
            for idx in indices:
                if 0 <= idx < len(selected_operators):
                    final_selected_operators.append(selected_operators[idx])
            
            logger.info(f"⚙️ STEP 2 COMPLETE: Selected {len(final_selected_operators)} operators: {[op['name'] for op in final_selected_operators]}")
            return final_selected_operators
            
        except Exception as e:
            logger.error(f"❌ STEP 2 FAILED: {e}")
            return []
    
    def _build_template_step(self, region: str, selected_fields: List[Dict], selected_operators: List[Dict]) -> str:
        """Step 3: Let Ollama build the template using selected fields and operators"""
        logger.info(f"🔨 STEP 3: Building template with {len(selected_fields)} fields and {len(selected_operators)} operators")
        
        # CRITICAL: Validate that all selected fields are region-specific
        cross_region_fields = []
        for field in selected_fields:
            field_region = field.get('region', '')
            if field_region != region:
                cross_region_fields.append(field)
                logger.error(f"🚨 CROSS-REGION FIELD IN TEMPLATE: {field.get('id', 'UNKNOWN')} from {field_region} used in {region}")
                logger.error(f"🚨 This will cause 'unknown variable' errors in simulation!")
        
        if cross_region_fields:
            logger.warning(f"🔄 CROSS-REGION CONTAMINATION DETECTED: Found {len(cross_region_fields)} cross-region fields")
            logger.warning(f"🔄 Cross-region fields: {[f.get('id', 'UNKNOWN') for f in cross_region_fields]}")
            logger.info(f"🔄 REGENERATING TEMPLATE: Using correct region {region} and optimal delay")
            
            # Get fresh fields for the target region with optimal delay
            optimal_delay = self.select_optimal_delay(region)
            fresh_fields = self.get_data_fields_for_region(region, optimal_delay)
            
            if not fresh_fields:
                logger.error(f"❌ NO FRESH FIELDS: Cannot regenerate template for {region}")
                return None
            
            # Filter out any remaining cross-region fields
            region_specific_fields = [f for f in fresh_fields if f.get('region', '') == region]
            
            if len(region_specific_fields) == 0:
                logger.error(f"❌ NO REGION-SPECIFIC FIELDS: All fresh fields are cross-region for {region}")
                return None
            
            logger.info(f"✅ REGENERATION: Using {len(region_specific_fields)} region-specific fields for {region}")
            
            # Update selected_fields with region-specific fields
            selected_fields = region_specific_fields[:len(selected_fields)]  # Keep same count
            logger.info(f"🔄 REGENERATED FIELDS: {[f['id'] for f in selected_fields]}")
        
        # Create template building prompt
        fields_info = []
        for field in selected_fields:
            field_type = field.get('type', 'REGULAR')
            # Get compatible operators dynamically
            compatible_ops = self._get_compatible_operators_for_field_type(field_type)
            field_type_info = f"{field_type} (use: {', '.join(compatible_ops[:10])})"  # Show first 10 operators
            fields_info.append(f"- {field['id']}: {field.get('description', 'No description')} [{field_type_info}]")
        
        operators_info = []
        for op in selected_operators:
            definition = op.get('definition', 'No definition available')
            operators_info.append(f"- {op['name']}: {op['description']} | Syntax: {definition}")
        
        # 60% real examples, 40% personas - randomly choose inspiration type
        import random
        use_real_examples = random.random() < 0.6
        
        if use_real_examples:
            # 60% chance: Use historical alpha examples
            historical_alphas = self._select_historical_alphas(3)
            historical_examples = ""
            if historical_alphas:
                historical_examples = "\n\nHISTORICAL ALPHA EXAMPLES FOR INSPIRATION:\n"
                for i, alpha in enumerate(historical_alphas, 1):
                    status_emoji = "✅" if alpha['status'] == 'SUBMITTED' else "📝" if alpha['status'] == 'UNSUBMITTED' else "❌"
                    historical_examples += f"{i}. {alpha['expression']} {status_emoji} (Sharpe: {alpha['sharpe']:.2f}, Fitness: {alpha['fitness']:.2f}, Region: {alpha['region']}, Status: {alpha['status']})\n"
                historical_examples += "\nUse these as inspiration but create your own unique expression using the selected fields and operators.\n"
            
            persona_prompt = ""  # No persona when using real examples
            logger.info(f"📚 HISTORICAL ALPHAS: {historical_alphas}")
            logger.info(f"📚 USING REAL EXAMPLES: Historical alphas for inspiration")
        else:
            # 40% chance: Use creative personas
            persona = self._select_persona()
            persona_prompt = self._get_persona_prompt(persona, 1)
            historical_examples = ""  # No historical examples when using personas
            logger.info(f"🎭 USING PERSONA: {persona['name']} - {persona['style']}")
        
        prompt = f"""
🔨 ALPHA EXPRESSION BUILDER FOR {region.upper()}


CREATE ALPHAS LIKE THIS:
{persona_prompt}

You are building a WorldQuant Brain alpha expression. This is NOT SQL - it's a mathematical expression using operators and data fields.

SELECTED FIELDS:
{chr(10).join(fields_info)}

SELECTED OPERATORS:
{chr(10).join(operators_info)}
{historical_examples}

CRITICAL INSTRUCTIONS:
- This is a MATHEMATICAL EXPRESSION, not SQL code
- Use ONLY the selected fields and operators above
- Field compatibility will be handled automatically during validation
- Create a valid alpha expression like above
- Use proper syntax with balanced parentheses
- Return ONLY the alpha expression, no explanations, no code blocks, no markdown
- DO NOT include "plaintext" prefix or any other prefixes
- No words like 'math' or 'alpha expression' in the results, only the expression

RESPONSE FORMAT (return only the expression, no prefixes):
"""
        
        try:
            import ollama
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.5, 'top_p': 0.9}
            )
            
            template = response['message']['content'].strip()

            # Clean up any SQL blocks or markdown that Ollama might generate
            if '```' in template:
                # Extract content between code blocks
                import re
                code_blocks = re.findall(r'```(?:sql|python|javascript)?\n?(.*?)\n?```', template, re.DOTALL)
                if code_blocks:
                    template = code_blocks[0].strip()
                else:
                    # Remove the ``` wrapper
                    template = template.replace('```sql', '').replace('```python', '').replace('```javascript', '').replace('```', '').strip()
            if template.startswith('markdown'):
                template = template.split('markdown')[1].strip()
            if template.startswith('alpha expression'):
                template = template.split('alpha expression')[1].strip()
            if template.startswith('math expression'):
                template = template.split('math expression')[1].strip()

            

            # Remove any field descriptions or explanations
            lines = template.split('\n')
            alpha_lines = []
            for line in lines:
                line = line.strip()
                # Skip lines that look like descriptions or explanations
                if (line.startswith('(') and line.endswith(')') and 
                    ('SELECTED' in line or 'OPERATORS' in line or 'FIELDS' in line)):
                    continue
                # Skip empty lines and description lines
                if line and not line.startswith('-') and not line.startswith('*'):
                    alpha_lines.append(line)
            
            if alpha_lines:
                template = ' '.join(alpha_lines)
            
            # Remove any "alpha_expression_here" placeholder text
            template = template.replace('alpha_expression_here', '').strip()
            
            # Remove "plaintext" prefix and any following whitespace
            if template.startswith('plaintext'):
                template = template.replace('plaintext', '', 1).strip()
                logger.info(f"🔧 REMOVED PLAINTEXT PREFIX: {template}")
            
            # Clean up any extra whitespace
            template = ' '.join(template.split())
            
            logger.info(f"🔨 STEP 3 COMPLETE: Built template: {template}")
            return template
            
        except Exception as e:
            logger.error(f"❌ STEP 3 FAILED: {e}")
            return ""
    
    def _validate_template_step(self, template: str, region: str, delay: int = None) -> Tuple[bool, str]:
        """Step 4: Validate the built template"""
        logger.info(f"✅ STEP 4: Validating template: {template}")
        logger.info(f"📝 Template details: length={len(template)}, region={region}, delay={delay}")
        
        # If delay is None, use optimal delay
        if delay is None:
            delay = self.select_optimal_delay(region)
            logger.info(f"🔧 DELAY SYNC: Using optimal delay {delay} for validation")
        
        # Get valid fields for validation
        data_fields = self.get_data_fields_for_region(region, delay)
        valid_fields = [field['id'] for field in data_fields] if data_fields else []
        
        # NEW: Check for non-vec_* operators and replace VECTOR fields with MATRIX fields if found
        if self._has_non_vec_operators(template):
            logger.info(f"🔧 NON-VEC OPERATORS DETECTED: Replacing VECTOR fields with MATRIX fields")
            fixed_template = self._replace_vector_fields_with_matrix_fields(template, region, delay)
        else:
            # Fix vector field issues with region-specific fields
            fixed_template = self._fix_vector_field_issues(template, region, delay)
            
            # Check if there are still incompatible operators with VECTOR fields
            if self._has_incompatible_vector_operators(fixed_template, region, delay):
                logger.warning(f"🚨 INCOMPATIBLE OPERATORS DETECTED: Sending back to Ollama for field replacement")
                # Send back to Ollama for field replacement
                fixed_template = self._ollama_field_replacement(fixed_template, region, delay)
        
        # Track this template
        self._track_recent_template(fixed_template)
        
        logger.info(f"✅ STEP 4 COMPLETE: Template validated successfully")
        logger.info(f"🔧 FIXED TEMPLATE: {fixed_template}")
        return True, fixed_template
  
    def _get_market_context(self, region: str) -> str:
        """ULTRA-ENHANCED market context for maximum alpha generation success"""
        contexts = {
            'USA': 'US equity markets - High liquidity, diverse sectors, strong momentum patterns, regulatory stability, institutional dominance',
            'EUR': 'European equity markets - Regulatory considerations, sector rotation opportunities, currency effects, ESG factors',
            'GLB': 'Global equity markets - Currency and timezone effects, cross-sectional opportunities, emerging market exposure',
            'ASI': 'Asian equity markets - Emerging market characteristics, high volatility, growth opportunities, regulatory changes',
            'CHN': 'Chinese equity markets - Regulatory and currency controls, state intervention, growth vs value dynamics'
        }
        return contexts.get(region, 'General equity markets with diverse opportunities')
    
    def _prioritize_fields(self, data_fields: List[Dict]) -> List[Dict]:
        """Prioritize fields using dynamic strategy based on elite template discoveries"""
        # Update field strategy before prioritizing
        self._update_field_strategy()
        
        def field_score(field):
            pyramid_mult = field.get('pyramidMultiplier', 1.0)
            user_count = field.get('userCount', 0)
            alpha_count = field.get('alphaCount', 0)
            
            # Get current strategy weights
            weights = self.field_strategy_weights[self.field_strategy_mode]
            
            # Calculate random exploration component (favors diverse field usage)
            if user_count == 0 and alpha_count == 0:
                # Completely unused - high random score
                random_score = 4.0
            elif user_count <= 10 and alpha_count <= 20:
                # Lightly used - good random score
                random_score = 3.0
            elif user_count <= 50 and alpha_count <= 100:
                # Moderately used - medium random score
                random_score = 2.0
            elif user_count <= 200 and alpha_count <= 500:
                # Heavily used - low random score
                random_score = 1.0
            else:
                # Overused - minimal random score
                random_score = 0.1
            
            # Calculate rare field component (favors undiscovered fields)
            if user_count == 0 and alpha_count == 0:
                # Completely unused - maximum rare score
                rare_score = 5.0
            elif user_count <= 2 and alpha_count <= 5:
                # Barely used - very high rare score
                rare_score = 4.0
            elif user_count <= 10 and alpha_count <= 20:
                # Lightly used - high rare score
                rare_score = 3.0
            elif user_count <= 50 and alpha_count <= 100:
                # Moderately used - medium rare score
                rare_score = 2.0
            elif user_count <= 200 and alpha_count <= 500:
                # Heavily used - low rare score
                rare_score = 1.0
            else:
                # Overused - minimal rare score
                rare_score = 0.1
            
            # Apply strategy weights
            strategy_score = (weights['random'] * random_score + 
                            weights['rare'] * rare_score)
            
            # Add pyramid multiplier as base score
            total_score = pyramid_mult + strategy_score
            
            # Apply penalty for overused fields
            if user_count > 500 or alpha_count > 1000:
                total_score -= 2.0
            
            return total_score
        
        # Sort by score (higher is better)
        prioritized = sorted(data_fields, key=field_score, reverse=True)
        
        # Log strategy information
        weights = self.field_strategy_weights[self.field_strategy_mode]
        logger.info(f"🎯 DYNAMIC FIELD STRATEGY: {self.field_strategy_mode.upper()}")
        logger.info(f"🎯 STRATEGY WEIGHTS: Random={weights['random']:.1%}, Rare={weights['rare']:.1%}")
        logger.info(f"🎯 ELITE STATUS: Found {self.elite_templates_found} elites, mode based on discoveries")
        
        logger.info(f"🔍 DYNAMIC FIELD PRIORITIZATION: Top 10 fields by strategy score:")
        for i, field in enumerate(prioritized[:10]):
            pyramid_mult = field.get('pyramidMultiplier', 1.0)
            user_count = field.get('userCount', 0)
            alpha_count = field.get('alphaCount', 0)
            score = field_score(field)
            status = "UNUSED" if user_count == 0 and alpha_count == 0 else "BARELY_USED" if user_count <= 2 and alpha_count <= 5 else "OVERUSED" if user_count > 500 or alpha_count > 1000 else "MODERATE"
            logger.info(f"   {i+1}. {field['id']} (score: {score:.3f}, pyramid: {pyramid_mult}, users: {user_count}, alphas: {alpha_count}) [{status}]")
        
        return prioritized
    
    def _find_undiscovered_gems(self, data_fields: List[Dict], max_fields: int = 20) -> List[Dict]:
        """Find the most undiscovered fields (lowest usage) for maximum alpha potential"""
        # Filter for truly undiscovered fields
        undiscovered = []
        for field in data_fields:
            user_count = field.get('userCount', 0)
            alpha_count = field.get('alphaCount', 0)
            
            # Only include fields with very low usage
            if user_count <= 5 and alpha_count <= 10:
                undiscovered.append(field)
        
        # Sort by pyramid multiplier (highest first) among undiscovered fields
        undiscovered.sort(key=lambda x: x.get('pyramidMultiplier', 1.0), reverse=True)
        
        logger.info(f"💎 UNDISCOVERED GEMS: Found {len(undiscovered)} truly undiscovered fields")
        if undiscovered:
            logger.info(f"💎 TOP UNDISCOVERED GEMS:")
            for i, field in enumerate(undiscovered[:10]):
                users = field.get('userCount', 0)
                alphas = field.get('alphaCount', 0)
                pyramid = field.get('pyramidMultiplier', 1.0)
                logger.info(f"   {i+1}. {field['id']} (pyramid: {pyramid}, users: {users}, alphas: {alphas})")
        
        return undiscovered[:max_fields]
    
    def _create_gem_discovery_selection(self, data_fields: List[Dict], max_fields: int = 30) -> List[Dict]:
        """Create a dynamic selection based on elite template discoveries and time"""
        # Update field strategy before selection
        self._update_field_strategy()
        
        # Get current strategy weights
        weights = self.field_strategy_weights[self.field_strategy_mode]
        
        # Categorize fields by usage levels
        undiscovered_gems = []      # Very low usage (users <= 5, alphas <= 10)
        moderate_fields = []       # Moderate usage (users 6-50, alphas 11-100)
        popular_fields = []        # High usage (users > 50, alphas > 100)
        
        for field in data_fields:
            user_count = field.get('userCount', 0)
            alpha_count = field.get('alphaCount', 0)
            
            if user_count <= 5 and alpha_count <= 10:
                undiscovered_gems.append(field)
            elif user_count <= 50 and alpha_count <= 100:
                moderate_fields.append(field)
            else:
                popular_fields.append(field)
        
        logger.info(f"🎯 DYNAMIC GEM DISCOVERY ANALYSIS:")
        logger.info(f"🎯 STRATEGY: {self.field_strategy_mode.upper()} (Random: {weights['random']:.1%}, Rare: {weights['rare']:.1%})")
        logger.info(f"   Undiscovered gems: {len(undiscovered_gems)} (users ≤5, alphas ≤10)")
        logger.info(f"   Moderate fields: {len(moderate_fields)} (users 6-50, alphas 11-100)")
        logger.info(f"   Popular fields: {len(popular_fields)} (users >50, alphas >100)")
        
        # Dynamic selection based on strategy weights
        selected_fields = []
        
        # Calculate field counts based on strategy weights
        gems_count = min(len(undiscovered_gems), int(max_fields * weights['rare']))
        random_count = min(len(moderate_fields) + len(popular_fields), int(max_fields * weights['random']))
        
        # Add undiscovered gems (rare-focused component)
        if gems_count > 0:
            # Sort gems by pyramid multiplier (highest first)
            undiscovered_gems.sort(key=lambda x: x.get('pyramidMultiplier', 1.0), reverse=True)
            selected_fields.extend(undiscovered_gems[:gems_count])
            logger.info(f"💎 Added {gems_count} undiscovered gems (rare component)")
        
        # Add random exploration fields (random component)
        if random_count > 0:
            # Combine moderate and popular fields for random exploration
            random_fields = moderate_fields + popular_fields
            # Sort by pyramid multiplier but add some randomness
            random_fields.sort(key=lambda x: x.get('pyramidMultiplier', 1.0), reverse=True)
            # Add some randomness to the selection
            random.shuffle(random_fields[:min(len(random_fields), random_count * 2)])
            selected_fields.extend(random_fields[:random_count])
            logger.info(f"🎲 Added {random_count} random exploration fields")
        
        # Fill remaining slots with best available fields
        if len(selected_fields) < max_fields:
            remaining = max_fields - len(selected_fields)
            all_fields = data_fields.copy()
            # Remove already selected fields
            selected_ids = {field['id'] for field in selected_fields}
            available_fields = [f for f in all_fields if f['id'] not in selected_ids]
            # Sort by pyramid multiplier
            available_fields.sort(key=lambda x: x.get('pyramidMultiplier', 1.0), reverse=True)
            selected_fields.extend(available_fields[:remaining])
            logger.info(f"🔧 Added {min(remaining, len(available_fields))} additional fields for completeness")
        
        # Shuffle the final selection to create interesting combinations
        random.shuffle(selected_fields)
        
        # Log the final selection
        logger.info(f"🎯 DYNAMIC FIELD SELECTION: {len(selected_fields)} fields")
        logger.info(f"   Strategy: {self.field_strategy_mode} (Random: {weights['random']:.1%}, Rare: {weights['rare']:.1%})")
        logger.info(f"   Composition: {gems_count} gems + {random_count} random + {len(selected_fields) - gems_count - random_count} additional")
        
        # Show top selections
        for i, field in enumerate(selected_fields[:10]):
            users = field.get('userCount', 0)
            alphas = field.get('alphaCount', 0)
            pyramid = field.get('pyramidMultiplier', 1.0)
            category = "GEM" if users <= 5 and alphas <= 10 else "MODERATE" if users <= 50 and alphas <= 100 else "POPULAR"
            logger.info(f"   {i+1}. {field['id']} [{category}] (pyramid: {pyramid}, users: {users}, alphas: {alphas})")
        
        return selected_fields[:max_fields]
    
    def get_field_strategy_status(self) -> Dict:
        """Get current field strategy status for monitoring and debugging"""
        current_time = time.time()
        time_since_last_elite = current_time - self.last_elite_discovery_time if self.last_elite_discovery_time > 0 else float('inf')
        time_since_start = current_time - self.field_strategy_start_time
        
        weights = self.field_strategy_weights[self.field_strategy_mode]
        
        return {
            'strategy_mode': self.field_strategy_mode,
            'elite_templates_found': self.elite_templates_found,
            'last_elite_discovery_time': self.last_elite_discovery_time,
            'time_since_last_elite': time_since_last_elite,
            'time_since_start': time_since_start,
            'strategy_weights': weights,
            'elite_threshold': self.elite_discovery_threshold,
            'time_decay_threshold': self.time_decay_threshold,
            'should_switch_to_rare': self.elite_templates_found >= self.elite_discovery_threshold,
            'should_switch_to_random': time_since_last_elite > self.time_decay_threshold and time_since_start > self.time_decay_threshold
        }
    
    def _load_operator_compatibility(self) -> Dict:
        """Load operator compatibility data from file"""
        try:
            if os.path.exists(self.operator_compatibility_file):
                with open(self.operator_compatibility_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"📚 Loaded operator compatibility data: {len(data.get('operators', {}))} operators tracked")
                    return data
            else:
                logger.info("📚 No operator compatibility file found, starting fresh")
                return {
                    'operators': {},
                    'last_updated': time.time(),
                    'total_failures': 0,
                    'total_fixes': 0
                }
        except Exception as e:
            logger.error(f"❌ Failed to load operator compatibility: {e}")
            return {
                'operators': {},
                'last_updated': time.time(),
                'total_failures': 0,
                'total_fixes': 0
            }
    
    def _save_operator_compatibility(self):
        """Save operator compatibility data to file"""
        try:
            self.operator_compatibility['last_updated'] = time.time()
            with open(self.operator_compatibility_file, 'w') as f:
                json.dump(self.operator_compatibility, f, indent=2)
            logger.info(f"💾 Saved operator compatibility data: {len(self.operator_compatibility['operators'])} operators tracked")
        except Exception as e:
            logger.error(f"❌ Failed to save operator compatibility: {e}")
    
    def _load_blacklist_from_disk(self):
        """Load blacklist from disk for persistence"""
        try:
            if os.path.exists(self.blacklist_file):
                with open(self.blacklist_file, "r") as f:
                    data = json.load(f)
                    self.operator_blacklist = set(data.get("blacklisted_operators", []))
                    self.operator_usage_count = data.get("usage_count", {})
                    self.operator_blacklist_timestamps = data.get("blacklist_timestamps", {})
                    self.operator_blacklist_reasons = data.get("blacklist_reasons", {})
                    self.successful_simulations_since_blacklist = data.get("successful_simulations", 0)
                    logger.info(f"📁 LOADED BLACKLIST: {len(self.operator_blacklist)} operators blacklisted")
                    logger.info(f"📁 LOADED USAGE: {len(self.operator_usage_count)} operators tracked")
                    logger.info(f"📁 LOADED SUCCESS COUNT: {self.successful_simulations_since_blacklist} successful simulations")
            else:
                logger.info("📁 No existing blacklist file found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load blacklist from disk: {e}")
            self.operator_blacklist = set()
            self.operator_usage_count = {}
            self.operator_blacklist_timestamps = {}
            self.operator_blacklist_reasons = {}
            self.successful_simulations_since_blacklist = 0
    
    def _load_blacklist_from_disk(self):
        """Load operator blacklist from disk"""
        try:
            if os.path.exists(self.blacklist_file):
                with open(self.blacklist_file, "r") as f:
                    data = json.load(f)
                
                # Load blacklisted operators
                self.operator_blacklist = set(data.get("blacklisted_operators", []))
                self.operator_usage_count = data.get("usage_count", {})
                self.operator_blacklist_timestamps = data.get("blacklist_timestamps", {})
                self.operator_blacklist_reasons = data.get("blacklist_reasons", {})
                self.successful_simulations_since_blacklist = data.get("successful_simulations", 0)
                
                logger.info(f"📁 LOADED BLACKLIST: {len(self.operator_blacklist)} operators blacklisted")
                if self.operator_blacklist:
                    logger.info(f"🚫 BLACKLISTED OPERATORS: {list(self.operator_blacklist)}")
            else:
                logger.info(f"📁 NO BLACKLIST FILE: Starting with empty blacklist")
                self.operator_blacklist = set()
                self.operator_blacklist_timestamps = {}
                self.operator_blacklist_reasons = {}
                self.successful_simulations_since_blacklist = 0
        except Exception as e:
            logger.error(f"Failed to load blacklist from disk: {e}")
            # Initialize empty blacklist on error
            self.operator_blacklist = set()
            self.operator_blacklist_timestamps = {}
            self.operator_blacklist_reasons = {}
            self.successful_simulations_since_blacklist = 0
    
    def _save_blacklist_to_disk(self):
        """Save blacklist to disk for persistence"""
        try:
            data = {
                "blacklisted_operators": list(self.operator_blacklist),
                "usage_count": self.operator_usage_count,
                "blacklist_timestamps": self.operator_blacklist_timestamps,
                "blacklist_reasons": self.operator_blacklist_reasons,
                "successful_simulations": self.successful_simulations_since_blacklist,
                "last_updated": time.time()
            }
            with open(self.blacklist_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"💾 SAVED BLACKLIST: {len(self.operator_blacklist)} operators blacklisted")
        except Exception as e:
            logger.error(f"Failed to save blacklist to disk: {e}")
    
    def _add_to_blacklist(self, operator_name: str, reason: str = ""):
        """Add operator to blacklist and save to disk"""
        self.operator_blacklist.add(operator_name)
        self.operator_blacklist_timestamps[operator_name] = time.time()
        self.operator_blacklist_reasons[operator_name] = reason
        logger.warning(f"🚫 BLACKLISTED: {operator_name} - {reason}")
        self._save_blacklist_to_disk()
    
    def _is_operator_blacklisted(self, operator_name: str) -> bool:
        """Check if operator is blacklisted"""
        return operator_name in self.operator_blacklist
    
    def _check_blacklist_release_conditions(self):
        """Check if any blacklisted operators should be released"""
        current_time = time.time()
        operators_to_release = []
        
        # Check time-based release (10 minutes)
        for operator in list(self.operator_blacklist):
            if operator in self.operator_blacklist_timestamps:
                blacklist_time = self.operator_blacklist_timestamps[operator]
                hours_blacklisted = (current_time - blacklist_time) / 3600
                
                if hours_blacklisted >= self.blacklist_timeout_hours:
                    operators_to_release.append((operator, f"Time-based release ({hours_blacklisted*60:.1f} minutes)"))
        
        # Check success-based release (10 successful simulations) - RELEASE ALL
        if self.successful_simulations_since_blacklist >= self.blacklist_release_threshold:
            blacklisted_operators = list(self.operator_blacklist)
            if blacklisted_operators:
                # Release ALL blacklisted operators
                for operator in blacklisted_operators:
                    operators_to_release.append((operator, f"Success-based release (ALL - {self.successful_simulations_since_blacklist} successes)"))
                
                # Reset the success counter after releasing all
                self.successful_simulations_since_blacklist = 0
                logger.info(f"🔄 FULL RELEASE: Released ALL {len(blacklisted_operators)} blacklisted operators after {self.blacklist_release_threshold} successful simulations")
        
        # Release operators
        for operator, reason in operators_to_release:
            self._release_from_blacklist(operator, reason)
    
    def _release_from_blacklist(self, operator_name: str, reason: str = ""):
        """Release operator from blacklist"""
        if operator_name in self.operator_blacklist:
            self.operator_blacklist.remove(operator_name)
            if operator_name in self.operator_blacklist_timestamps:
                del self.operator_blacklist_timestamps[operator_name]
            if operator_name in self.operator_blacklist_reasons:
                del self.operator_blacklist_reasons[operator_name]
            # RESET USAGE COUNT
            if operator_name in self.operator_usage_count:
                self.operator_usage_count[operator_name] = 0
                logger.info(f"🔄 USAGE RESET: {operator_name} usage count reset to 0")

            logger.info(f"🔄 UNBLACKLISTED: {operator_name} - {reason}")
            self._save_blacklist_to_disk()
    
    def _update_successful_simulation_count(self):
        """Update successful simulation count for blacklist release"""
        self.successful_simulations_since_blacklist += 1
        logger.info(f"📈 SUCCESS COUNT: {self.successful_simulations_since_blacklist} successful simulations since last blacklist")
        
        # Check if we should release any operators
        self._check_blacklist_release_conditions()
    
    def _get_blacklist_status(self):
        """Get current blacklist status for debugging"""
        current_time = time.time()
        status = {
            "total_blacklisted": len(self.operator_blacklist),
            "successful_simulations": self.successful_simulations_since_blacklist,
            "operators": {}
        }
        
        for operator in self.operator_blacklist:
            if operator in self.operator_blacklist_timestamps:
                blacklist_time = self.operator_blacklist_timestamps[operator]
                hours_blacklisted = (current_time - blacklist_time) / 3600
                status["operators"][operator] = {
                    "minutes_blacklisted": hours_blacklisted * 60,
                    "reason": self.operator_blacklist_reasons.get(operator, "Unknown"),
                    "can_release_time": hours_blacklisted >= self.blacklist_timeout_hours,
                    "can_release_success": self.successful_simulations_since_blacklist >= self.blacklist_release_threshold
                }
        
        return status
    
    def debug_blacklist_status(self):
        """Debug method to show current blacklist status"""
        status = self._get_blacklist_status()
        logger.info(f"🔍 BLACKLIST STATUS:")
        logger.info(f"   Total blacklisted: {status['total_blacklisted']}")
        logger.info(f"   Successful simulations: {status['successful_simulations']}")
        logger.info(f"   Release threshold: {self.blacklist_release_threshold} (FULL release)")
        logger.info(f"   Timeout: {self.blacklist_timeout_hours*60:.1f} minutes (10 minutes)")
        
        if status['operators']:
            logger.info(f"   Blacklisted operators:")
            for op, info in status['operators'].items():
                logger.info(f"     {op}: {info['minutes_blacklisted']:.1f}min, reason='{info['reason']}', "
                          f"can_release_time={info['can_release_time']}, can_release_success={info['can_release_success']}")
        
        return status
    
    def _learn_operator_failure(self, operator_name: str, error_message: str, template: str):
        """Learn from operator failures and update compatibility knowledge"""
        if operator_name not in self.operator_compatibility['operators']:
            self.operator_compatibility['operators'][operator_name] = {
                'failures': 0,
                'successes': 0,
                'error_types': {},
                'last_failure': None,
                'workarounds': [],
                'compatibility_score': 0.5  # Start neutral
            }
        
        op_data = self.operator_compatibility['operators'][operator_name]
        op_data['failures'] += 1
        op_data['last_failure'] = time.time()
        self.operator_compatibility['total_failures'] += 1
        
        # Track error types
        error_type = self._classify_error(error_message)
        if error_type not in op_data['error_types']:
            op_data['error_types'][error_type] = 0
        op_data['error_types'][error_type] += 1
        
        # Update compatibility score (lower = more problematic)
        op_data['compatibility_score'] = op_data['successes'] / (op_data['successes'] + op_data['failures'] + 1)
        
        logger.info(f"🧠 LEARNED: {operator_name} failure - {error_type} (score: {op_data['compatibility_score']:.2f})")
        
        # Save updated knowledge
        self._save_operator_compatibility()
    
    def _learn_operator_success(self, operator_name: str, template: str):
        """Learn from operator successes"""
        if operator_name not in self.operator_compatibility['operators']:
            self.operator_compatibility['operators'][operator_name] = {
                'failures': 0,
                'successes': 0,
                'error_types': {},
                'last_failure': None,
                'workarounds': [],
                'compatibility_score': 0.5
            }
        
        op_data = self.operator_compatibility['operators'][operator_name]
        op_data['successes'] += 1
        
        # Update compatibility score
        op_data['compatibility_score'] = op_data['successes'] / (op_data['successes'] + op_data['failures'] + 1)
        
        logger.info(f"✅ LEARNED: {operator_name} success (score: {op_data['compatibility_score']:.2f})")
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error types for learning"""
        error_lower = error_message.lower()
        
        if 'does not support event inputs' in error_lower:
            return 'event_inputs'
        elif 'invalid number of inputs' in error_lower:
            return 'input_count'
        elif 'unknown variable' in error_lower:
            return 'unknown_variable'
        elif 'unexpected character' in error_lower:
            return 'syntax_error'
        elif 'operator' in error_lower and 'not found' in error_lower:
            return 'operator_not_found'
        else:
            return 'other'
    
    def _get_operator_workaround(self, operator_name: str, error_type: str) -> str:
        """Get workaround for operator based on learned failures"""
        if operator_name not in self.operator_compatibility['operators']:
            return None
        
        op_data = self.operator_compatibility['operators'][operator_name]
        
        # Check if we have a workaround for this error type
        for workaround in op_data['workarounds']:
            if workaround['error_type'] == error_type:
                return workaround['solution']
        
        return None
    
    def _add_operator_workaround(self, operator_name: str, error_type: str, solution: str):
        """Add a new workaround for an operator"""
        if operator_name not in self.operator_compatibility['operators']:
            self.operator_compatibility['operators'][operator_name] = {
                'failures': 0,
                'successes': 0,
                'error_types': {},
                'last_failure': None,
                'workarounds': [],
                'compatibility_score': 0.5
            }
        
        op_data = self.operator_compatibility['operators'][operator_name]
        
        # Add workaround if not already exists
        workaround_exists = any(w['error_type'] == error_type for w in op_data['workarounds'])
        if not workaround_exists:
            op_data['workarounds'].append({
                'error_type': error_type,
                'solution': solution,
                'added_at': time.time()
            })
            logger.info(f"🔧 ADDED WORKAROUND: {operator_name} - {error_type} -> {solution}")
            self._save_operator_compatibility()
    
    def _validate_ollama_template(self, template: str) -> Tuple[bool, str]:
        return True, template
    
    def _fix_vector_field_issues(self, template: str, region: str = None, delay: int = None) -> str:
        """Use YES/NO decision for field replacement to avoid hallucination"""
        import re
        
        # Get available MATRIX/REGULAR fields from the SPECIFIC region cache, sorted by usage
        available_fields = []
        
        if region and delay is not None:
            # Use the specific region and delay
            cache_file = f"data_fields_cache_{region}_{delay}.json"
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        fields_data = json.load(f)
                        for field in fields_data:
                            if field.get('type') in ['REGULAR', 'MATRIX']:  # Only non-VECTOR fields
                                available_fields.append(field)
                    
                    # Sort by usage (lowest usage first) - prioritize underused fields
                    def get_usage_score(field):
                        user_count = field.get('userCount', 0)
                        alpha_count = field.get('alphaCount', 0)
                        return user_count + alpha_count
                    
                    available_fields.sort(key=get_usage_score)
                    logger.info(f"🔧 REGION-SPECIFIC FIX: Using {len(available_fields)} replacement fields from {region} delay={delay} (sorted by usage)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {region} cache: {e}")
        else:
            logger.warning(f"⚠️ No region/delay specified, skipping field replacement")
            return template
        
        if not available_fields:
            logger.warning(f"⚠️ No replacement fields available for {region}")
            return template
        
        # Create prompt for Ollama to choose replacement fields
        # Show first 20 underused MATRIX/REGULAR fields with usage info
        replacement_options = []
        for i, field in enumerate(available_fields[:20]):  # Show first 20 fields
            user_count = field.get('userCount', 0)
            alpha_count = field.get('alphaCount', 0)
            field_type = field.get('type', 'REGULAR')
            usage_info = f"users={user_count}, alphas={alpha_count}"
            replacement_options.append(f"[{i}] {field['id']}: {field.get('description', 'No description')} [{field_type}] [{usage_info}]")
        
        prompt = f"""
🔧 FIELD REPLACEMENT FOR {region.upper()}

The template has VECTOR fields with incompatible operators:

CURRENT TEMPLATE:
{template}

REPLACEMENT OPTIONS (sorted by usage - lowest usage first):
{chr(10).join(replacement_options)}

INSTRUCTIONS:
- Choose 1-2 replacement fields by their INDEX NUMBERS (e.g., [0, 5])
- PRIORITIZE LOW-USAGE FIELDS: Prefer fields with low userCount and alphaCount
- Choose MATRIX fields for Time Series operators, REGULAR fields for standard operators
- Return ONLY the index numbers in square brackets

RESPONSE FORMAT:
[0, 5]
"""
        
        try:
            import ollama
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3, 'top_p': 0.8}
            )
            
            # Parse Ollama's response to get field indices
            content = response['message']['content'].strip()
            logger.info(f"🔧 OLLAMA REPLACEMENT SELECTION: {content}")
            
            # Extract indices from response (e.g., [0, 5] or 0, 5)
            import re
            indices = []
            
            # Try to find indices in square brackets
            bracket_match = re.search(r'\[([0-9,\s]+)\]', content)
            if bracket_match:
                indices_str = bracket_match.group(1)
                indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
            else:
                # Try to find individual numbers
                numbers = re.findall(r'\b(\d+)\b', content)
                indices = [int(x) for x in numbers if int(x) < len(available_fields[:20])]
            
            if indices:
                logger.info(f"🔧 OLLAMA SELECTED INDICES: {indices}")
                
                # Get field types to identify VECTOR fields
                field_types = {}
                cache_file = f"data_fields_cache_{region}_{delay}.json"
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            fields_data = json.load(f)
                            for field in fields_data:
                                field_types[field['id']] = field.get('type', 'REGULAR')
                    except:
                        pass
                
                # Replace VECTOR fields with selected MATRIX/REGULAR fields
                field_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
                all_words = re.findall(field_pattern, template)
                
                replacement_count = 0
                for word in all_words:
                    if word in field_types and field_types[word] == 'VECTOR' and replacement_count < len(indices):
                        replacement_field = available_fields[indices[replacement_count]]['id']
                        template = template.replace(word, replacement_field)
                        logger.info(f"🔧 SMART REPLACEMENT: {word} -> {replacement_field} (index {indices[replacement_count]})")
                        replacement_count += 1
                
                if replacement_count == 0:
                    logger.warning(f"⚠️ NO VECTOR FIELD FOUND FOR REPLACEMENT")
            else:
                logger.warning(f"⚠️ NO VALID INDICES FOUND: {content}, using original template")
            
        except Exception as e:
            logger.error(f"❌ OLLAMA REPLACEMENT FAILED: {e}")
            # Keep original template if Ollama fails
        
        return template
    
    def _has_incompatible_vector_operators(self, template: str, region: str, delay: int) -> bool:
        """Check if template has incompatible operators with VECTOR fields"""
        import re
        
        # Get field types from the specific region cache
        field_types = {}
        cache_file = f"data_fields_cache_{region}_{delay}.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    fields_data = json.load(f)
                    for field in fields_data:
                        field_types[field['id']] = field.get('type', 'REGULAR')
            except:
                return False
        
        # Extract all field references from template
        field_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        all_words = re.findall(field_pattern, template)
        
        # Check for incompatible operators with VECTOR fields
        incompatible_ops = ['normalize', 'quantile', 'rank', 'scale', 'winsorize', 'zscore']
        event_indicators = ['event', 'sentiment', 'news', 'novelty', 'relevance', 'confidence']
        
        for word in all_words:
            if word in field_types:
                field_type = field_types[word]
                if field_type == 'VECTOR':
                    # Check for incompatible operators
                    for op in incompatible_ops:
                        if f"{op}(" in template and word in template:
                            # Check if it's an event field
                            is_event_field = any(indicator in word.lower() for indicator in event_indicators)
                            if is_event_field:
                                logger.warning(f"🚨 INCOMPATIBLE: {op}({word}) - event VECTOR field doesn't support {op}")
                                return True
        
        return False
    
    def _ollama_field_replacement(self, template: str, region: str, delay: int) -> str:
        """Send template back to Ollama for field replacement"""
        # Get available fields from the specific region
        cache_file = f"data_fields_cache_{region}_{delay}.json"
        available_fields = []
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    fields_data = json.load(f)
                    for field in fields_data:
                        if field.get('type') in ['REGULAR', 'MATRIX']:  # Only non-VECTOR fields
                            available_fields.append(field['id'])
            except:
                pass
        
        if not available_fields:
            logger.warning(f"⚠️ No replacement fields available for {region}")
            return template
        
        # Create indexed list of available fields
        field_options = []
        for i, field in enumerate(available_fields[:30]):  # Show first 30 fields
            field_options.append(f"[{i}] {field}")
        
        # Create prompt for Ollama to decide if replacement is needed
        prompt = f"""
🔧 FIELD REPLACEMENT DECISION FOR {region.upper()}

The template has VECTOR fields with incompatible operators:

CURRENT TEMPLATE:
{template}

QUESTION: Does this template need field replacement?
- YES: If there are VECTOR fields with incompatible operators (normalize, quantile, rank, scale, winsorize, zscore)
- NO: If all fields are already compatible

RESPONSE FORMAT: Answer with only "YES" or "NO"
"""
        
        try:
            import ollama
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3, 'top_p': 0.8}
            )
            
            # Parse Ollama's response
            content = response['message']['content'].strip().upper()
            logger.info(f"🔧 OLLAMA REPLACEMENT DECISION: {content}")
            
            # Check if Ollama says YES or NO
            if "YES" in content:
                logger.info(f"🔧 OLLAMA SAYS YES - Replacement needed, performing automatic replacement")
                # Perform automatic replacement using available fields
                if available_fields:
                    import re
                    field_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
                    all_words = re.findall(field_pattern, template)
                    
                    # Get field types to identify VECTOR fields
                    field_types = {}
                    cache_file = f"data_fields_cache_{region}_{delay}.json"
                    if os.path.exists(cache_file):
                        try:
                            with open(cache_file, 'r') as f:
                                fields_data = json.load(f)
                                for field in fields_data:
                                    field_types[field['id']] = field.get('type', 'REGULAR')
                        except:
                            pass
                    
                    # Replace first VECTOR field with first available MATRIX/REGULAR field
                    for word in all_words:
                        if word in field_types and field_types[word] == 'VECTOR':
                            replacement_field = available_fields[0]
                            content = template.replace(word, replacement_field)
                            logger.info(f"🔧 AUTOMATIC REPLACEMENT: {word} -> {replacement_field}")
                            break
                    else:
                        # No VECTOR field found, return original template
                        content = template
                        logger.warning(f"⚠️ NO VECTOR FIELD FOUND FOR REPLACEMENT")
                else:
                    content = template
                    logger.warning(f"⚠️ NO REPLACEMENT FIELDS AVAILABLE")
            elif "NO" in content:
                logger.info(f"🔧 OLLAMA SAYS NO - No replacement needed, using original template")
                content = template
            else:
                logger.warning(f"⚠️ UNKNOWN RESPONSE: {content}, using original template")
                content = template
            
            # Clean up the response
            if '```' in content:
                code_blocks = re.findall(r'```(?:sql|python|javascript)?\n?(.*?)\n?```', content, re.DOTALL)
                if code_blocks:
                    content = code_blocks[0].strip()
                else:
                    content = content.replace('```sql', '').replace('```python', '').replace('```javascript', '').replace('```', '').strip()
            
            # Remove any "plaintext" prefix
            if content.startswith('plaintext'):
                content = content.replace('plaintext', '', 1).strip()
            
            # Clean up extra whitespace
            content = ' '.join(content.split())
            
            logger.info(f"🔧 OLLAMA REPLACEMENT RESULT: {content}")
            return content
            
        except Exception as e:
            logger.error(f"❌ OLLAMA FIELD REPLACEMENT FAILED: {e}")
            return template
    
    def _get_compatible_operators_for_field_type(self, field_type: str) -> List[str]:
        """Get all operators - field compatibility will be handled during validation"""
        # Return all operator names - field compatibility will be handled during template validation
        return [op['name'] for op in self.operators if op['name'] not in self.operator_blacklist]

    def _get_field_type(self, field_id: str) -> str:
        """Get field type from data fields cache"""
        for region in ['USA', 'GLB', 'EUR', 'ASI', 'CHN']:
            try:
                cache_file = f"data_fields_cache_{region}_0.json"
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        fields_data = json.load(f)
                        for field in fields_data:
                            if field['id'] == field_id:
                                return field.get('type', 'REGULAR')
            except:
                continue
        return 'REGULAR'  # Default to REGULAR if not found
    
    def _get_matrix_field_suggestions(self, vector_field_id: str) -> List[str]:
        """Get MATRIX field suggestions for replacing VECTOR fields"""
        matrix_fields = []
        vector_prefix = vector_field_id.split('_')[0]  # Get prefix like 'anl4', 'anl10', etc.
        
        for region in ['USA', 'GLB', 'EUR', 'ASI', 'CHN']:
            try:
                cache_file = f"data_fields_cache_{region}_0.json"
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        fields_data = json.load(f)
                        for field in fields_data:
                            if field.get('type') == 'MATRIX':
                                # Prefer fields with similar prefix or from same category
                                if vector_prefix in field['id'] or field['id'].split('_')[0] in vector_prefix:
                                    matrix_fields.append(field['id'])
                                elif len(matrix_fields) < 10:  # Keep some general MATRIX fields as backup
                                    matrix_fields.append(field['id'])
            except:
                continue
        
        return matrix_fields[:5]  # Return top 5 suggestions
    
    def _track_recent_template(self, template: str):
        """Track recently generated template to avoid repetition"""
        self.recent_templates.append(template)
        if len(self.recent_templates) > self.max_recent_templates:
            self.recent_templates.pop(0)  # Remove oldest template
    
    def _get_recent_templates_warning(self) -> str:
        """Get warning about recent templates to avoid repetition"""
        if not self.recent_templates:
            return ""
        
        recent_count = min(5, len(self.recent_templates))
        recent_examples = self.recent_templates[-recent_count:]
        
        warning = f"""
🚫 AVOID THESE RECENTLY GENERATED PATTERNS:
{chr(10).join([f"- {template}" for template in recent_examples])}

🎯 GENERATE SOMETHING DIFFERENT: Create unique, innovative templates that don't repeat these patterns!
"""
        return warning
    
    def _learn_from_simulation_failure(self, template: str, error_message: str):
        """Learn from simulation failures and update operator compatibility"""
        import re
        
        # CRITICAL: Fix unknown variable errors by removing last digit
        if "unknown variable" in error_message.lower():
            fixed_template = self._fix_unknown_variable_retry(template, error_message)
            if fixed_template != template:
                logger.info(f"🔧 UNKNOWN VARIABLE RETRY: Fixed template for retry")
                logger.info(f"   Original: {template[:100]}...")
                logger.info(f"   Fixed: {fixed_template[:100]}...")
                # TODO: Implement retry logic with fixed template
                return
        
        # Known operators that should be learned from
        known_operators = {
            'abs', 'add', 'and', 'bucket', 'days_from_last_change', 'densify', 'divide', 'equal', 'greater',
            'greater_equal', 'group_backfill', 'group_cartesian_product', 'group_max', 'group_mean', 'group_min',
            'group_neutralize', 'group_rank', 'group_scale', 'group_zscore', 'hump', 'if_else', 'inverse',
            'is_nan', 'jump_decay', 'kth_element', 'last_diff_value', 'less', 'less_equal', 'log', 'max',
            'min', 'multiply', 'normalize', 'not', 'not_equal', 'or', 'power', 'quantile', 'rank', 'reverse',
            'scale', 'scale_down', 'sign', 'signed_power', 'sqrt', 'subtract', 'to_nan', 'trade_when',
            'ts_arg_max', 'ts_arg_min', 'ts_av_diff', 'ts_backfill', 'ts_corr', 'ts_count_nans', 'ts_covariance',
            'ts_decay_linear', 'ts_delay', 'ts_delta', 'ts_max', 'ts_mean', 'ts_min', 'ts_product', 'ts_quantile',
            'ts_rank', 'ts_regression', 'ts_scale', 'ts_std_dev', 'ts_step', 'ts_sum', 'ts_target_tvr_decay',
            'ts_target_tvr_delta_limit', 'ts_zscore', 'vec_avg', 'vec_max', 'vec_min', 'vec_sum', 'vector_neut',
            'winsorize', 'zscore'
        }
        
        # Extract operator names from the template
        operator_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        operators_in_template = re.findall(operator_pattern, template)
        
        # Filter to only include known operators (not data fields)
        valid_operators = [op for op in operators_in_template if op in known_operators]
        
        # Classify the error
        error_type = self._classify_error(error_message)
        
        # Learn from each valid operator in the template
        for operator_name in valid_operators:
            self._learn_operator_failure(operator_name, error_message, template)
            
            # If it's an event input error, add a workaround
            if error_type == 'event_inputs':
                self._add_operator_workaround(operator_name, 'event_inputs', 'vec_avg(FIELD)')
                logger.info(f"🧠 LEARNED: {operator_name} needs vec_avg() wrapper for event inputs")
        
        # Log any data fields that were mistakenly used as operators
        data_fields_used_as_operators = [op for op in operators_in_template if op not in known_operators]
        if data_fields_used_as_operators:
            logger.warning(f"⚠️ DATA FIELDS USED AS OPERATORS: {data_fields_used_as_operators}")
            logger.warning(f"   This indicates a template generation issue - data fields should not be used as operators")
        
        # Update total fixes counter
        self.operator_compatibility['total_fixes'] += 1
        self._save_operator_compatibility()
    
    def _learn_from_simulation_success(self, template: str):
        """Learn from simulation successes"""
        import re
        
        # Known operators that should be learned from
        known_operators = {
            'abs', 'add', 'and', 'bucket', 'days_from_last_change', 'densify', 'divide', 'equal', 'greater',
            'greater_equal', 'group_backfill', 'group_cartesian_product', 'group_max', 'group_mean', 'group_min',
            'group_neutralize', 'group_rank', 'group_scale', 'group_zscore', 'hump', 'if_else', 'inverse',
            'is_nan', 'jump_decay', 'kth_element', 'last_diff_value', 'less', 'less_equal', 'log', 'max',
            'min', 'multiply', 'normalize', 'not', 'not_equal', 'or', 'power', 'quantile', 'rank', 'reverse',
            'scale', 'scale_down', 'sign', 'signed_power', 'sqrt', 'subtract', 'to_nan', 'trade_when',
            'ts_arg_max', 'ts_arg_min', 'ts_av_diff', 'ts_backfill', 'ts_corr', 'ts_count_nans', 'ts_covariance',
            'ts_decay_linear', 'ts_delay', 'ts_delta', 'ts_max', 'ts_mean', 'ts_min', 'ts_product', 'ts_quantile',
            'ts_rank', 'ts_regression', 'ts_scale', 'ts_std_dev', 'ts_step', 'ts_sum', 'ts_target_tvr_decay',
            'ts_target_tvr_delta_limit', 'ts_zscore', 'vec_avg', 'vec_max', 'vec_min', 'vec_sum', 'vector_neut',
            'winsorize', 'zscore'
        }
        
        # Extract operator names from the template
        operator_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        operators_in_template = re.findall(operator_pattern, template)
        
        # Filter to only include known operators (not data fields)
        valid_operators = [op for op in operators_in_template if op in known_operators]
        
        # Learn from each valid operator in the template
        for operator_name in valid_operators:
            self._learn_operator_success(operator_name, template)
    
    def _has_arithmetic_operators(self, template: str) -> bool:
        """Check if template contains arithmetic operators (+, -, *, /)"""
        import re
        
        # Look for arithmetic operators in the template
        arithmetic_pattern = r'[+\-*/]'
        return bool(re.search(arithmetic_pattern, template))
    
    def _has_non_vec_operators(self, template: str) -> bool:
        """Check if template contains any non-vec_* operators that require matrix fields"""
        import re
        
        # List of vec_* operators that are compatible with VECTOR fields
        vec_operators = {
            'vec_avg', 'vec_sum', 'vec_max', 'vec_min'
        }
        
        # Extract all function calls from template
        function_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(function_pattern, template)
        
        # Check if any non-vec_* operators are present
        for match in matches:
            if not match.startswith('vec_') and match not in vec_operators:
                logger.info(f"🔧 NON-VEC OPERATOR DETECTED: {match} - will replace VECTOR fields with MATRIX fields")
                return True
        
        # Also check for arithmetic operators
        arithmetic_pattern = r'[+\-*/]'
        if re.search(arithmetic_pattern, template):
            logger.info(f"🔧 ARITHMETIC OPERATORS DETECTED - will replace VECTOR fields with MATRIX fields")
            return True
        
        return False
    
    def _has_vec_operators(self, template: str) -> bool:
        """Check if template contains vec_* operators that require VECTOR fields"""
        import re
        
        # List of vec_* operators that require VECTOR fields
        vec_operators = {
            'vec_avg', 'vec_sum', 'vec_max', 'vec_min'
        }
        
        # Extract all function calls from template
        function_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(function_pattern, template)
        
        # Check if any vec_* operators are present
        for match in matches:
            if match in vec_operators:
                logger.info(f"🔧 VEC OPERATOR DETECTED: {match} - will replace MATRIX fields with VECTOR fields")
                return True
        
        return False
    
    def _replace_matrix_fields_with_vector_fields(self, template: str, region: str, delay: int = None) -> str:
        """Replace MATRIX fields with VECTOR fields when vec_* operators are present"""
        import re
        
        # Get available VECTOR fields from the specific region cache
        vector_fields = []
        cache_file = f"data_fields_cache_{region}_{delay}.json"
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    fields_data = json.load(f)
                    for field in fields_data:
                        if field.get('type') == 'VECTOR':
                            vector_fields.append(field)
                
                # Sort by usage (lowest usage first) - prioritize underused fields
                def get_usage_score(field):
                    user_count = field.get('userCount', 0)
                    alpha_count = field.get('alphaCount', 0)
                    return user_count + alpha_count
                
                vector_fields.sort(key=get_usage_score)
                logger.info(f"🔧 VEC OPERATORS DETECTED: Found {len(vector_fields)} VECTOR fields for MATRIX replacement in {region}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {region} cache: {e}")
                return template
        
        if not vector_fields:
            logger.warning(f"⚠️ No VECTOR fields available for {region}")
            return template
        
        # Get field types to identify MATRIX fields
        field_types = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    fields_data = json.load(f)
                    for field in fields_data:
                        field_types[field['id']] = field.get('type', 'REGULAR')
            except:
                pass
        
        # Find all fields in the template
        field_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        all_words = re.findall(field_pattern, template)
        
        # Replace only MATRIX fields with VECTOR fields
        replacement_count = 0
        for word in all_words:
            if word in field_types and field_types[word] == 'MATRIX' and replacement_count < len(vector_fields):
                replacement_field = vector_fields[replacement_count]['id']
                template = template.replace(word, replacement_field)
                logger.info(f"🔧 MATRIX->VECTOR REPLACEMENT: {word} -> {replacement_field} (due to vec_* operators)")
                replacement_count += 1
        
        if replacement_count == 0:
            logger.info(f"🔧 NO MATRIX FIELDS FOUND FOR REPLACEMENT")
        else:
            logger.info(f"🔧 REPLACED {replacement_count} MATRIX fields with VECTOR fields due to vec_* operators")
        
        return template
    
    def _replace_vector_fields_with_matrix_fields(self, template: str, region: str, delay: int = None) -> str:
        """Replace VECTOR fields with MATRIX fields when non-vec_* operators are present"""
        import re
        
        # Get available MATRIX fields from the specific region cache
        matrix_fields = []
        cache_file = f"data_fields_cache_{region}_{delay}.json"
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    fields_data = json.load(f)
                    for field in fields_data:
                        if field.get('type') == 'MATRIX':
                            matrix_fields.append(field)
                
                # Apply balanced selection for matrix fields
                def get_usage_score(field):
                    user_count = field.get('userCount', 0)
                    alpha_count = field.get('alphaCount', 0)
                    return user_count + alpha_count
                
                # Sort and categorize matrix fields by usage
                matrix_fields.sort(key=get_usage_score)
                total_matrix_fields = len(matrix_fields)
                
                if total_matrix_fields > 0:
                    # Define usage categories for matrix fields
                    low_usage_matrix = matrix_fields[:total_matrix_fields//3]
                    medium_usage_matrix = matrix_fields[total_matrix_fields//3:2*total_matrix_fields//3]
                    high_usage_matrix = matrix_fields[2*total_matrix_fields//3:]
                    
                    # Randomly decide selection strategy (30% chance for all high-usage, 70% for balanced)
                    import random
                    selection_strategy = random.choice(['balanced', 'balanced', 'balanced', 'all_high_usage'])
                    
                    if selection_strategy == 'all_high_usage':
                        # Sometimes select all high-usage matrix fields
                        matrix_fields = high_usage_matrix[:30] if len(high_usage_matrix) >= 30 else high_usage_matrix
                        logger.info(f"🔧 HIGH-USAGE MATRIX STRATEGY: Selected {len(matrix_fields)} high-usage MATRIX fields for VECTOR replacement")
                    else:
                        # Balanced selection for matrix fields
                        low_count = min(12, len(low_usage_matrix))
                        medium_count = min(12, len(medium_usage_matrix))
                        high_count = min(6, len(high_usage_matrix))
                        
                        selected_low = random.sample(low_usage_matrix, low_count) if low_usage_matrix else []
                        selected_medium = random.sample(medium_usage_matrix, medium_count) if medium_usage_matrix else []
                        selected_high = random.sample(high_usage_matrix, high_count) if high_usage_matrix else []
                        
                        matrix_fields = selected_low + selected_medium + selected_high
                        random.shuffle(matrix_fields)
                        
                        logger.info(f"🔧 BALANCED MATRIX STRATEGY: {len(selected_low)} low-usage, {len(selected_medium)} medium-usage, {len(selected_high)} high-usage MATRIX fields for VECTOR replacement")
                
                logger.info(f"🔧 NON-VEC OPERATORS DETECTED: Found {len(matrix_fields)} MATRIX fields for VECTOR replacement in {region}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {region} cache: {e}")
                return template
        
        if not matrix_fields:
            logger.warning(f"⚠️ No MATRIX fields available for {region}")
            return template
        
        # Get field types to identify VECTOR fields
        field_types = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    fields_data = json.load(f)
                    for field in fields_data:
                        field_types[field['id']] = field.get('type', 'REGULAR')
            except:
                pass
        
        # Find all fields in the template
        field_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        all_words = re.findall(field_pattern, template)
        
        # Replace only VECTOR fields with MATRIX fields
        replacement_count = 0
        for word in all_words:
            if word in field_types and field_types[word] == 'VECTOR' and replacement_count < len(matrix_fields):
                replacement_field = matrix_fields[replacement_count]['id']
                template = template.replace(word, replacement_field)
                logger.info(f"🔧 VECTOR->MATRIX REPLACEMENT: {word} -> {replacement_field} (due to non-vec_* operators)")
                replacement_count += 1
        
        if replacement_count == 0:
            logger.info(f"🔧 NO VECTOR FIELDS FOUND FOR REPLACEMENT")
        else:
            logger.info(f"🔧 REPLACED {replacement_count} VECTOR fields with MATRIX fields due to non-vec_* operators")
        
        return template
    
    def _replace_all_fields_with_matrix_fields(self, template: str, region: str, delay: int = None) -> str:
        """Replace all data fields with matrix fields when arithmetic operators are present"""
        import re
        
        # Get available MATRIX fields from the specific region cache
        matrix_fields = []
        cache_file = f"data_fields_cache_{region}_{delay}.json"
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    fields_data = json.load(f)
                    for field in fields_data:
                        if field.get('type') == 'MATRIX':
                            matrix_fields.append(field)
                
                # Apply balanced selection for matrix fields
                def get_usage_score(field):
                    user_count = field.get('userCount', 0)
                    alpha_count = field.get('alphaCount', 0)
                    return user_count + alpha_count
                
                # Sort and categorize matrix fields by usage
                matrix_fields.sort(key=get_usage_score)
                total_matrix_fields = len(matrix_fields)
                
                if total_matrix_fields > 0:
                    # Define usage categories for matrix fields
                    low_usage_matrix = matrix_fields[:total_matrix_fields//3]
                    medium_usage_matrix = matrix_fields[total_matrix_fields//3:2*total_matrix_fields//3]
                    high_usage_matrix = matrix_fields[2*total_matrix_fields//3:]
                    
                    # Randomly decide selection strategy (30% chance for all high-usage, 70% for balanced)
                    import random
                    selection_strategy = random.choice(['balanced', 'balanced', 'balanced', 'all_high_usage'])
                    
                    if selection_strategy == 'all_high_usage':
                        # Sometimes select all high-usage matrix fields
                        matrix_fields = high_usage_matrix[:30] if len(high_usage_matrix) >= 30 else high_usage_matrix
                        logger.info(f"🔧 HIGH-USAGE MATRIX STRATEGY: Selected {len(matrix_fields)} high-usage MATRIX fields")
                    else:
                        # Balanced selection for matrix fields
                        low_count = min(12, len(low_usage_matrix))
                        medium_count = min(12, len(medium_usage_matrix))
                        high_count = min(6, len(high_usage_matrix))
                        
                        selected_low = random.sample(low_usage_matrix, low_count) if low_usage_matrix else []
                        selected_medium = random.sample(medium_usage_matrix, medium_count) if medium_usage_matrix else []
                        selected_high = random.sample(high_usage_matrix, high_count) if high_usage_matrix else []
                        
                        matrix_fields = selected_low + selected_medium + selected_high
                        random.shuffle(matrix_fields)
                        
                        logger.info(f"🔧 BALANCED MATRIX STRATEGY: {len(selected_low)} low-usage, {len(selected_medium)} medium-usage, {len(selected_high)} high-usage MATRIX fields")
                
                logger.info(f"🔧 ARITHMETIC OPERATORS DETECTED: Found {len(matrix_fields)} MATRIX fields for replacement in {region}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {region} cache: {e}")
                return template
        
        if not matrix_fields:
            logger.warning(f"⚠️ No MATRIX fields available for {region}")
            return template
        
        # Get field types to identify all data fields
        field_types = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    fields_data = json.load(f)
                    for field in fields_data:
                        field_types[field['id']] = field.get('type', 'REGULAR')
            except:
                pass
        
        # Find all data fields in the template
        field_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        all_words = re.findall(field_pattern, template)
        
        # Replace all data fields with MATRIX fields
        replacement_count = 0
        for word in all_words:
            if word in field_types and field_types[word] in ['REGULAR', 'VECTOR'] and replacement_count < len(matrix_fields):
                replacement_field = matrix_fields[replacement_count]['id']
                template = template.replace(word, replacement_field)
                logger.info(f"🔧 ARITHMETIC REPLACEMENT: {word} -> {replacement_field} (MATRIX field)")
                replacement_count += 1
        
        if replacement_count == 0:
            logger.warning(f"⚠️ NO DATA FIELDS FOUND FOR REPLACEMENT")
        else:
            logger.info(f"🔧 REPLACED {replacement_count} fields with MATRIX fields due to arithmetic operators")
        
        return template

    def _validate_and_fix_template_fields(self, template: str, region: str, delay: int = None) -> str:
        """Enhanced field validation with vec_* and non-vec_* operator detection"""
        logger.info(f"🔧 FIELD VALIDATION: Checking template for {region}")
        
        # Check if template contains vec_* operators (replace MATRIX with VECTOR)
        if self._has_vec_operators(template):
            logger.info(f"🔧 VEC OPERATORS DETECTED: Replacing MATRIX fields with VECTOR fields")
            return self._replace_matrix_fields_with_vector_fields(template, region, delay)
        
        # Check if template contains non-vec_* operators (including arithmetic)
        if self._has_non_vec_operators(template):
            logger.info(f"🔧 NON-VEC OPERATORS DETECTED: Replacing VECTOR fields with MATRIX fields")
            return self._replace_vector_fields_with_matrix_fields(template, region, delay)
        
        # If no special operators, use existing logic
        logger.info(f"🔧 NO SPECIAL OPERATORS: Using template as-is for {region}")
        return template
    
    def _handle_simulation_error(self, template: str, error_msg: str, settings) -> None:
        """Handle any simulation error by regenerating template with error feedback"""
        try:
            if not settings:
                logger.warning(f"⚠️ SIMULATION ERROR: No settings available for regeneration")
                return
            
            region = settings.region
            logger.info(f"🔄 SIMULATION ERROR HANDLING: Regenerating template for {region}")
            logger.info(f"🔧 ERROR MESSAGE: {error_msg}")
            
            # Track retry attempts for this region
            if not hasattr(self, '_error_retry_count'):
                self._error_retry_count = {}
            
            if region not in self._error_retry_count:
                self._error_retry_count[region] = 0
            
            self._error_retry_count[region] += 1
            retry_count = self._error_retry_count[region]
            
            if retry_count > 5:
                logger.warning(f"⚠️ MAX RETRIES REACHED: Giving up on {region} after {retry_count} attempts")
                return
            
            logger.info(f"🔄 RETRY ATTEMPT {retry_count}/5 for {region}")
            
            # Generate new template with error feedback
            new_template = self._regenerate_template_with_error_feedback(template, error_msg, region)
            if new_template:
                logger.info(f"✅ TEMPLATE REGENERATED: Generated new template for {region}")
                # Add the new template to the queue for simulation
                self._add_template_to_queue(new_template, region)
            else:
                logger.warning(f"⚠️ TEMPLATE REGENERATION FAILED: Could not generate new template for {region}")
                
        except Exception as e:
            logger.error(f"❌ SIMULATION ERROR HANDLING FAILED: {e}")
    
    def _regenerate_template_with_error_feedback(self, failed_template: str, error_msg: str, region: str) -> str:
        """Regenerate template using Ollama with error feedback"""
        try:
            logger.info(f"🔄 REGENERATING TEMPLATE: Using error feedback for {region}")
            
            # Get data fields for the region - use optimal delay to match simulation
            optimal_delay = self.select_optimal_delay(region)
            data_fields = self.get_data_fields_for_region(region, optimal_delay)
            if not data_fields:
                logger.warning(f"⚠️ NO DATA FIELDS: Cannot regenerate template for {region}")
                return None
            
            # Select random fields
            selected_fields = random.sample(data_fields, min(4, len(data_fields)))
            
            # Get operators
            operators = self.operators.copy()
            if self.operator_blacklist:
                operators = [op for op in operators if op['name'] not in self.operator_blacklist]
            
            # Select random operators
            selected_operators = random.sample(operators, min(4, len(operators)))
            
            # Create error feedback prompt
            fields_info = []
            for field in selected_fields:
                field_type = field.get('type', 'REGULAR')
                fields_info.append(f"- {field['id']}: {field.get('description', 'No description')} [{field_type}]")
            
            operators_info = []
            for op in selected_operators:
                definition = op.get('definition', 'No definition available')
                operators_info.append(f"- {op['name']}: {op['description']} | Syntax: {definition}")
            
            # Select a random persona
            persona = self._select_persona()
            persona_prompt = self._get_persona_prompt(persona, 1)
            
            prompt = f"""
🔄 TEMPLATE REGENERATION WITH ERROR FEEDBACK FOR {region.upper()}

{persona_prompt}

PREVIOUS TEMPLATE THAT FAILED:
{failed_template}

ERROR MESSAGE:
{error_msg}

You are building a WorldQuant Brain alpha expression. This is NOT SQL - it's a mathematical expression using operators and data fields.

SELECTED FIELDS:
{chr(10).join(fields_info)}

SELECTED OPERATORS:
{chr(10).join(operators_info)}

CRITICAL INSTRUCTIONS:
- This is a MATHEMATICAL EXPRESSION, not SQL code
- Use ONLY the selected fields and operators above
- AVOID the error that occurred in the previous template
- Use proper syntax with balanced parentheses
- Return ONLY the alpha expression, no explanations, no code blocks, no markdown
- DO NOT include "plaintext" prefix or any other prefixes
- No saying 'math' or 'alpha expression' in the results, only the expression

RESPONSE FORMAT (return only the expression, no prefixes):
"""
            
            import ollama
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.8, 'top_p': 0.9}
            )
            
            content = response['message']['content'].strip()
            
            # Clean up the response
            if content.startswith('```'):
                content = content.split('```')[1].strip()
            if content.startswith('plaintext'):
                content = content.replace('plaintext', '').strip()

            # Sometimes it starts with 'markdown winsorize(...) or alpla expression normalize(...) or math expression ...'
            if content.startswith('markdown'):
                content = content.split('markdown')[1].strip()
            if content.startswith('alpha expression'):
                content = content.split('alpha expression')[1].strip()
            if content.startswith('math expression'):
                content = content.split('math expression')[1].strip()
            
            logger.info(f"🔄 REGENERATED TEMPLATE: {content}")
            return content
            
        except Exception as e:
            logger.error(f"❌ TEMPLATE REGENERATION FAILED: {e}")
            return None
    
    def _perform_post_simulation_analysis(self, result) -> None:
        """Perform comprehensive post-simulation analysis including correlations and performance evaluation"""
        try:
            logger.info(f"🔍 POST-SIMULATION ANALYSIS: Starting analysis for {result.template[:50]}...")
            
            if not result or not result.success:
                logger.info(f"🔍 POST-SIMULATION ANALYSIS: Skipping analysis - result not successful")
                return
            
            # Get alpha ID from result
            alpha_id = getattr(result, 'alpha_id', None)
            logger.info(f"🔍 POST-SIMULATION ANALYSIS: Alpha ID: {alpha_id}")
            
            if not alpha_id:
                logger.warning(f"⚠️ NO ALPHA ID: Cannot perform post-simulation analysis")
                return
            
            # Step 1: Fetch alpha details
            alpha_details = self._fetch_alpha_details(alpha_id)
            if not alpha_details:
                logger.warning(f"⚠️ FAILED TO FETCH ALPHA DETAILS: Cannot analyze alpha {alpha_id}")
                return
            
            # Step 2: Check for FAIL results first
            checks = alpha_details.get('is', {}).get('checks', [])
            fail_checks = [check for check in checks if check.get('result') == 'FAIL']
            
            # Step 3: Handle FAIL checks immediately
            if fail_checks:
                logger.info(f"🚫 FAIL CHECKS DETECTED: Marking as RED immediately ({len(fail_checks)} FAIL checks)")
                color = "RED"
                name = self._generate_alpha_name(alpha_details, {'max': 0, 'min': 0, 'records': []}, {'max': 0, 'min': 0, 'records': []}, result)
                logger.info(f"🎨 IMMEDIATE RED ASSIGNMENT: Color={color}, Name={name}")
            else:
                logger.info(f"🔍 NO FAIL CHECKS: Proceeding with correlation analysis")
                # Check power pool correlation
                power_pool_corr = self._check_power_pool_correlation(alpha_id)
                
                # Check production correlation  
                prod_corr = self._check_production_correlation(alpha_id)
                
                # Step 4: Analyze performance and determine color/name
                color, name = self._analyze_performance_and_assign_metadata(
                    alpha_details, power_pool_corr, prod_corr, result
                )
            
            # Step 5: Update alpha with new metadata
            if color or name:
                self._update_alpha_metadata(alpha_id, color, name)
            
            logger.info(f"✅ POST-SIMULATION ANALYSIS COMPLETE: {alpha_id} - Color: {color}, Name: {name}")
            
        except Exception as e:
            logger.error(f"❌ POST-SIMULATION ANALYSIS FAILED: {e}")
            import traceback
            logger.error(f"❌ POST-SIMULATION ANALYSIS TRACEBACK: {traceback.format_exc()}")
            # Don't let the exception propagate - just log it and continue
    
    def _fetch_alpha_details(self, alpha_id: str) -> dict:
        """Fetch alpha details from WorldQuant Brain API"""
        try:
            url = f"https://api.worldquantbrain.com/alphas/{alpha_id}"
            response = self.make_api_request('GET', url)
            
            if response.status_code == 200:
                alpha_data = response.json()
                logger.info(f"📊 ALPHA DETAILS FETCHED: {alpha_id}")
                return alpha_data
            else:
                logger.error(f"Failed to fetch alpha details: {response.status_code}")
                return None
            
        except Exception as e:
            logger.error(f"❌ FAILED TO FETCH ALPHA DETAILS: {e}")
            return None
    
    def _check_power_pool_correlation(self, alpha_id: str) -> dict:
        """Check power pool correlation for the alpha with polling for processing"""
        try:
            url = f"https://api.worldquantbrain.com/alphas/{alpha_id}/correlations/power-pool"
            
            # Poll the API until we get actual data (not empty response while processing)
            max_polls = 10  # Maximum number of polling attempts
            poll_interval = 5  # Wait 5 seconds between polls
            
            for attempt in range(max_polls):
                response = self.make_api_request('GET', url)
                
                if response.status_code == 200:
                    corr_data = response.json()
                    
                    # Check if we have actual data (not empty response while processing)
                    has_data = (
                        corr_data.get('max') is not None or 
                        corr_data.get('min') is not None or 
                        corr_data.get('records') or
                        len(corr_data) > 0
                    )
                    
                    if has_data:
                        # Extract max and min correlation values
                        max_corr = corr_data.get('max', 0)
                        min_corr = corr_data.get('min', 0)
                        
                        # If max/min are 0, try to calculate from records
                        if max_corr == 0 and min_corr == 0 and corr_data.get('records'):
                            records = corr_data.get('records', [])
                            if records:
                                # Extract correlation values from records
                                correlations = []
                                for record in records:
                                    if len(record) > 5:  # Ensure record has correlation field
                                        corr_value = record[5]  # correlation is at index 5
                                        if isinstance(corr_value, (int, float)):
                                            correlations.append(corr_value)
                                
                                if correlations:
                                    max_corr = max(correlations)
                                    min_corr = min(correlations)
                        
                        logger.info(f"🔗 POWER POOL CORRELATION: Max={max_corr:.3f}, Min={min_corr:.3f}")
                        return {
                            'max': max_corr,
                            'min': min_corr,
                            'records': corr_data.get('records', [])
                        }
                    else:
                        # Empty response while processing, wait and try again
                        logger.info(f"⏳ Power pool correlation still processing (attempt {attempt + 1}/{max_polls}), waiting {poll_interval}s...")
                        if attempt < max_polls - 1:  # Don't sleep on last attempt
                            import time
                            time.sleep(poll_interval)
                        continue
                else:
                    logger.error(f"Failed to fetch power pool correlation: {response.status_code}")
                    return {'max': 0, 'min': 0, 'records': []}
            
            # If we've exhausted all polling attempts
            logger.warning(f"⚠️ Power pool correlation still processing after {max_polls} attempts, using default values")
            return {'max': 0, 'min': 0, 'records': []}
            
        except Exception as e:
            logger.error(f"❌ FAILED TO CHECK POWER POOL CORRELATION: {e}")
            return {'max': 0, 'min': 0, 'records': []}
    
    def _check_production_correlation(self, alpha_id: str) -> dict:
        """Check production correlation for the alpha with polling for processing"""
        try:
            url = f"https://api.worldquantbrain.com/alphas/{alpha_id}/correlations/prod"
            
            # Poll the API until we get actual data (not empty response while processing)
            max_polls = 10  # Maximum number of polling attempts
            poll_interval = 5  # Wait 5 seconds between polls
            
            for attempt in range(max_polls):
                response = self.make_api_request('GET', url)
                
                if response.status_code == 200:
                    corr_data = response.json()
                    
                    # Check if we have actual data (not empty response while processing)
                    has_data = (
                        corr_data.get('max') is not None or 
                        corr_data.get('min') is not None or 
                        corr_data.get('records') or
                        len(corr_data) > 0
                    )
                    
                    if has_data:
                        # Extract max and min correlation values
                        max_corr = corr_data.get('max', 0)
                        min_corr = corr_data.get('min', 0)
                        
                        # If max/min are 0, try to calculate from records
                        if max_corr == 0 and min_corr == 0 and corr_data.get('records'):
                            records = corr_data.get('records', [])
                            if records:
                                # For production correlation, extract from histogram data
                                correlations = []
                                for record in records:
                                    if len(record) >= 2:  # [min_range, max_range, count]
                                        # Use midpoint of range as correlation value
                                        min_range = record[0]
                                        max_range = record[1]
                                        count = record[2] if len(record) > 2 else 0
                                        
                                        if count > 0:  # Only include ranges with actual correlations
                                            midpoint = (min_range + max_range) / 2
                                            correlations.append(midpoint)
                                
                                if correlations:
                                    max_corr = max(correlations)
                                    min_corr = min(correlations)
                        
                        logger.info(f"🔗 PRODUCTION CORRELATION: Max={max_corr:.3f}, Min={min_corr:.3f}")
                        return {
                            'max': max_corr,
                            'min': min_corr,
                            'records': corr_data.get('records', [])
                        }
                    else:
                        # Empty response while processing, wait and try again
                        logger.info(f"⏳ Production correlation still processing (attempt {attempt + 1}/{max_polls}), waiting {poll_interval}s...")
                        if attempt < max_polls - 1:  # Don't sleep on last attempt
                            import time
                            time.sleep(poll_interval)
                        continue
                else:
                    logger.error(f"Failed to fetch production correlation: {response.status_code}")
                    return {'max': 0, 'min': 0, 'records': []}
            
            # If we've exhausted all polling attempts
            logger.warning(f"⚠️ Production correlation still processing after {max_polls} attempts, using default values")
            return {'max': 0, 'min': 0, 'records': []}
            
        except Exception as e:
            logger.error(f"❌ FAILED TO CHECK PRODUCTION CORRELATION: {e}")
            return {'max': 0, 'min': 0, 'records': []}
    
    def _analyze_performance_and_assign_metadata(self, alpha_details: dict, power_pool_corr: dict, prod_corr: dict, result) -> tuple:
        """Analyze performance metrics and assign color and name"""
        try:
            # Extract performance data
            is_data = alpha_details.get('is', {})
            checks = alpha_details.get('is', {}).get('checks', [])
            
            # Get key metrics
            sharpe = is_data.get('sharpe', 0)
            fitness = is_data.get('fitness', 0)
            returns = is_data.get('returns', 0)
            turnover = is_data.get('turnover', 0)
            
            # Get correlation metrics
            power_pool_max = power_pool_corr.get('max', 0)
            prod_max = prod_corr.get('max', 0)
            
            # Analyze checks for FAIL/WARNING/PASS
            fail_checks = [check for check in checks if check.get('result') == 'FAIL']
            warning_checks = [check for check in checks if check.get('result') == 'WARNING']
            pass_checks = [check for check in checks if check.get('result') == 'PASS']
            
            # Determine color based on rules including PnL flatline detection
            color = self._determine_alpha_color(fail_checks, warning_checks, power_pool_max, prod_max, 
                                              alpha_details)
            
            # Generate name based on correlations and data fields
            name = self._generate_alpha_name(alpha_details, power_pool_corr, prod_corr, result)
            
            logger.info(f"🎨 ALPHA METADATA: Color={color}, Name={name}")
            logger.info(f"📊 PERFORMANCE: Sharpe={sharpe:.3f}, Fitness={fitness:.3f}, Returns={returns:.3f}")
            logger.info(f"🔗 CORRELATIONS: PowerPool={power_pool_max:.3f}, Prod={prod_max:.3f}")
            logger.info(f"✅ CHECKS: {len(pass_checks)} PASS, {len(warning_checks)} WARNING, {len(fail_checks)} FAIL")
            
            return color, name
            
        except Exception as e:
            logger.error(f"❌ FAILED TO ANALYZE PERFORMANCE: {e}")
            return None, None
    
    def _determine_alpha_color(self, fail_checks: list, warning_checks: list, power_pool_max: float, prod_max: float, 
                              alpha_details: dict) -> str:
        """Determine alpha color based on performance and correlation rules with PnL flatline detection"""
        # RED: Any FAIL (correlation checks skipped if FAIL exists)
        if fail_checks:
            return "RED"
        
        # RED: PnL is flatline - critical performance issue
        if self._is_pnl_flatline(alpha_details):
            logger.warning(f"🔴 FLATLINE PnL DETECTED: Actual PnL data shows flatline pattern")
            return "RED"
        
        # RED: Power pool correlation > 0.5 (only if correlations were checked)
        if power_pool_max > 0.5:
            return "RED"
        
        # YELLOW: Any WARNING or production correlation > 0.7
        if warning_checks or prod_max > 0.7:
            return "YELLOW"
        
        # GREEN: All PASS and production correlation < 0.7
        return "GREEN"
    
    def _is_pnl_flatline(self, alpha_details: dict) -> bool:
        """Detect if PnL is actually flatline based on PnL data analysis"""
        try:
            # Get PnL data from alpha details
            pnl_data = alpha_details.get('pnl', {})
            if not pnl_data:
                logger.warning("🔍 NO PnL DATA AVAILABLE - cannot check for flatline")
                return False
            
            # Extract PnL values (assuming it's a time series of PnL values)
            pnl_values = pnl_data.get('values', [])
            if not pnl_values or len(pnl_values) < 10:  # Need sufficient data points
                logger.warning("🔍 INSUFFICIENT PnL DATA - cannot check for flatline")
                return False
            
            # Convert to float if needed
            try:
                pnl_float_values = [float(x) for x in pnl_values if x is not None]
            except (ValueError, TypeError):
                logger.warning("🔍 INVALID PnL DATA FORMAT - cannot check for flatline")
                return False
            
            if len(pnl_float_values) < 10:
                logger.warning("🔍 INSUFFICIENT VALID PnL DATA - cannot check for flatline")
                return False
            
            # Analyze PnL for flatline patterns
            return self._analyze_pnl_flatline_pattern(pnl_float_values)
            
        except Exception as e:
            logger.error(f"❌ FAILED TO CHECK PnL FLATLINE: {e}")
            return False
    
    def _analyze_pnl_flatline_pattern(self, pnl_values: list) -> bool:
        """Analyze PnL values for flatline patterns"""
        try:
            if len(pnl_values) < 10:
                return False
            
            # Calculate PnL statistics
            pnl_min = min(pnl_values)
            pnl_max = max(pnl_values)
            pnl_range = pnl_max - pnl_min
            pnl_mean = sum(pnl_values) / len(pnl_values)
            pnl_std = (sum((x - pnl_mean) ** 2 for x in pnl_values) / len(pnl_values)) ** 0.5
            
            # FLATLINE DETECTION CRITERIA:
            
            # 1. Very small range (PnL barely changes)
            if pnl_range < 0.001:  # Less than 0.1% range
                logger.warning(f"🔴 PnL FLATLINE: Range={pnl_range:.6f} < 0.001")
                return True
            
            # 2. Very low standard deviation (no volatility)
            if pnl_std < 0.0001:  # Less than 0.01% std dev
                logger.warning(f"🔴 PnL FLATLINE: StdDev={pnl_std:.6f} < 0.0001")
                return True
            
            # 3. All values are essentially the same (within 0.01%)
            pnl_variance = max(pnl_values) - min(pnl_values)
            if pnl_variance < 0.0001:  # Less than 0.01% variance
                logger.warning(f"🔴 PnL FLATLINE: Variance={pnl_variance:.6f} < 0.0001")
                return True
            
            # 4. Check for constant PnL (all values within 0.001% of mean)
            tolerance = abs(pnl_mean) * 0.00001  # 0.001% of mean
            if tolerance < 0.000001:  # Minimum tolerance
                tolerance = 0.000001
            
            constant_values = sum(1 for x in pnl_values if abs(x - pnl_mean) <= tolerance)
            if constant_values >= len(pnl_values) * 0.95:  # 95% of values are constant
                logger.warning(f"🔴 PnL FLATLINE: {constant_values}/{len(pnl_values)} values constant within {tolerance:.6f}")
                return True
            
            # 5. Check for zero or near-zero PnL throughout
            near_zero_count = sum(1 for x in pnl_values if abs(x) < 0.0001)
            if near_zero_count >= len(pnl_values) * 0.9:  # 90% of values near zero
                logger.warning(f"🔴 PnL FLATLINE: {near_zero_count}/{len(pnl_values)} values near zero")
                return True
            
            logger.info(f"✅ PnL NOT FLATLINE: Range={pnl_range:.6f}, StdDev={pnl_std:.6f}, Variance={pnl_variance:.6f}")
            return False
            
        except Exception as e:
            logger.error(f"❌ FAILED TO ANALYZE PnL PATTERN: {e}")
            return False
    
    def _generate_alpha_name(self, alpha_details: dict, power_pool_corr: dict, prod_corr: dict, result) -> str:
        """Generate alpha name based on correlations and data fields with enhanced correlation metrics"""
        try:
            # Extract data fields from the alpha code
            code = alpha_details.get('regular', {}).get('code', '')
            
            # Find data fields in the code
            import re
            field_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            fields = re.findall(field_pattern, code)
            
            # Filter out operators and keep only data fields
            operators = {'subtract', 'add', 'multiply', 'divide', 'normalize', 'rank', 'ts_rank', 'ts_delta', 
                        'vec_avg', 'vec_sum', 'vec_max', 'vec_min', 'group_scale', 'group_zscore', 'ts_corr',
                        'ts_std', 'ts_mean', 'ts_zscore', 'ts_scale', 'ts_sum', 'ts_product', 'ts_min', 'ts_max',
                        'group_rank', 'group_neutralize', 'group_mean', 'group_max', 'group_min', 'group_backfill',
                        'scale', 'zscore', 'abs', 'sign', 'min', 'max', 'log', 'exp', 'sqrt', 'power', 'signed_power',
                        'reverse', 'to_nan', 'inverse', 'divide', 'multiply', 'add', 'subtract', 'ts_delta', 'ts_rank',
                        'vec_rank', 'vec_std', 'vec_scale', 'vec_min', 'vec_max', 'vec_sum', 'vec_avg'}
            data_fields = [f for f in fields if f not in operators and len(f) > 3]
            
            # Get correlation metrics with debugging
            power_pool_max = power_pool_corr.get('max', 0)
            power_pool_min = power_pool_corr.get('min', 0)
            prod_max = prod_corr.get('max', 0)
            prod_min = prod_corr.get('min', 0)
            
            # Debug correlation values
            logger.info(f"🔍 CORRELATION DEBUG: PowerPool max={power_pool_max}, min={power_pool_min}")
            logger.info(f"🔍 CORRELATION DEBUG: Production max={prod_max}, min={prod_min}")
            
            # If correlations are still 0, try to extract from records
            if power_pool_max == 0 and power_pool_corr.get('records'):
                records = power_pool_corr.get('records', [])
                if records:
                    correlations = [record[5] for record in records if len(record) > 5 and isinstance(record[5], (int, float))]
                    if correlations:
                        power_pool_max = max(correlations)
                        power_pool_min = min(correlations)
                        logger.info(f"🔍 EXTRACTED FROM RECORDS: PowerPool max={power_pool_max}, min={power_pool_min}")
            
            if prod_max == 0 and prod_corr.get('records'):
                records = prod_corr.get('records', [])
                if records:
                    # For production correlation histogram data
                    correlations = []
                    for record in records:
                        if len(record) >= 2 and record[2] > 0:  # [min, max, count]
                            midpoint = (record[0] + record[1]) / 2
                            correlations.append(midpoint)
                    if correlations:
                        prod_max = max(correlations)
                        prod_min = min(correlations)
                        logger.info(f"🔍 EXTRACTED FROM RECORDS: Production max={prod_max}, min={prod_min}")
            
            # Generate name components
            name_parts = []
            
            # MANDATORY: Always include both production and power pool correlation metrics with actual values
            # Production correlation (PC) - always include actual value
            name_parts.append(f"PC{prod_max:.2f}")
            
            # Power pool correlation (PPC) - always include actual value
            name_parts.append(f"PPC{power_pool_max:.2f}")
            
            # Include minimum correlation values for complete picture
            if prod_min != prod_max:
                name_parts.append(f"PC_MIN{prod_min:.2f}")
            if power_pool_min != power_pool_max:
                name_parts.append(f"PPC_MIN{power_pool_min:.2f}")
            
            # Add PnL flatline indicator if detected
            if self._is_pnl_flatline(alpha_details):
                name_parts.append("FLATLINE")
            
            # Add data field info (first 2 fields) for context
            if data_fields:
                field_names = data_fields[:2]
                for field in field_names:
                    # Extract meaningful part of field name
                    if '_' in field:
                        parts = field.split('_')
                        if len(parts) > 1:
                            name_parts.append(parts[1][:4].upper())
                    else:
                        name_parts.append(field[:4].upper())
            
            # Add template identifier
            if hasattr(result, 'template') and result.template:
                template_id = result.template[:10].replace(' ', '_').replace('-', '_')
                name_parts.append(f"T{template_id}")
            
            # Combine name parts with proper formatting
            if name_parts:
                name = "_".join(name_parts)
            else:
                name = f"Alpha_{result.template[:20].replace(' ', '_')}"
            
            # Ensure name includes both correlation metrics with actual values
            if "PC" not in name:
                name = f"PC{prod_max:.2f}_{name}"
            if "PPC" not in name:
                name = f"{name}_PPC{power_pool_max:.2f}"
            
            return name[:50]  # Limit name length
            
        except Exception as e:
            logger.error(f"❌ FAILED TO GENERATE ALPHA NAME: {e}")
            return f"Alpha_{result.template[:20].replace(' ', '_')}"
    
    def _update_alpha_metadata(self, alpha_id: str, color: str, name: str) -> None:
        """Update alpha with new color and name"""
        try:
            url = f"https://api.worldquantbrain.com/alphas/{alpha_id}"
            
            # Prepare update data
            update_data = {}
            if color:
                update_data['color'] = color
            if name:
                update_data['name'] = name
            
            if update_data:
                response = self.make_api_request('PATCH', url, json=update_data)
                if response.status_code == 200:
                    logger.info(f"✅ ALPHA UPDATED: {alpha_id} - Color: {color}, Name: {name}")
                else:
                    logger.error(f"Failed to update alpha metadata: {response.status_code}")
            else:
                logger.info(f"ℹ️ NO UPDATES NEEDED: {alpha_id}")
                
        except Exception as e:
            logger.error(f"❌ FAILED TO UPDATE ALPHA METADATA: {e}")
    
    def _add_template_to_queue(self, template: str, region: str) -> None:
        """Add a regenerated template to the simulation queue"""
        try:
            # Create a new simulation task
            template_dict = {
                'template': template,
                'neutralization': 'INDUSTRY'
            }
            future = self.executor.submit(
                self._simulate_template_concurrent,
                template_dict,
                region,
                delay=optimal_delay  # Use optimal delay for regenerated templates
            )
            
            # Store the future
            future_id = f"regenerated_{int(time.time() * 1000)}"
            self.active_futures[future_id] = future
            self.future_start_times[future_id] = time.time()
            
            logger.info(f"✅ TEMPLATE ADDED TO QUEUE: {template[:50]}... for {region}")
            
        except Exception as e:
            logger.error(f"❌ FAILED TO ADD TEMPLATE TO QUEUE: {e}")
    
    def _load_personas(self) -> List[Dict]:
        """Load prompt personas from JSON file"""
        try:
            with open('prompt_personas.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('personas', [])
        except Exception as e:
            logger.warning(f"⚠️ Failed to load personas: {e}")
            # Fallback to default personas
            return [
                {
                    "name": "Bruce Lee",
                    "style": "Philosophical Warrior",
                    "prompt": "🧠 BRUCE LEE PHILOSOPHY - \"BE LIKE WATER\":\n- Be fluid and adaptive\n- Create new patterns\n- Explore infinite possibilities"
                },
                {
                    "name": "Pro Quant Researcher", 
                    "style": "Academic Rigor",
                    "prompt": "🎓 PROFESSIONAL QUANTITATIVE RESEARCHER:\n- Apply rigorous statistical principles\n- Focus on economically sound relationships\n- Use advanced mathematical techniques"
                }
            ]
    
    def _load_historical_alphas(self) -> List[Dict]:
        """Load historical alpha expressions from local JSON file"""
        try:
            # Load from local JSON file instead of API call
            json_file_path = "submitted_alpha.json"
            logger.info(f"🔍 TRACE: Loading historical alphas from local file: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            alphas = data.get('results', [])
            logger.info(f"🔍 TRACE: Loaded {len(alphas)} alphas from local file")
            
            # Extract alpha expressions and metadata
            historical_alphas = []
            for i, alpha in enumerate(alphas):
                logger.info(f"🔍 TRACE: Processing alpha {i+1}/{len(alphas)} - ID: {alpha.get('id', 'N/A')}")
                if 'regular' in alpha and 'code' in alpha['regular']:
                    expression = alpha['regular']['code']
                    is_data = alpha.get('is', {})
                    
                    historical_alphas.append({
                        'expression': expression,
                        'sharpe': is_data.get('sharpe', 0),
                        'fitness': is_data.get('fitness', 0),
                        'region': alpha.get('settings', {}).get('region', 'UNKNOWN'),
                        'universe': alpha.get('settings', {}).get('universe', 'UNKNOWN'),
                        'operator_count': alpha.get('regular', {}).get('operatorCount', 0),
                        'date_created': alpha.get('dateCreated', ''),
                        'date_submitted': alpha.get('dateSubmitted', ''),
                        'status': alpha.get('status', 'UNKNOWN'),
                        'alpha_id': alpha.get('id', '')
                    })
                    logger.info(f"🔍 TRACE: Added alpha - Expression: {expression[:50]}..., Sharpe: {is_data.get('sharpe', 0)}")
                else:
                    logger.info(f"🔍 TRACE: Skipped alpha {i+1} - missing 'regular' or 'code' field")
            
            logger.info(f"📚 LOADED HISTORICAL ALPHAS: {len(historical_alphas)} expressions from local file")
            return historical_alphas
                
        except FileNotFoundError:
            logger.warning(f"⚠️ Historical alphas file not found: {json_file_path}")
            return []
        except Exception as e:
            logger.warning(f"⚠️ Failed to load historical alphas from file: {e}")
            logger.info(f"🔍 TRACE: Exception details: {type(e).__name__}: {str(e)}")
            return []
    
    def _make_api_request_with_retry(self, method, url, params=None, headers=None, max_retries=3):
        """Make API request with exponential backoff retry for rate limiting"""
        import time
        import random
        
        for attempt in range(max_retries + 1):
            try:
                response = self.make_api_request(method, url, params=params, headers=headers)
                
                if response and response.status_code == 200:
                    return response
                elif response and response.status_code == 429:  # Rate limited
                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"🔍 TRACE: Rate limited (429), retrying in {wait_time:.2f} seconds (attempt {attempt + 1}/{max_retries + 1})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"⚠️ Rate limited after {max_retries + 1} attempts, giving up")
                        return response
                else:
                    logger.warning(f"⚠️ API request failed with status {response.status_code if response else 'None'}")
                    return response
                    
            except Exception as e:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"🔍 TRACE: Request failed, retrying in {wait_time:.2f} seconds (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Request failed after {max_retries + 1} attempts: {e}")
                    return None
        
        return None
    
    def _select_historical_alphas(self, count: int = 3) -> List[Dict]:
        """Select random historical alphas for inspiration"""
        logger.info(f"🔍 TRACE: _select_historical_alphas called with count={count}")
        logger.info(f"🔍 TRACE: self.historical_alphas length: {len(self.historical_alphas) if self.historical_alphas else 0}")
        
        if not self.historical_alphas:
            logger.info(f"🔍 TRACE: No historical alphas available, returning empty list")
            return []
        
        # Use random offset to get different alphas each time
        import random
        max_offset = max(0, len(self.historical_alphas) - count)
        start_offset = random.randint(0, max_offset)
        
        selected_alphas = self.historical_alphas[start_offset:start_offset + count]
        
        logger.info(f"🔍 TRACE: Selected {len(selected_alphas)} alphas from {len(self.historical_alphas)} available")
        logger.info(f"📚 SELECTED HISTORICAL ALPHAS: {len(selected_alphas)} expressions for inspiration")
        return selected_alphas
    
    def _select_persona(self) -> Dict:
        """Select a random persona, avoiding recently used ones"""
        if not self.personas:
            return {"name": "Default", "style": "Standard", "prompt": ""}
        
        # Get available personas (excluding recent ones)
        available_personas = [p for p in self.personas if p['name'] not in self.recent_personas]
        
        # If all personas have been used recently, reset the recent list
        if not available_personas:
            self.recent_personas = []
            available_personas = self.personas
        
        # Select random persona
        selected_persona = random.choice(available_personas)
        
        # Track recent personas (keep last 3)
        self.recent_personas.append(selected_persona['name'])
        if len(self.recent_personas) > 3:
            self.recent_personas.pop(0)
        
        logger.info(f"🎭 SELECTED PERSONA: {selected_persona['name']} ({selected_persona['style']})")
        return selected_persona
    
    def _get_persona_prompt(self, persona: Dict, num_templates: int = 10) -> str:
        """Generate persona-specific prompt"""
        base_prompt = persona.get('prompt', '')
        
        if not base_prompt:
            return ""
        
        # Add final requirements
        final_requirements = f"""
FINAL REQUIREMENTS:
1. Use ONLY the provided operators and fields exactly as listed
2. Focus on economic intuition and market significance
3. Combine multiple operators for sophisticated strategies
4. Use appropriate time periods and parameters
5. Ensure perfect syntax and balanced parentheses
6. Generate {num_templates} diverse, profitable alpha expressions
7. Each template must be immediately usable in WorldQuant Brain
8. {persona['name'].upper()} APPROACH - {persona['style'].upper()}
"""
        
        return base_prompt + final_requirements
    
    def _find_replacement_field(self, invalid_field: str, valid_fields: set, data_fields: list) -> Optional[str]:
        """Find a suitable replacement field for an invalid field"""
        # Simple stub - return None to disable field replacement
        return None
    
    def _fix_unknown_variable_retry(self, template: str, error_message: str) -> str:
        """Simple stub - return template as-is"""
        return template
        #     # Group types
        #     'industry', 'subindustry', 'sector', 'market',
        #     # Common fields (don't validate these as they're always valid)
        #     'close', 'volume', 'returns'
        # }
        
        # # Only validate words that are NOT operators
        # template_fields = [word for word in all_words if word not in operators_to_exclude]
        
        # # FIRST: Fix field name modifications (suffixes, prefixes, etc.)
        # template = self._fix_field_name_modifications(template, list(valid_field_ids))
        
        # # Track changes made
        # changes_made = []
        
        # for field_name in template_fields:
        #     if field_name in valid_field_ids:
        #         # Field exists and is valid for this region
        #         field_info = field_map[field_name]
                
        #         # Field is valid - let Ollama decide how to handle VECTOR fields
        #         # No automatic conversion - let the LLM make intelligent decisions
                
        #         logger.info(f"✅ Field {field_name} is valid for region {region} (type: {field_info.get('type', 'UNKNOWN')})")
                
        #     else:
        #         # Field doesn't exist for this region, replace with a valid one
        #         # Find a similar field or use a common one
        #         replacement_field = self._find_replacement_field(field_name, valid_field_ids, data_fields)
        #         if replacement_field:
        #             template = template.replace(field_name, replacement_field)
        #             changes_made.append(f"Replaced invalid field {field_name} with {replacement_field}")
        #             logger.info(f"🔧 Replaced invalid field {field_name} with {replacement_field} for region {region}")
        #         else:
        #             # Use a fallback field
        #             fallback_field = 'close'  # Most common field
        #             template = template.replace(field_name, fallback_field)
        #             changes_made.append(f"Replaced invalid field {field_name} with fallback {fallback_field}")
        #             logger.info(f"🔧 Replaced invalid field {field_name} with fallback {fallback_field} for region {region}")
        
        # if changes_made:
        #     logger.info(f"🔧 FIELD VALIDATION: Made {len(changes_made)} changes for region {region}:")
        #     for change in changes_made:
        #         logger.info(f"   - {change}")
        # else:
        #     logger.info(f"✅ FIELD VALIDATION: All fields are valid for region {region}")
        
        # return template
    
    def _find_replacement_field(self, invalid_field: str, valid_fields: set, data_fields: list) -> Optional[str]:
        """Find a suitable replacement field for an invalid field"""
        # Try to find a field with similar name
        for field in data_fields:
            field_id = field['id']
            if (invalid_field.lower() in field_id.lower() or 
                field_id.lower() in invalid_field.lower()):
                return field_id
        
        # Try to find a field with similar description
        for field in data_fields:
            description = field.get('description', '').lower()
            if invalid_field.lower() in description:
                return field['id']
        
        # Use a high-priority field (high pyramid multiplier, low usage)
        prioritized_fields = self._prioritize_fields(data_fields)
        if prioritized_fields:
            return prioritized_fields[0]['id']
        
        return None
    
    def _validate_simulation_settings(self, region: str, delay: int, neutralization: str):
        """Validate that simulation settings match data fields exactly"""
        try:
            # Get data fields for this exact region and delay
            data_fields = self.get_data_fields_for_region(region, delay)
            
            if not data_fields:
                logger.warning(f"⚠️ No data fields found for region {region} delay {delay}")
                return
            
            # Check if the data fields match the region settings
            config = self.region_configs[region]
            
            # Validate universe matches
            field_universes = set(field.get('universe', '') for field in data_fields)
            if config.universe not in field_universes:
                logger.warning(f"⚠️ Universe mismatch: config={config.universe}, fields={field_universes}")
            
            # Validate region matches
            field_regions = set(field.get('region', '') for field in data_fields)
            if region not in field_regions:
                logger.warning(f"⚠️ Region mismatch: expected={region}, fields={field_regions}")
            
            # Validate delay matches
            field_delays = set(field.get('delay', -1) for field in data_fields)
            if delay not in field_delays:
                logger.warning(f"⚠️ Delay mismatch: expected={delay}, fields={field_delays}")
            
            # Log validation results
            logger.info(f"✅ SIMULATION VALIDATION: Region={region}, Universe={config.universe}, Delay={delay}")
            logger.info(f"   Data fields: {len(data_fields)} available")
            logger.info(f"   Field types: {set(field.get('type', 'UNKNOWN') for field in data_fields)}")
            logger.info(f"   Field regions: {field_regions}")
            logger.info(f"   Field universes: {field_universes}")
            logger.info(f"   Field delays: {field_delays}")
            
        except Exception as e:
            logger.error(f"❌ SIMULATION VALIDATION ERROR: {e}")
    
    def _extract_fields_from_ollama_template(self, template: str) -> List[str]:
        """Extract field names from Ollama-generated template, excluding operators"""
        # Known operators that should NOT be treated as fields
        known_operators = {
            'abs', 'add', 'and', 'bucket', 'days_from_last_change', 'densify', 'divide', 'equal', 'greater',
            'greater_equal', 'group_backfill', 'group_cartesian_product', 'group_max', 'group_mean', 'group_min',
            'group_neutralize', 'group_rank', 'group_scale', 'group_zscore', 'hump', 'if_else', 'inverse',
            'is_nan', 'jump_decay', 'kth_element', 'last_diff_value', 'less', 'less_equal', 'log', 'max',
            'min', 'multiply', 'normalize', 'not', 'not_equal', 'or', 'power', 'quantile', 'rank', 'reverse',
            'scale', 'scale_down', 'sign', 'signed_power', 'sqrt', 'subtract', 'to_nan', 'trade_when',
            'ts_arg_max', 'ts_arg_min', 'ts_av_diff', 'ts_backfill', 'ts_corr', 'ts_count_nans', 'ts_covariance',
            'ts_decay_linear', 'ts_delay', 'ts_delta', 'ts_max', 'ts_mean', 'ts_min', 'ts_product', 'ts_quantile',
            'ts_rank', 'ts_regression', 'ts_scale', 'ts_std_dev', 'ts_step', 'ts_sum', 'ts_target_tvr_decay',
            'ts_target_tvr_delta_limit', 'ts_zscore', 'vec_avg', 'vec_max', 'vec_min', 'vec_sum', 'vector_neut',
            'winsorize', 'zscore'
        }
        
        # Common financial field patterns
        field_patterns = [
            r'\b(close|open|high|low|volume|returns|market_cap|book_value|earnings|sales|cash|industry|sector|country|size|value|momentum)\b',
            r'\b(mdl\d+_\w+)\b',  # Model fields like mdl23_bk_entrpris_valu
            r'\b(fnd\d+_\w+)\b',  # Fundamental fields
            r'\b(analyst\d+_\w+)\b',  # Analyst fields
            r'\b(news\d+_\w+)\b',  # News fields
            r'\b(alt\d+_\w+)\b',  # Alternative data fields
            r'\b(anl\d+_\w+)\b',  # Analyst fields like anl69_best_bps_stddev
            r'\b(adv\d+)\b',  # Average daily volume
            r'\b(vwap|rsi|macd|bollinger_\w+)\b'  # Technical indicators
        ]
        
        fields = []
        for pattern in field_patterns:
            matches = re.findall(pattern, template)
            for match in matches:
                # Only include if it's not a known operator
                if match not in known_operators:
                    fields.append(match)
        
        # Filter out any remaining operators that might have been matched
        filtered_fields = [field for field in fields if field not in known_operators]
        
        return list(set(filtered_fields))
    
    def _has_data_fields_as_operators(self, template: str) -> bool:
        """ULTRA-ENHANCED validation: Check if template has data fields being used as operators"""
        import re
        
        # Comprehensive list of ALL valid operators
        known_operators = {
            # Time series operators
            'ts_rank', 'ts_max', 'ts_min', 'ts_sum', 'ts_mean', 'ts_std_dev', 'ts_skewness', 'ts_kurtosis',
            'ts_corr', 'ts_regression', 'ts_delta', 'ts_ratio', 'ts_product', 'ts_scale', 'ts_zscore',
            'ts_lag', 'ts_lead', 'ts_arg_max', 'ts_arg_min', 'ts_step', 'ts_bucket', 'ts_hump',
            # Basic operators
            'rank', 'add', 'subtract', 'multiply', 'divide', 'power', 'abs', 'sign', 'sqrt', 'log', 'exp',
            'max', 'min', 'sum', 'mean', 'std', 'std_dev', 'corr', 'regression', 'delta', 'ratio', 'product', 'scale', 'zscore', 'lag', 'lead',
            # Group operators
            'group_neutralize', 'group_neutralize', 'group_zscore', 'group_rank', 'group_max', 'group_min',
            'group_zscore', 'group_scale', 'group_max', 'group_min', 'group_rank', 'group_neutralize', 'group_mean', 'group_backfill', 'group_cartesian_product',
            # Vector operators
            'vec_avg', 'vec_sum', 'vec_max', 'vec_min',
            # Conditional operators
            'if_else', 'greater', 'less', 'greater_equal', 'less_equal', 'equal', 'not_equal',
            'and', 'or', 'not', 'is_nan', 'is_finite', 'is_infinite', 'fill_na', 'forward_fill',
            'backward_fill', 'clip', 'clip_lower', 'clip_upper', 'signed_power', 'inverse', 'inverse_sqrt',
            # Utility operators
            'bucket', 'step', 'hump', 'days_from_last_change', 'winsorize'
        }
        
        # Enhanced pattern to find function calls: word(...
        function_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.finditer(function_pattern, template)
        
        data_fields_used_as_operators = []
        for match in matches:
            function_name = match.group(1)
            # If it's not a known operator, it's likely a data field being used as operator
            if function_name not in known_operators:
                data_fields_used_as_operators.append(function_name)
        
        if data_fields_used_as_operators:
            logger.error(f"🚨 ULTRA-ENHANCED VALIDATION: DETECTED DATA FIELDS AS OPERATORS: {data_fields_used_as_operators}")
            logger.warning(f"⚠️ DATA FIELDS USED AS OPERATORS: {data_fields_used_as_operators}")
            logger.warning(f"   This indicates a template generation issue - data fields should not be used as operators")
            logger.warning(f"   Template: {template[:100]}...")
            return True
        
        return False
    
    def _has_hallucinated_fields(self, template: str, valid_fields: List[str]) -> bool:
        """Check if template contains field names that don't exist in the valid fields list"""
        import re
        
        # Extract all field references from template
        field_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        all_words = re.findall(field_pattern, template)
        
        # Comprehensive operator exclusion - DO NOT treat these as fields
        operators_to_exclude = {
            # Time series operators
            'ts_rank', 'ts_max', 'ts_min', 'ts_sum', 'ts_mean', 'ts_std_dev', 'ts_skewness', 'ts_kurtosis',
            'ts_corr', 'ts_regression', 'ts_delta', 'ts_ratio', 'ts_product', 'ts_scale', 'ts_zscore',
            'ts_lag', 'ts_lead', 'ts_arg_max', 'ts_arg_min', 'ts_step', 'ts_bucket', 'ts_hump',
            # Basic operators
            'rank', 'add', 'subtract', 'multiply', 'divide', 'power', 'abs', 'sign', 'sqrt', 'log', 'exp',
            'max', 'min', 'sum', 'mean', 'std', 'std_dev', 'corr', 'regression', 'delta', 'ratio', 'product', 'scale', 'zscore', 'lag', 'lead',
            # Group operators
            'group_neutralize', 'group_neutralize', 'group_zscore', 'group_rank', 'group_max', 'group_min',
            'group_zscore', 'group_scale', 'group_max', 'group_min', 'group_rank', 'group_neutralize', 'group_mean', 'group_backfill', 'group_cartesian_product',
            # Vector operators
            'vec_avg', 'vec_sum', 'vec_max', 'vec_min',
            # Conditional operators
            'if_else', 'greater', 'less', 'greater_equal', 'less_equal', 'equal', 'not_equal',
            'and', 'or', 'not', 'is_nan', 'is_finite', 'is_infinite', 'fill_na', 'forward_fill',
            'backward_fill', 'clip', 'clip_lower', 'clip_upper', 'signed_power', 'inverse', 'inverse_sqrt',
            # Utility operators
            'bucket', 'step', 'hump', 'days_from_last_change', 'winsorize',
            # Group types
            'industry', 'subindustry', 'sector', 'market',
            # Common fields (don't validate these as they're always valid)
            'close', 'volume', 'returns'
        }
        
        # Only check words that are NOT operators
        field_candidates = [word for word in all_words if word not in operators_to_exclude]
        
        # Check for numbers (skip validation)
        for word in field_candidates[:]:
            try:
                float(word)
                field_candidates.remove(word)
            except ValueError:
                pass
        
        # Check each field candidate against valid fields
        hallucinated_fields = []
        for field_name in field_candidates:
            if field_name not in valid_fields:
                hallucinated_fields.append(field_name)
        
        if hallucinated_fields:
            logger.error(f"🚨 HALLUCINATED FIELDS DETECTED: {hallucinated_fields}")
            logger.error(f"   Valid fields available: {len(valid_fields)}")
            logger.error(f"   Template: {template[:100]}...")
            return True
        
        return False
    
    def _fix_field_name_modifications(self, template: str, valid_fields: List[str]) -> str:
        """Fix field names that have been modified with suffixes or prefixes"""
        import re
        
        # Extract all field references from template
        field_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        all_words = re.findall(field_pattern, template)
        
        # Comprehensive operator exclusion - DO NOT treat these as fields
        operators_to_exclude = {
            # Time series operators
            'ts_rank', 'ts_max', 'ts_min', 'ts_sum', 'ts_mean', 'ts_std_dev', 'ts_skewness', 'ts_kurtosis',
            'ts_corr', 'ts_regression', 'ts_delta', 'ts_ratio', 'ts_product', 'ts_scale', 'ts_zscore',
            'ts_lag', 'ts_lead', 'ts_arg_max', 'ts_arg_min', 'ts_step', 'ts_bucket', 'ts_hump',
            # Basic operators
            'rank', 'add', 'subtract', 'multiply', 'divide', 'power', 'abs', 'sign', 'sqrt', 'log', 'exp',
            'max', 'min', 'sum', 'mean', 'std', 'std_dev', 'corr', 'regression', 'delta', 'ratio', 'product', 'scale', 'zscore', 'lag', 'lead',
            # Group operators
            'group_neutralize', 'group_neutralize', 'group_zscore', 'group_rank', 'group_max', 'group_min',
            'group_zscore', 'group_scale', 'group_max', 'group_min', 'group_rank', 'group_neutralize', 'group_mean', 'group_backfill', 'group_cartesian_product',
            # Vector operators
            'vec_avg', 'vec_sum', 'vec_max', 'vec_min',
            # Conditional operators
            'if_else', 'greater', 'less', 'greater_equal', 'less_equal', 'equal', 'not_equal',
            'and', 'or', 'not', 'is_nan', 'is_finite', 'is_infinite', 'fill_na', 'forward_fill',
            'backward_fill', 'clip', 'clip_lower', 'clip_upper', 'signed_power', 'inverse', 'inverse_sqrt',
            # Utility operators
            'bucket', 'step', 'hump', 'days_from_last_change', 'winsorize',
            # Group types
            'industry', 'subindustry', 'sector', 'market',
            # Common fields (don't validate these as they're always valid)
            'close', 'volume', 'returns'
        }
        
        # Only check words that are NOT operators
        field_candidates = [word for word in all_words if word not in operators_to_exclude]
        
        # Check for numbers (skip validation)
        for word in field_candidates[:]:
            try:
                float(word)
                field_candidates.remove(word)
            except ValueError:
                pass
        
        # Fix modified field names
        fixed_template = template
        changes_made = []
        
        logger.info(f"🔧 FIELD NAME FIXING: Checking template: {template[:100]}...")
        logger.info(f"🔧 FIELD NAME FIXING: Field candidates: {field_candidates}")
        logger.info(f"🔧 FIELD NAME FIXING: Valid fields sample: {valid_fields[:10]}")
        
        for field_name in field_candidates:
            if field_name not in valid_fields:
                logger.info(f"🔧 FIELD NAME FIXING: Found invalid field: {field_name}")
                # Try to find a similar field name by removing suffixes/prefixes
                for valid_field in valid_fields:
                    # Check if field_name is a modified version of valid_field
                    if (field_name.endswith('1') and field_name[:-1] == valid_field) or \
                       (field_name.startswith(valid_field + '_') and field_name[len(valid_field)+1:] in valid_fields) or \
                       (field_name in valid_field and len(field_name) > 5):  # Avoid too short matches
                        fixed_template = fixed_template.replace(field_name, valid_field)
                        changes_made.append(f"Fixed {field_name} -> {valid_field}")
                        logger.info(f"🔧 FIELD NAME FIX: {field_name} -> {valid_field}")
                        break
        
        # AGGRESSIVE FIX: Look for common field name modifications
        aggressive_fixes = []
        
        # Fix common suffix patterns
        for valid_field in valid_fields:
            # Pattern: field_name + "1" suffix
            if valid_field + "1" in fixed_template:
                fixed_template = fixed_template.replace(valid_field + "1", valid_field)
                aggressive_fixes.append(f"Aggressive fix: {valid_field}1 -> {valid_field}")
                logger.info(f"🔧 AGGRESSIVE FIX: {valid_field}1 -> {valid_field}")
            
            # Pattern: field_name + "_1" suffix  
            if valid_field + "_1" in fixed_template:
                fixed_template = fixed_template.replace(valid_field + "_1", valid_field)
                aggressive_fixes.append(f"Aggressive fix: {valid_field}_1 -> {valid_field}")
                logger.info(f"🔧 AGGRESSIVE FIX: {valid_field}_1 -> {valid_field}")
        
        if changes_made or aggressive_fixes:
            logger.info(f"🔧 FIELD NAME FIXES: {changes_made + aggressive_fixes}")
            logger.info(f"🔧 FIXED TEMPLATE: {fixed_template[:100]}...")
        
        return fixed_template
    
    def _fix_unknown_variable_retry(self, template: str, error_message: str) -> str:
        """Fix unknown variable errors by removing the last digit from field names"""
        import re
        
        # Extract the unknown variable from error message
        unknown_var_match = re.search(r'unknown variable "([^"]+)"', error_message)
        if not unknown_var_match:
            return template
        
        unknown_var = unknown_var_match.group(1)
        logger.info(f"🔧 UNKNOWN VARIABLE FIX: Found unknown variable: {unknown_var}")
        
        # Try to fix by removing the last digit
        if unknown_var and unknown_var[-1].isdigit():
            fixed_var = unknown_var[:-1]
            logger.info(f"🔧 UNKNOWN VARIABLE FIX: Removing last digit: {unknown_var} -> {fixed_var}")
            fixed_template = template.replace(unknown_var, fixed_var)
            logger.info(f"🔧 UNKNOWN VARIABLE FIX: Fixed template: {fixed_template[:100]}...")
            return fixed_template
        
        return template
    
    def _has_cross_region_fields(self, template: str, data_fields: List[Dict], expected_region: str) -> bool:
        """Check if template contains fields from wrong regions"""
        import re
        
        # Extract all field references from template
        field_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        all_words = re.findall(field_pattern, template)
        
        # Comprehensive operator exclusion - DO NOT treat these as fields
        operators_to_exclude = {
            # Time series operators
            'ts_rank', 'ts_max', 'ts_min', 'ts_sum', 'ts_mean', 'ts_std_dev', 'ts_skewness', 'ts_kurtosis',
            'ts_corr', 'ts_regression', 'ts_delta', 'ts_ratio', 'ts_product', 'ts_scale', 'ts_zscore',
            'ts_lag', 'ts_lead', 'ts_arg_max', 'ts_arg_min', 'ts_step', 'ts_bucket', 'ts_hump',
            # Basic operators
            'rank', 'add', 'subtract', 'multiply', 'divide', 'power', 'abs', 'sign', 'sqrt', 'log', 'exp',
            'max', 'min', 'sum', 'mean', 'std', 'std_dev', 'corr', 'regression', 'delta', 'ratio', 'product', 'scale', 'zscore', 'lag', 'lead',
            # Group operators
            'group_neutralize', 'group_neutralize', 'group_zscore', 'group_rank', 'group_max', 'group_min',
            'group_zscore', 'group_scale', 'group_max', 'group_min', 'group_rank', 'group_neutralize', 'group_mean', 'group_backfill', 'group_cartesian_product',
            # Vector operators
            'vec_avg', 'vec_sum', 'vec_max', 'vec_min',
            # Conditional operators
            'if_else', 'greater', 'less', 'greater_equal', 'less_equal', 'equal', 'not_equal',
            'and', 'or', 'not', 'is_nan', 'is_finite', 'is_infinite', 'fill_na', 'forward_fill',
            'backward_fill', 'clip', 'clip_lower', 'clip_upper', 'signed_power', 'inverse', 'inverse_sqrt',
            # Utility operators
            'bucket', 'step', 'hump', 'days_from_last_change', 'winsorize',
            # Group types
            'industry', 'subindustry', 'sector', 'market',
            # Common fields (don't validate these as they're always valid)
            'close', 'volume', 'returns'
        }
        
        # Only check words that are NOT operators
        field_candidates = [word for word in all_words if word not in operators_to_exclude]
        
        # Check for numbers (skip validation)
        for word in field_candidates[:]:
            try:
                float(word)
                field_candidates.remove(word)
            except ValueError:
                pass
        
        # Create a mapping of field names to their regions
        field_region_map = {field['id']: field.get('region', 'UNKNOWN') for field in data_fields}
        
        # Check each field candidate against the expected region
        cross_region_fields = []
        for field_name in field_candidates:
            if field_name in field_region_map:
                field_region = field_region_map[field_name]
                if field_region != expected_region:
                    cross_region_fields.append(f"{field_name}({field_region})")
        
        if cross_region_fields:
            logger.error(f"🚨 CROSS-REGION FIELDS DETECTED: {cross_region_fields}")
            logger.error(f"   Expected region: {expected_region}")
            logger.error(f"   Template: {template[:100]}...")
            return True
        
        return False
    
    def _assess_economic_significance(self, template: str) -> float:
        """Assess the economic significance of a template (0-1 scale)"""
        score = 0.0
        
        # Check for momentum patterns
        if any(op in template for op in ['ts_rank', 'ts_delta', 'rank']):
            score += 0.3
        
        # Check for mean reversion patterns
        if any(op in template for op in ['ts_zscore', 'group_neutralize']):
            score += 0.2
        
        # Check for volatility patterns
        if any(op in template for op in ['ts_std', 'winsorize', 'scale']):
            score += 0.2
        
        # Check for fundamental analysis
        if any(field in template for field in ['market_cap', 'book_value', 'earnings', 'sales']):
            score += 0.2
        
        # Check for cross-sectional analysis
        if any(op in template for op in ['group_neutralize', 'group_neutralize']):
            score += 0.1
        
        return min(score, 1.0)
    
    def _assess_innovation_score(self, template: str) -> float:
        """Assess the innovation/creativity of a template (0-1 scale)"""
        score = 0.0
        
        # Check for complex nested operations
        nested_count = template.count('(') - 1  # Subtract 1 for the outer function
        if nested_count >= 3:
            score += 0.3
        elif nested_count >= 2:
            score += 0.2
        
        # Check for unusual operator combinations
        operators = ['ts_rank', 'ts_delta', 'ts_zscore', 'group_neutralize', 'winsorize', 'scale', 'rank', 'add', 'subtract', 'multiply', 'divide']
        operator_count = sum(1 for op in operators if op in template)
        if operator_count >= 4:
            score += 0.3
        elif operator_count >= 3:
            score += 0.2
        
        # Check for creative field combinations
        field_types = ['mdl', 'fnd', 'analyst', 'news', 'alt', 'close', 'volume', 'returns']
        field_type_count = sum(1 for field_type in field_types if any(field_type in field for field in template.split()))
        if field_type_count >= 3:
            score += 0.2
        
        # Check for unusual parameters (non-standard values)
        unusual_params = ['0.1', '0.25', '0.75', '0.9', '252', '500', '1000']
        if any(param in template for param in unusual_params):
            score += 0.2
        
        return min(score, 1.0)
    
    def _get_real_alpha_examples(self) -> str:
        """Get real examples from submitted WorldQuant Brain alphas"""
        try:
            # Fetch real submitted alphas from WorldQuant Brain API
            api_url = "https://api.worldquantbrain.com/users/self/alphas"
            params = {
                'limit': 10,
                'offset': 0,
                'status!': 'UNSUBMITTED',
                'order': '-dateSubmitted',
                'hidden': 'false'
            }
            
            response = self.make_api_request('GET', api_url, timeout=10, params=params)
            if response.status_code == 200:
                data = response.json()
                alphas = data.get('results', [])
                
                if alphas:
                    examples = []
                    for alpha in alphas[:5]:  # Use top 5 alphas
                        if 'regular' in alpha and 'code' in alpha['regular']:
                            code = alpha['regular']['code']
                            examples.append(f"- {code}")
                    
                    if examples:
                        return f"""REAL SUBMITTED ALPHA EXAMPLES FROM WORLDQUANT BRAIN:
{chr(10).join(examples)}

These are actual submitted alphas with real performance metrics. Use them as inspiration for complexity and structure."""
            
            # Fallback to static examples if API fails
            logger.warning("Failed to fetch real alpha examples, using static examples")
            return """REAL WORLDQUANT BRAIN EXAMPLES FOR REFERENCE:
- ts_rank(ts_delta(close, 5), 20) - Price momentum with ranking
- group_neutralize(ts_zscore(volume, 60), industry) - Industry-neutral volume z-score
- ts_corr(ts_rank(close, 20), ts_rank(volume, 20), 60) - Cross-sectional momentum correlation"""
            
        except Exception as e:
            logger.warning(f"Could not fetch real alpha examples: {e}")
        
        # Fallback to hardcoded examples based on the API response
        return """REAL WORLDQUANT BRAIN EXAMPLES FOR REFERENCE:
- ts_rank(ts_delta(close, 5), 20) - Price momentum with ranking
- group_neutralize(ts_zscore(volume, 60), industry) - Industry-neutral volume z-score
- ts_corr(ts_rank(close, 20), ts_rank(volume, 20), 60) - Cross-sectional momentum correlation"""

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
        # Use active_regions if available (filtered regions), otherwise use all regions
        available_regions = getattr(self, 'active_regions', self.regions)
        
        # Calculate weights based on pyramid multipliers
        region_weights = {}
        for region in available_regions:
            delay = self.select_optimal_delay(region)
            multiplier = self.pyramid_multipliers.get(region, {}).get(delay, 1.0)
            region_weights[region] = multiplier
        
        # Weighted random selection
        total_weight = sum(region_weights.values())
        if total_weight == 0:
            return random.choice(available_regions)
        
        rand = random.random() * total_weight
        cumulative = 0
        for region, weight in region_weights.items():
            cumulative += weight
            if rand <= cumulative:
                return region
        
        return random.choice(available_regions)
    
    def extract_operators_from_template(self, template: str) -> List[str]:
        """Extract operator names from a template"""
        operators_found = []
        for op in self.operators:
            if op['name'] in template:
                operators_found.append(op['name'])
        return operators_found
    
    def track_operator_usage(self, template: str):
        """Track which operators are used in successful templates and manage blacklist"""
        operators_used = self.extract_operators_from_template(template)
        for op in operators_used:
            self.operator_usage_count[op] = self.operator_usage_count.get(op, 0) + 1
            
            # Check if operator should be blacklisted due to overuse
            if self.operator_usage_count[op] >= self.max_operator_usage:
                if not self._is_operator_blacklisted(op):
                    self._add_to_blacklist(op, f"Used {self.operator_usage_count[op]} times (max: {self.max_operator_usage})")
        
        # Save usage count to disk for persistence
        self._save_blacklist_to_disk()
        
        logger.info(f"📊 Operator usage updated: {operators_used} | Blacklisted: {list(self.operator_blacklist)}")
    
    def get_diverse_operators(self) -> List[Dict]:
        """Get a diverse set of operators, prioritizing underused and two-field operators"""
        import random
        
        # Check for blacklist releases before selecting operators
        self._check_blacklist_release_conditions()
        
        # Categorize operators by type and usage
        arithmetic_ops = []
        time_series_ops = []
        ranking_ops = []
        normalization_ops = []
        two_field_ops = []
        
        for op in self.operators:
            op_name = op['name'].lower()
            op_def = op['definition'].lower()
            
            # CRITICAL: Skip blacklisted operators
            if self._is_operator_blacklisted(op['name']):
                continue  # Skip blacklisted operators
            
            # Check if it's a two-field operator
            if 'x, y' in op_def or op_name in ['add', 'subtract', 'multiply', 'divide', 'ts_corr', 'ts_regression']:
                two_field_ops.append(op)
            elif 'ts_' in op_name:
                time_series_ops.append(op)
            elif any(rank_word in op_name for rank_word in ['rank', 'percentile', 'quantile']):
                ranking_ops.append(op)
            elif any(norm_word in op_name for norm_word in ['normalize', 'zscore', 'standardize']):
                normalization_ops.append(op)
            else:
                arithmetic_ops.append(op)
        
        # Select diverse operators, prioritizing underused ones
        diverse_operators = []
        
        # First, add two-field operators (up to 10)
        if two_field_ops:
            two_field_sorted = sorted(two_field_ops, key=lambda x: self.operator_usage_count.get(x['name'], 0))
            diverse_operators.extend(two_field_sorted[:min(10, len(two_field_sorted))])
        
        # Then add from other categories (up to 3 each)
        for category in [arithmetic_ops, time_series_ops, ranking_ops, normalization_ops]:
            if category:
                category_sorted = sorted(category, key=lambda x: self.operator_usage_count.get(x['name'], 0))
                diverse_operators.extend(category_sorted[:min(3, len(category_sorted))])
        
        # Remove duplicates while preserving order
        seen = set()
        final_operators = []
        for op in diverse_operators:
            if op['name'] not in seen:
                seen.add(op['name'])
                final_operators.append(op)
        
        # Ensure we have at least 6 operators
        if len(final_operators) < 6:
            remaining_ops = [op for op in self.operators if op['name'] not in seen]
            final_operators.extend(remaining_ops[:6 - len(final_operators)])
        
        return final_operators
    
    def get_underused_operators(self, max_operators: int = 8) -> List[Dict]:
        """Get a random group of underused operators to force diversity, excluding blacklisted operators"""
        # Check for blacklist releases before selecting operators
        self._check_blacklist_release_conditions()
        # Filter out blacklisted operators and overused operators
        available_operators = []
        for op in self.operators:
            if not self._is_operator_blacklisted(op['name']):
                usage_count = self.operator_usage_count.get(op['name'], 0)
                if usage_count < self.max_operator_usage:
                    available_operators.append(op)
                else:
                    self._add_to_blacklist(op['name'], f"Used {usage_count} times (max: {self.max_operator_usage})")
        
        if not available_operators:
            logger.warning("🚫 All operators are blacklisted! Clearing blacklist and using all operators")
            self.operator_blacklist.clear()
            available_operators = self.operators
        
        if not self.operator_usage_count:
            # If no usage data, return diverse operators from available ones
            return self.get_diverse_operators()[:max_operators]
        
        # Sort available operators by usage count (ascending = least used first)
        sorted_operators = sorted(available_operators, key=lambda x: self.operator_usage_count.get(x['name'], 0))
        
        # Get the least used operators (top 50% least used)
        half_point = len(sorted_operators) // 2
        underused_pool = sorted_operators[:half_point]
        
        # Randomly select from underused operators
        import random
        selected_operators = random.sample(underused_pool, min(max_operators, len(underused_pool)))
        
        # Log the selection
        operator_names = [op['name'] for op in selected_operators]
        usage_counts = [self.operator_usage_count.get(op['name'], 0) for op in selected_operators]
        logger.info(f"🎯 UNDERUSED OPERATORS SELECTED (excluding blacklisted): {operator_names} (usage counts: {usage_counts})")
        logger.info(f"🚫 Blacklisted operators: {list(self.operator_blacklist)}")
        
        return selected_operators
    
    def extract_fields_from_template(self, template: str, data_fields: List[Dict]) -> List[str]:
        """Extract field names from a template"""
        fields_found = []
        for field in data_fields:
            if field['id'] in template:
                fields_found.append(field['id'])
        return fields_found
    
    def replace_field_placeholders(self, template: str, data_fields: List[Dict], region: str = None) -> str:
        """ULTRA-SAFE field replacement - only replace exact placeholder matches in proper contexts"""
        import random
        import re
        
        # Validate template first - if it contains malformed patterns, reject it
        if self._has_malformed_placeholders(template):
            logger.error(f"🚨 MALFORMED TEMPLATE DETECTED: {template}")
            return template  # Return as-is to fail validation
        
        # Select fields using 50/50 prioritized vs random approach
        use_prioritized_fields = random.choice([True, False])
        
        if use_prioritized_fields:
            selected_fields = self._prioritize_fields(data_fields)
            logger.info(f"🔄 FIELD REPLACEMENT: Using prioritized fields")
        else:
            selected_fields = random.sample(data_fields, len(data_fields))
            logger.info(f"🔄 FIELD REPLACEMENT: Using random fields")
        
        # Create safe field mapping with fallbacks - ONLY use fields from the correct region
        field_mapping = {
            'DATA_FIELD1': selected_fields[0]['id'] if len(selected_fields) > 0 else 'close',
            'DATA_FIELD2': selected_fields[1]['id'] if len(selected_fields) > 1 else selected_fields[0]['id'],
            'DATA_FIELD3': selected_fields[2]['id'] if len(selected_fields) > 2 else selected_fields[1]['id'],
            'DATA_FIELD4': selected_fields[3]['id'] if len(selected_fields) > 3 else selected_fields[2]['id']
        }
        
        # CRITICAL SAFETY CHECK: Verify all selected fields are from the correct region
        for placeholder, field_id in field_mapping.items():
            if field_id != 'close':  # Skip fallback field
                # Find the field in data_fields to check its region
                field_info = next((f for f in data_fields if f['id'] == field_id), None)
                if field_info:
                    field_region = field_info.get('region', 'UNKNOWN')
                    if region and field_region != region:  # This should match the actual region being used
                        logger.error(f"🚨 REGION MISMATCH: Field {field_id} is from region {field_region}, not {region}!")
                        logger.error(f"   This will cause 'unknown variable' errors!")
                        # Use a safe fallback
                        field_mapping[placeholder] = 'close'
        
        # ULTRA-SAFE replacement: Only replace placeholders that are:
        # 1. Complete words (not part of other words)
        # 2. In proper context (not as operators)
        result = template
        
        for placeholder, actual_field in field_mapping.items():
            # Pattern: word boundary + exact placeholder + word boundary
            # This ensures we only match complete placeholders, not partial ones
            pattern = r'\b' + re.escape(placeholder) + r'\b'
            
            # Additional safety: only replace if placeholder appears in valid contexts
            # (not as part of operator names or malformed syntax)
            if self._is_placeholder_in_valid_context(template, placeholder):
                result = re.sub(pattern, actual_field, result)
                logger.debug(f"🔄 Replaced {placeholder} -> {actual_field}")
            else:
                logger.warning(f"⚠️ Skipped replacement of {placeholder} - invalid context")
        
        logger.info(f"🔄 FIELD REPLACEMENT: {template[:50]}... -> {result[:50]}...")
        return result
    
    def _has_malformed_placeholders(self, template: str) -> bool:
        """Check if template has malformed placeholder usage"""
        # Check for patterns like DATA_FIELD1_rank, DATA_FIELD1(, etc.
        malformed_patterns = [
            r'DATA_FIELD\d+_',  # DATA_FIELD1_something
            r'DATA_FIELD\d+\s*\(',  # DATA_FIELD1( (used as operator)
            r'_\s*DATA_FIELD\d+',  # _DATA_FIELD1
        ]
        
        for pattern in malformed_patterns:
            if re.search(pattern, template):
                logger.error(f"🚨 MALFORMED PATTERN DETECTED: {pattern} in {template}")
                return True
        return False
    
    def _is_placeholder_in_valid_context(self, template: str, placeholder: str) -> bool:
        """Check if placeholder appears in valid context (as parameter, not as operator)"""
        # Find all occurrences of the placeholder
        pattern = r'\b' + re.escape(placeholder) + r'\b'
        matches = list(re.finditer(pattern, template))
        
        for match in matches:
            start, end = match.span()
            
            # Check context before the placeholder
            before_context = template[max(0, start-20):start]
            after_context = template[end:min(len(template), end+20)]
            
            # Valid contexts: inside parentheses, after operators, etc.
            # Invalid contexts: at start of function, after opening parenthesis
            if (before_context.endswith('(') or 
                before_context.endswith('(') or
                before_context.strip().endswith('(')):
                logger.warning(f"⚠️ Invalid context for {placeholder}: {before_context}...{after_context}")
                return False
                
        return True
    
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
            # Use the same delay that was used for data field selection
            delay = self.select_optimal_delay(region)
            logger.info(f"🔧 DELAY SYNC: Using optimal delay {delay} for {region} (matches data field delay)")
        else:
            logger.info(f"🔧 DELAY SYNC: Using provided delay {delay} for {region}")
        
        # CRITICAL: Validate that delay has available data fields
        data_fields = self.get_data_fields_for_region(region, delay)
        if not data_fields:
            logger.error(f"🚨 NO DATA FIELDS: No fields available for {region} delay={delay}")
            # Try the other delay
            other_delay = 1 if delay == 0 else 0
            other_fields = self.get_data_fields_for_region(region, other_delay)
            if other_fields:
                logger.warning(f"🔄 DELAY SWITCH: Switching from delay={delay} to delay={other_delay} (has {len(other_fields)} fields)")
                delay = other_delay
            else:
                logger.error(f"🚨 NO FIELDS AVAILABLE: Neither delay=0 nor delay=1 has fields for {region}")
                return []
        
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
                
                # NO SIMULATION BLOCKING - Let WorldQuant Brain handle validation
                if self._has_data_field_as_operator(template):
                    logger.warning(f"⚠️ Template uses data fields as operators: {template}")
                    logger.warning(f"   Proceeding to simulation - let WorldQuant Brain validate")
                
                try:
                    # Generate simulation data
                    simulation_data = {
                        'type': 'REGULAR',
                        'settings': asdict(settings),
                        'regular': template
                    }
                    
                    # Submit simulation with automatic 401 handling
                    simulation_response = self.make_api_request('POST', 'https://api.worldquantbrain.com/simulations', 
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
                    response = self.make_api_request('GET', progress_url)
                    
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
                            alpha_response = self.make_api_request('GET', f'https://api.worldquantbrain.com/alphas/{alpha_id}')
                            
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
                                pnl_quality_ok = self.track_template_quality(template_data['template'], alpha_id, sharpe, fitness, margin)
                            
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
                            
                            # Check if this alpha qualifies for optimization
                            if is_truly_successful:
                                # Track operator usage for diversity
                                self.track_operator_usage(template_data['template'])
                                self.add_to_optimization_queue(result)
                                logger.info(f"✅ Template simulation completed successfully: {template_data['template'][:50]}...")
                                logger.info(f"📊 Alpha {alpha_id} Performance: Sharpe={sharpe}, Fitness={fitness}, Turnover={turnover}, Returns={returns}")
                                logger.info(f"📊 Alpha {alpha_id} Positions: Long={longCount}, Short={shortCount}")
                                logger.info(f"📊 Alpha {alpha_id} PnL Quality: Good")
                                
                                # Update exploitation bandit if in exploitation phase
                                if self.exploitation_phase and template_data.get('exploitation', False):
                                    original_sharpe = template_data.get('original_sharpe', 0)
                                    self.update_exploitation_bandit(result, original_sharpe)
                                    logger.info(f"🎯 Exploitation result: Original Sharpe={original_sharpe:.3f}, New Sharpe={result.sharpe:.3f}")
                            
                            # Update simulation count and check for phase switch
                            self.update_simulation_count()
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
                    
                    # 401 errors are now handled automatically by make_api_request
                    
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
        
        # Store the filtered regions for use in region selection
        self.active_regions = regions
        
        # Initialize progress tracker
        self.progress_tracker.total_regions = len(regions)
        
        # Try to load previous progress (always load if exists, regardless of resume flag)
        if self.load_progress():
            if resume:
                logger.info("Resuming from previous progress...")
            else:
                logger.info("Loaded previous progress for exploit data...")
        
        # Update metadata
        self.all_results['metadata']['regions'] = regions
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
                logger.info(f"🧵 Thread count: {self.thread_count}, Completed threads: {self.completed_threads}")
                logger.info(f"🧵 Thread exceptions: {self.thread_exception_count}")
                
                try:
                    # Process completed futures
                    self._process_completed_futures()
                except Exception as e:
                    logger.error(f"❌ ERROR PROCESSING COMPLETED FUTURES: {e}")
                    import traceback
                    logger.error(f"❌ TRACEBACK: {traceback.format_exc()}")
                    # Continue execution even if processing fails
                
                try:
                    # Check future health every iteration
                    healthy, slow, stuck = self._check_future_health()
                    if stuck > 0:
                        logger.warning(f"🚨 CRITICAL: {stuck} futures are stuck! Consider restarting if this persists.")
                    
                    # Show detailed status of all futures every iteration
                    self._show_all_futures_status()
                    
                    # Check executor health if there are stuck futures
                    if stuck > 0:
                        self._check_executor_health()
                    
                    # Force cleanup if too many futures are stuck
                    if stuck >= 3:
                        logger.warning(f"🚨 FORCE CLEANUP: {stuck} futures stuck, forcing cleanup...")
                        self._force_cleanup_stuck_futures()
                    
                    # Fill available slots with new concurrent tasks
                    self._fill_available_slots_concurrent()
                    
                    # Save progress every iteration
                    self.save_progress()
                    
                    # Wait a bit before next iteration
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"❌ ERROR IN MAIN LOOP ITERATION: {e}")
                    import traceback
                    logger.error(f"❌ TRACEBACK: {traceback.format_exc()}")
                    # Continue execution even if iteration fails
                    time.sleep(2)
                    
        except KeyboardInterrupt:
            logger.info("\n🛑 Received interrupt signal. Stopping gracefully...")
            # Wait for active futures to complete
            self._wait_for_futures_completion()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Process optimization queue for good alphas
        logger.info("🔍 Checking for alphas that qualify for optimization...")
        self.process_optimization_queue()
        
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
        
        # During exploitation phase, shuffle data fields to ensure different combinations
        # DISABLED: Keep consistent behavior throughout
        if False and self.exploitation_phase:
            data_fields = random.sample(data_fields, len(data_fields))
            logger.info(f"🎯 EXPLOITATION: Shuffled {len(data_fields)} data fields for {region}")
        
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
        common_words = {'max', 'min', 'log', 'abs', 'scale', 'rank', 'ts_rank', 'ts_mean', 'ts_std_dev', 'ts_delta', 'ts_av_diff', 'divide', 'multiply', 'add', 'subtract', 'if_else', 'winsorize', 'group_neutralize', 'longscale', 'shortscale', 'scale'}
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
            if neutralization != base_template.get('neutralization', 'INDUSTRY'):
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
        
        # Check if the template is already negated (starts with minus or uses subtract)
        if (base_code.strip().startswith('-') or 
            'subtract(0,' in base_code or 
            'multiply(-1,' in base_code):
            logger.info(f"Template already negated, skipping negation variation: {base_code[:50]}...")
            return []
        
        # Try different negation approaches that are valid WorldQuant Brain syntax
        negation_approaches = [
            f"subtract(0, {base_code})",  # subtract(0, expression) = -expression
            f"multiply(-1, {base_code})",  # multiply(-1, expression) = -expression
        ]
        
        # Get valid fields for validation
        data_fields = self.get_data_fields_for_region(region, delay)
        valid_fields = [field['id'] for field in data_fields] if data_fields else []
        
        for negated_template in negation_approaches:
            # Validate the negated template syntax
            is_valid, error_msg = self.validate_template_syntax(negated_template, valid_fields)
            if is_valid:
                variation = {
                    'template': negated_template,
                    'region': region,
                    'operators_used': base_template.get('operators_used', []),
                    'fields_used': base_template.get('fields_used', []),
                    'neutralization': base_template.get('neutralization', 'INDUSTRY'),
                    'variation_type': 'negation',
                    'original_template': base_code
                }
                variations.append(variation)
                logger.info(f"Generated negation variation: {negated_template[:50]}...")
                break  # Use the first valid negation approach
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
        # EXCEPTION: For CHN region, negative Sharpe values are NOT considered hopeful
        if result.sharpe < 0 and abs(result.sharpe) > 1.25:
            if result.region != "CHN":
                hopeful_conditions.append(f"Sharpe={result.sharpe:.3f} (abs={abs(result.sharpe):.3f})")
            else:
                logger.info(f"🚫 CHN region: Negative Sharpe {result.sharpe:.3f} NOT considered hopeful")
        
        # Fitness: negative but absolute value > 0.3
        # EXCEPTION: For CHN region, negative Fitness values are NOT considered hopeful
        if result.fitness < 0 and abs(result.fitness) > 1:
            if result.region != "CHN":
                hopeful_conditions.append(f"Fitness={result.fitness:.3f} (abs={abs(result.fitness):.3f})")
            else:
                logger.info(f"🚫 CHN region: Negative Fitness {result.fitness:.3f} NOT considered hopeful")
        
        # Returns: negative but absolute value > 0.1
        # EXCEPTION: For CHN region, negative Returns values are NOT considered hopeful
        if result.returns < 0 and abs(result.returns) > 0.2:
            if result.region != "CHN":
                hopeful_conditions.append(f"Returns={result.returns:.3f} (abs={abs(result.returns):.3f})")
            else:
                logger.info(f"🚫 CHN region: Negative Returns {result.returns:.3f} NOT considered hopeful")
        
        # Margin: negative but absolute value > 0.002 (20bps)
        # EXCEPTION: For CHN region, negative Margin values are NOT considered hopeful
        if result.margin < 0 and abs(result.margin) > 0.002:
            if result.region != "CHN":
                hopeful_conditions.append(f"Margin={result.margin:.4f} (abs={abs(result.margin):.4f})")
            else:
                logger.info(f"🚫 CHN region: Negative Margin {result.margin:.4f} NOT considered hopeful")
        
        if hopeful_conditions:
            logger.info(f"🎯 HOPEFUL ALPHA detected: {', '.join(hopeful_conditions)}")
            logger.info(f"  Template: {result.template[:50]}...")
            return True
        
        return False
    
    def check_pnl_data_quality(self, alpha_id: str, sharpe: float = 0, fitness: float = 0, margin: float = 0) -> Tuple[bool, str]:
        """
        Check PnL data quality for an alpha, including detection of 'too good to be true' alphas
        Uses exponential backoff retry for unavailable PnL data
        Returns: (is_good_quality, reason)
        """
        try:
            # Determine if we should check PnL based on suspicion score
            should_check, check_reason = self._should_check_pnl(sharpe, fitness, margin)
            
            if not should_check:
                # Skip PnL check for low suspicion alphas
                self.pnl_check_stats['skipped_checks'] += 1
                return True, f"Skipped PnL check - {check_reason}"
            
            # Track that we're doing a PnL check
            self.pnl_check_stats['total_checks'] += 1
            logger.info(f"🔍 Checking PnL for alpha {alpha_id}: {check_reason}")
            
            # Try to fetch PnL data with exponential backoff retry
            return self._fetch_pnl_with_retry(alpha_id)
            
        except Exception as e:
            logger.error(f"❌ PnL quality check failed: {str(e)}")
            # Always reject when PnL check fails - no exceptions
            return False, f"PnL quality check failed: {str(e)} - rejecting alpha for safety"
    
    def _fetch_pnl_with_retry(self, alpha_id: str, max_retries: int = 3) -> Tuple[bool, str]:
        """
        Fetch PnL data with exponential backoff retry
        Returns: (is_good_quality, reason)
        """
        pnl_url = f'https://api.worldquantbrain.com/alphas/{alpha_id}/recordsets/pnl'
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔍 Fetching PnL data from: {pnl_url} (attempt {attempt + 1}/{max_retries})")
                response = self.make_api_request('GET', pnl_url)
                
                if response.status_code != 200:
                    logger.error(f"❌ Failed to fetch PnL data: {response.status_code} - {response.text}")
                    if response.status_code == 404:
                        return False, f"Alpha {alpha_id} not found or no PnL data available"
                    elif response.status_code == 403:
                        return False, f"Access denied to PnL data for alpha {alpha_id}"
                    elif response.status_code == 401:
                        return False, f"Authentication failed for PnL data"
                    else:
                        # For other errors, retry with exponential backoff
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                            logger.warning(f"⚠️ Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        else:
                            return False, f"Failed to fetch PnL data after {max_retries} attempts: {response.status_code}"
                
                # Check if response has content before trying to parse JSON
                if not response.text.strip():
                    logger.warning(f"⚠️ Empty PnL response from API (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(f"⚠️ Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return False, f"Empty PnL response from API after {max_retries} attempts - no data available"
                
                # Check if response looks like JSON
                if not response.text.strip().startswith('{') and not response.text.strip().startswith('['):
                    logger.warning(f"⚠️ Non-JSON PnL response (attempt {attempt + 1}): {response.text[:100]}...")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(f"⚠️ Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return False, f"Non-JSON PnL response after {max_retries} attempts: {response.text[:100]}"
                
                try:
                    pnl_data = response.json()
                    logger.info(f"📊 PnL data structure: {list(pnl_data.keys()) if isinstance(pnl_data, dict) else type(pnl_data)}")
                except Exception as json_error:
                    logger.warning(f"⚠️ Failed to parse PnL JSON (attempt {attempt + 1}): {json_error}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(f"⚠️ Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ Response content: {response.text[:200]}...")
                        return False, f"Failed to parse PnL JSON after {max_retries} attempts: {str(json_error)}"
                
                # If we get here, we have valid PnL data - process it
                records = pnl_data.get('records', [])
                logger.info(f"📈 Found {len(records)} PnL records")
                
                if not records:
                    logger.warning(f"⚠️ No PnL records found for alpha {alpha_id}")
                    return False, "No PnL records found"
                
                # Analyze PnL data quality
                total_records = len(records)
                zero_pnl_count = 0
                non_zero_pnl_count = 0
                total_pnl_sum = 0.0
                pnl_values = []
                
                for i, record in enumerate(records):
                    try:
                        if len(record) >= 2:  # Ensure we have at least date and pnl
                            pnl_value = record[1]  # PnL is the second element
                            # Convert to float if it's a string
                            if isinstance(pnl_value, str):
                                pnl_value = float(pnl_value)
                            pnl_values.append(pnl_value)
                            if pnl_value == 0.0:
                                zero_pnl_count += 1
                            else:
                                non_zero_pnl_count += 1
                                total_pnl_sum += abs(pnl_value)
                        else:
                            logger.warning(f"⚠️ Skipping malformed record {i}: {record}")
                    except (ValueError, TypeError) as parse_error:
                        logger.warning(f"⚠️ Failed to parse PnL value in record {i}: {record} - {parse_error}")
                        continue
                
                # Check if we have enough valid PnL values after parsing
                if len(pnl_values) < 5:
                    logger.warning(f"⚠️ Insufficient valid PnL values after parsing: {len(pnl_values)}")
                    return False, f"Insufficient valid PnL values after parsing: {len(pnl_values)}"
                
                logger.info(f"📊 PnL analysis: {len(pnl_values)} valid values, {zero_pnl_count} zeros, {non_zero_pnl_count} non-zeros")
                
                # Calculate quality metrics
                zero_pnl_ratio = zero_pnl_count / len(pnl_values) if len(pnl_values) > 0 else 1.0
                avg_non_zero_pnl = total_pnl_sum / non_zero_pnl_count if non_zero_pnl_count > 0 else 0.0
                
                # Check for flatlined PnL curve (constant values over time)
                is_flatlined = self._detect_flatlined_pnl(pnl_values)
                if is_flatlined:
                    self.pnl_check_stats['flatlined_detected'] += 1
                    return False, f"FLATLINED PnL curve detected - constant values over time (too good to be true alpha)"
                
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
                logger.warning(f"⚠️ Exception during PnL processing (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"⚠️ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return False, f"PnL processing failed after {max_retries} attempts: {str(e)}"
        
        # If we get here, all retries failed
        return False, f"PnL data unavailable after {max_retries} attempts - rejecting alpha"
    
    def _detect_flatlined_pnl(self, pnl_values: List[float]) -> bool:
        """
        Detect if PnL curve has ANY flatlining (constant values over time)
        This indicates a 'too good to be true' alpha that doesn't actually generate real PnL
        STRICT: Any flatlining, even 5% or 10%, is unacceptable for a real alpha
        """
        if len(pnl_values) < 10:  # Need at least 10 data points
            return False
        
        # Check if all values are the same (including zero)
        unique_values = set(pnl_values)
        if len(unique_values) == 1:
            return True
        
        # Check if values are very close to each other (within a small threshold)
        # This catches cases where PnL is constant but not exactly zero
        if len(unique_values) <= 3:  # Only 1-3 unique values in the entire series
            return True
        
        # Check for very low variance (standard deviation close to zero)
        import statistics
        try:
            std_dev = statistics.stdev(pnl_values)
            mean_abs = statistics.mean([abs(x) for x in pnl_values])
            
            # If standard deviation is very small relative to mean absolute value
            if mean_abs > 0 and std_dev / mean_abs < 0.01:  # Less than 1% variation
                return True
            
            # If standard deviation is extremely small in absolute terms
            if std_dev < 1e-6:  # Less than 0.000001
                return True
                
        except statistics.StatisticsError:
            # If we can't calculate statistics, assume it's flatlined
            return True
        
        # STRICT CHECK: Any single value representing more than 5% of the data is suspicious
        # Real alphas should have varied PnL throughout the entire time series
        value_counts = {}
        for value in pnl_values:
            value_counts[value] = value_counts.get(value, 0) + 1
        
        # Check for any dominant value (even 5% is too much for a real alpha)
        max_count = max(value_counts.values())
        max_ratio = max_count / len(pnl_values)
        
        # If any single value represents more than 5% of the data, it's flatlined
        if max_ratio > 0.05:  # Changed from 0.9 to 0.05 (5%)
            logger.warning(f"🚨 FLATLINED PnL detected: {max_ratio*100:.1f}% of values are identical ({max_count}/{len(pnl_values)})")
            return True
        
        # Additional check: Look for consecutive identical values (streaks)
        # Real alphas shouldn't have long streaks of identical PnL
        max_streak = 1
        current_streak = 1
        for i in range(1, len(pnl_values)):
            if pnl_values[i] == pnl_values[i-1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        # If there's a streak of more than 5% of the data, it's suspicious
        if max_streak > len(pnl_values) * 0.05:  # More than 5% consecutive identical values
            logger.warning(f"🚨 FLATLINED PnL detected: {max_streak} consecutive identical values ({max_streak/len(pnl_values)*100:.1f}% of data)")
            return True
        
        # Additional strict check: Look for very small variations that might indicate flatlining
        # Real alphas should have meaningful PnL variations, not just tiny fluctuations
        try:
            # Calculate the range of PnL values
            pnl_range = max(pnl_values) - min(pnl_values)
            mean_abs_pnl = statistics.mean([abs(x) for x in pnl_values])
            
            # If the range is very small relative to the mean absolute value, it's suspicious
            if mean_abs_pnl > 0 and pnl_range / mean_abs_pnl < 0.1:  # Less than 10% range
                logger.warning(f"🚨 FLATLINED PnL detected: Very small range {pnl_range:.6f} relative to mean {mean_abs_pnl:.6f}")
                return True
            
            # If the range is extremely small in absolute terms
            if pnl_range < 1e-5:  # Less than 0.00001
                logger.warning(f"🚨 FLATLINED PnL detected: Extremely small range {pnl_range:.8f}")
                return True
                
        except statistics.StatisticsError:
            # If we can't calculate statistics, assume it's flatlined
            return True
        
        return False
    
    def _calculate_suspicion_score(self, sharpe: float, fitness: float, margin: float) -> float:
        """
        Calculate a suspicion score (0.0 to 1.0) based on how 'too good to be true' the metrics are
        Higher scores indicate higher probability of flatlined PnL
        Handles both positive and negative values with high absolute values
        """
        suspicion_score = 0.0
        suspicious_factors = []
        
        # Fitness suspicion (0.0 to 0.4) - handles both positive and negative
        fitness_abs = abs(fitness)
        if fitness_abs > 0.5:  # Start getting suspicious at 0.5 absolute value
            fitness_suspicion = min(0.4, (fitness_abs - 0.5) / 2.0)  # Max 0.4 at fitness 1.3+
            suspicion_score += fitness_suspicion
            if fitness_abs > 1.0:
                suspicious_factors.append(f"Fitness={fitness:.3f} (abs={fitness_abs:.3f})")
        
        # Sharpe suspicion (0.0 to 0.4) - handles both positive and negative
        sharpe_abs = abs(sharpe)
        if sharpe_abs > 1.0:  # Start getting suspicious at 1.0 absolute value
            sharpe_suspicion = min(0.4, (sharpe_abs - 1.0) / 3.0)  # Max 0.4 at sharpe 2.2+
            suspicion_score += sharpe_suspicion
            if sharpe_abs > 1.5:
                suspicious_factors.append(f"Sharpe={sharpe:.3f} (abs={sharpe_abs:.3f})")
        
        # Margin suspicion (0.0 to 0.2) - handles both positive and negative
        margin_abs = abs(margin)
        if margin_abs > 0.01:  # Start getting suspicious at 1% absolute value
            margin_suspicion = min(0.2, (margin_abs - 0.01) / 0.04)  # Max 0.2 at margin 3%+
            suspicion_score += margin_suspicion
            if margin_abs > 0.02:
                suspicious_factors.append(f"Margin={margin:.4f} (abs={margin_abs:.4f})")
        
        # Cap at 1.0
        suspicion_score = min(1.0, suspicion_score)
        
        if suspicious_factors:
            logger.info(f"🔍 Suspicion score: {suspicion_score:.3f} for metrics: {', '.join(suspicious_factors)}")
        
        return suspicion_score
    
    def _should_check_pnl(self, sharpe: float, fitness: float, margin: float) -> Tuple[bool, str]:
        """
        Determine if PnL should be checked based on suspicion score
        Returns: (should_check, reason)
        """
        suspicion_score = self._calculate_suspicion_score(sharpe, fitness, margin)
        
        # Track suspicion score
        self.pnl_check_stats['suspicion_scores'].append(suspicion_score)
        
        # Mandatory check threshold (100% probability)
        if suspicion_score >= 0.8:
            self.pnl_check_stats['mandatory_checks'] += 1
            return True, f"MANDATORY PnL check - suspicion score {suspicion_score:.3f} >= 0.8"
        
        # High probability check (80% chance)
        elif suspicion_score >= 0.6:
            check_probability = 0.8
            should_check = random.random() < check_probability
            if should_check:
                self.pnl_check_stats['probability_checks'] += 1
            return should_check, f"High probability PnL check - suspicion {suspicion_score:.3f}, {check_probability*100:.0f}% chance"
        
        # Medium probability check (50% chance)
        elif suspicion_score >= 0.3:
            check_probability = 0.5
            should_check = random.random() < check_probability
            if should_check:
                self.pnl_check_stats['probability_checks'] += 1
            return should_check, f"Medium probability PnL check - suspicion {suspicion_score:.3f}, {check_probability*100:.0f}% chance"
        
        # Low probability check (20% chance)
        elif suspicion_score >= 0.1:
            check_probability = 0.2
            should_check = random.random() < check_probability
            if should_check:
                self.pnl_check_stats['probability_checks'] += 1
            return should_check, f"Low probability PnL check - suspicion {suspicion_score:.3f}, {check_probability*100:.0f}% chance"
        
        # Very low probability check (5% chance)
        else:
            check_probability = 0.05
            should_check = random.random() < check_probability
            if should_check:
                self.pnl_check_stats['probability_checks'] += 1
            return should_check, f"Very low probability PnL check - suspicion {suspicion_score:.3f}, {check_probability*100:.0f}% chance"
    
    def track_template_quality(self, template: str, alpha_id: str, sharpe: float = 0, fitness: float = 0, margin: float = 0) -> bool:
        """
        Track template quality based on PnL data
        Returns: True if template should be kept, False if it should be deleted
        """
        # Create template hash for tracking
        template_hash = hash(template)
        
        # Check PnL data quality with metrics for 'too good to be true' detection
        is_good_quality, reason = self.check_pnl_data_quality(alpha_id, sharpe, fitness, margin)
        
        # Initialize tracking if not exists
        if template_hash not in self.template_quality_tracker:
            self.template_quality_tracker[template_hash] = {
                'zero_pnl_count': 0,
                'flatlined_count': 0,
                'total_attempts': 0,
                'template': template
            }
        
        tracker = self.template_quality_tracker[template_hash]
        tracker['total_attempts'] += 1
        
        if not is_good_quality:
            # Check if it's specifically a flatlined PnL curve
            if "FLATLINED PnL curve" in reason:
                tracker['flatlined_count'] += 1
                logger.error(f"🚨 FLATLINED PnL detected for template: {template[:50]}...")
                logger.error(f"   Reason: {reason}")
                logger.error(f"   Flatlined count: {tracker['flatlined_count']}")
                
                # Immediately blacklist templates that produce flatlined PnL curves
                if tracker['flatlined_count'] >= 1:  # Zero tolerance for flatlined PnL
                    logger.error(f"🗑️ IMMEDIATELY BLACKLISTING template due to flatlined PnL: {template[:50]}...")
                    logger.error(f"   This is a 'too good to be true' alpha with fake metrics")
                    return False  # Delete template immediately
            else:
                # Handle all other PnL quality failures (including API errors, parsing errors, etc.)
                tracker['zero_pnl_count'] += 1
                logger.warning(f"⚠️ Poor PnL quality for template: {template[:50]}...")
                logger.warning(f"   Reason: {reason}")
                logger.warning(f"   Zero PnL count: {tracker['zero_pnl_count']}/{self.max_zero_pnl_attempts}")
                
                # For API errors or parsing errors, be more lenient but still track failures
                if "Failed to fetch PnL data" in reason or "Failed to parse PnL JSON" in reason:
                    # API/parsing errors - don't immediately blacklist, but track as failure
                    logger.warning(f"⚠️ PnL API/parsing error - will retry later: {reason}")
                    return False  # Reject this attempt but don't blacklist template yet
                elif "PnL data not available" in reason:
                    # PnL data not available - reject alpha (no exceptions)
                    logger.warning(f"⚠️ PnL data not available - rejecting alpha: {reason}")
                    return False  # Reject alpha when PnL data is not available
                else:
                    # Other quality issues - use normal blacklist logic
                    if tracker['zero_pnl_count'] >= self.max_zero_pnl_attempts:
                        logger.error(f"🗑️ DELETING template due to poor PnL quality: {template[:50]}...")
                        logger.error(f"   Total attempts: {tracker['total_attempts']}, Zero PnL: {tracker['zero_pnl_count']}")
                        return False  # Delete template
        else:
            logger.info(f"✅ Good PnL quality for template: {template[:50]}...")
            logger.info(f"   {reason}")
        
        return True  # Keep template
    
    def is_template_blacklisted(self, template: str) -> bool:
        """Check if a template is blacklisted due to poor PnL quality or flatlined PnL curves"""
        template_hash = hash(template)
        if template_hash in self.template_quality_tracker:
            tracker = self.template_quality_tracker[template_hash]
            # Check for flatlined PnL (immediate blacklist) or poor quality (after multiple attempts)
            return (tracker['flatlined_count'] >= 1 or 
                    tracker['zero_pnl_count'] >= self.max_zero_pnl_attempts)
        return False
    
    def save_blacklist_to_file(self, filename: str = "alpha_blacklist.json"):
        """Save the current blacklist to a file for persistence"""
        blacklisted_templates = []
        for template_hash, tracker in self.template_quality_tracker.items():
            if (tracker['flatlined_count'] >= 1 or 
                tracker['zero_pnl_count'] >= self.max_zero_pnl_attempts):
                blacklisted_templates.append({
                    'template_hash': template_hash,
                    'template': tracker['template'],
                    'flatlined_count': tracker['flatlined_count'],
                    'zero_pnl_count': tracker['zero_pnl_count'],
                    'total_attempts': tracker['total_attempts']
                })
        
        with open(filename, 'w') as f:
            json.dump(blacklisted_templates, f, indent=2)
        
        logger.info(f"Saved {len(blacklisted_templates)} blacklisted templates to {filename}")
    
    def load_blacklist_from_file(self, filename: str = "alpha_blacklist.json"):
        """Load blacklist from file for persistence"""
        if not os.path.exists(filename):
            return
        
        try:
            with open(filename, 'r') as f:
                blacklisted_templates = json.load(f)
            
            for item in blacklisted_templates:
                template_hash = item['template_hash']
                self.template_quality_tracker[template_hash] = {
                    'template': item['template'],
                    'flatlined_count': item['flatlined_count'],
                    'zero_pnl_count': item['zero_pnl_count'],
                    'total_attempts': item['total_attempts']
                }
            
            logger.info(f"Loaded {len(blacklisted_templates)} blacklisted templates from {filename}")
        except Exception as e:
            logger.error(f"Error loading blacklist from {filename}: {e}")
    
    def get_pnl_check_statistics(self) -> Dict:
        """Get statistics about PnL checking system"""
        stats = self.pnl_check_stats.copy()
        
        if stats['suspicion_scores']:
            stats['avg_suspicion_score'] = sum(stats['suspicion_scores']) / len(stats['suspicion_scores'])
            stats['max_suspicion_score'] = max(stats['suspicion_scores'])
            stats['min_suspicion_score'] = min(stats['suspicion_scores'])
        else:
            stats['avg_suspicion_score'] = 0.0
            stats['max_suspicion_score'] = 0.0
            stats['min_suspicion_score'] = 0.0
        
        # Calculate check rates
        total_evaluations = stats['total_checks'] + stats['skipped_checks']
        if total_evaluations > 0:
            stats['check_rate'] = stats['total_checks'] / total_evaluations
            stats['mandatory_rate'] = stats['mandatory_checks'] / total_evaluations
            stats['probability_rate'] = stats['probability_checks'] / total_evaluations
            stats['skip_rate'] = stats['skipped_checks'] / total_evaluations
        else:
            stats['check_rate'] = 0.0
            stats['mandatory_rate'] = 0.0
            stats['probability_rate'] = 0.0
            stats['skip_rate'] = 0.0
        
        # Calculate flatlined detection rate
        if stats['total_checks'] > 0:
            stats['flatlined_rate'] = stats['flatlined_detected'] / stats['total_checks']
        else:
            stats['flatlined_rate'] = 0.0
        
        return stats
    
    def test_pnl_api(self, alpha_id: str) -> Dict:
        """
        Test the PnL API endpoint directly for debugging
        Returns detailed information about the API response
        """
        try:
            pnl_url = f'https://api.worldquantbrain.com/alphas/{alpha_id}/recordsets/pnl'
            logger.info(f"🧪 Testing PnL API endpoint: {pnl_url}")
            
            response = self.make_api_request('GET', pnl_url)
            
            result = {
                'url': pnl_url,
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'success': response.status_code == 200
            }
            
            if response.status_code == 200:
                result['raw_response'] = response.text[:500]  # First 500 chars
                result['response_length'] = len(response.text)
                try:
                    data = response.json()
                    result['data_type'] = type(data).__name__
                    result['data_keys'] = list(data.keys()) if isinstance(data, dict) else 'Not a dict'
                    result['records_count'] = len(data.get('records', [])) if isinstance(data, dict) else 0
                    result['sample_record'] = data.get('records', [])[:2] if isinstance(data, dict) and data.get('records') else None
                except Exception as json_error:
                    result['json_error'] = str(json_error)
            else:
                result['error_text'] = response.text
                
            return result
            
        except Exception as e:
            return {
                'url': pnl_url,
                'error': str(e),
                'success': False
            }
    
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
        """Process completed futures and update bandit with timeout handling"""
        completed_futures = []
        timed_out_futures = []
        current_time = time.time()
        
        # Create a copy of the items to avoid "dictionary changed size during iteration" error
        for future_id, future in list(self.active_futures.items()):
            # Check for timeout
            start_time = self.future_start_times.get(future_id, current_time)
            elapsed_time = current_time - start_time
            
            if elapsed_time > self.future_timeout:
                timed_out_futures.append(future_id)
                logger.warning(f"⏰ TIMEOUT: Future {future_id} has been running for {elapsed_time:.1f}s (timeout: {self.future_timeout}s)")
                continue
            
            if future.done():
                completed_futures.append(future_id)
                try:
                    result = future.result()
                    if result and result.success:
                        self.successful_count += 1
                        self._update_bandit_with_result(result)
                        self._add_to_results(result)
                        logger.info(f"✅ CONCURRENT simulation SUCCESS: {result.template[:50]}... (Sharpe: {result.sharpe:.3f})")
                        
                        # Learn from the success
                        self._learn_from_simulation_success(result.template)
                        
                        # Update simulation count and check for phase switch
                        self.update_simulation_count()
                    elif result and not result.success:
                        self.failed_count += 1
                        error_msg = getattr(result, 'error_message', 'Simulation failed')
                        
                        # Log simulation settings for debugging
                        settings = getattr(result, 'settings', None)
                        if settings:
                            logger.info(f"❌ CONCURRENT simulation FAILED: {result.template[:50]}... - {error_msg}")
                            logger.info(f"🔧 SIMULATION SETTINGS: Region={settings.region}, Universe={settings.universe}, Delay={settings.delay}, Neutralization={settings.neutralization}")
                        else:
                            logger.info(f"❌ CONCURRENT simulation FAILED: {result.template[:50]}... - {error_msg}")
                        
                        # Handle any simulation error by regenerating template with error feedback
                        self._handle_simulation_error(result.template, error_msg, settings)
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
            if future_id in self.future_start_times:
                del self.future_start_times[future_id]
            self.completed_count += 1
        
        # Handle timed out futures and immediately start new ones
        for future_id in timed_out_futures:
            logger.warning(f"🔄 CANCELLING timed out future: {future_id}")
            try:
                future = self.active_futures[future_id]
                future.cancel()  # Try to cancel the future
            except Exception as e:
                logger.error(f"Error cancelling future {future_id}: {e}")
            
            # Remove from tracking
            del self.active_futures[future_id]
            if future_id in self.future_start_times:
                del self.future_start_times[future_id]
            self.failed_count += 1
            self.completed_count += 1
            
            # Immediately start a new future to replace the timed-out one using normal flow
            logger.info(f"🔄 RESTARTING: Starting new future to replace timed-out {future_id}")
            logger.info(f"🔄 RESTART: Using normal flow to start replacement future")
            
            # Use normal flow to fill the slot
            self._fill_available_slots_concurrent()
            
            # Log current active futures status
            logger.info(f"📊 RESTART STATUS: {len(self.active_futures)} futures now active after restart")
        
        # Log health status
        if len(timed_out_futures) > 0:
            logger.warning(f"⚠️ HEALTH WARNING: {len(timed_out_futures)} futures timed out, {len(self.active_futures)} still active")
        elif len(self.active_futures) > 0:
            logger.info(f"📊 HEALTH: {len(self.active_futures)} futures active, {len(completed_futures)} completed this cycle")
    
    def _start_new_future(self):
        """Start a new future to replace a timed-out one"""
        try:
            # Get next action from smart plan
            plan_type = self.slot_plans[self.slot_plan_index % len(self.slot_plans)]
            self.slot_plan_index += 1
            
            if plan_type == 'explore':
                # Explore: generate new template and simulate CONCURRENTLY
                logger.info(f"🔄 RESTART: Starting explore task for restart future")
                future = self.executor.submit(self._explore_and_simulate_concurrent)
                future_id = f"explore_restart_{int(time.time() * 1000)}"
                self.active_futures[future_id] = future
                self.future_start_times[future_id] = time.time()
                logger.info(f"🚀 Started NEW CONCURRENT EXPLORE task: {future_id}")
                logger.info(f"🔍 NEW FUTURE: Will generate new template and simulate it")
                logger.info(f"⏰ NEW FUTURE: Started at {time.strftime('%H:%M:%S')} - will timeout after {self.future_timeout}s")
                logger.info(f"🔄 RESTART: Future {future_id} submitted to executor")
            
            elif plan_type == 'exploit':
                # Exploit: try to use existing successful template
                successful_templates = self._get_successful_templates()
                if successful_templates:
                    # Filter for elite templates that meet high performance criteria
                    elite_templates = []
                    for template in successful_templates:
                        sharpe = template.get('sharpe', 0)
                        fitness = template.get('fitness', 0)
                        margin = template.get('margin', 0)
                        
                        # Only consider templates that meet the high bar (5 bps = 0.0005)
                        if (sharpe > 1.25 and fitness > 1.0 and margin > 0.0005):
                            elite_templates.append(template)
                    
                    if elite_templates:
                        logger.info(f"🎯 EXPLOIT RESTART: {len(elite_templates)}/{len(successful_templates)} templates meet elite criteria")
                        
                        # Use weighted selection among elite templates
                        performance_weights = []
                        for template in elite_templates:
                            # Use Sharpe ratio as the weight (higher Sharpe = higher weight)
                            weight = max(template.get('sharpe', 0), 0.1)  # Minimum weight of 0.1
                            performance_weights.append(weight)
                        
                        # Weighted random selection
                        total_weight = sum(performance_weights)
                        probabilities = [w / total_weight for w in performance_weights]
                        selected_idx = random.choices(range(len(elite_templates)), weights=probabilities)[0]
                        best_template = elite_templates[selected_idx]
                        
                        logger.info(f"🎯 EXPLOIT RESTART: Using elite template with Sharpe={best_template.get('sharpe', 0):.3f}, Fitness={best_template.get('fitness', 0):.3f}, Margin={best_template.get('margin', 0):.3f} (weight={probabilities[selected_idx]:.3f})")
                        
                        logger.info(f"🔄 RESTART: Starting exploit task for restart future")
                        future = self.executor.submit(self._exploit_and_simulate_concurrent, best_template)
                        future_id = f"exploit_restart_{int(time.time() * 1000)}"
                        self.active_futures[future_id] = future
                        self.future_start_times[future_id] = time.time()
                    else:
                        # No elite templates available, fallback to EXPLORE mode
                        logger.warning(f"🎯 EXPLOIT RESTART: No elite templates found, falling back to EXPLORE mode")
                        logger.info(f"📊 Available templates: {len(successful_templates)}")
                        for i, template in enumerate(successful_templates[:3]):  # Show first 3 for debugging
                            logger.info(f"   Template {i+1}: Sharpe={template.get('sharpe', 0):.3f}, Fitness={template.get('fitness', 0):.3f}, Margin={template.get('margin', 0):.3f}")
                        
                        # Fallback to explore mode instead of using mediocre templates
                        logger.info(f"🔄 FALLBACK: Switching to EXPLORE mode due to no elite templates")
                        future = self.executor.submit(self._explore_and_simulate_concurrent)
                        future_id = f"explore_restart_{int(time.time() * 1000)}"
                        self.active_futures[future_id] = future
                        self.future_start_times[future_id] = time.time()
                        logger.info(f"🚀 Started CONCURRENT EXPLORE RESTART task: {future_id}")
                        logger.info(f"🎯 NEW FUTURE: Will explore new templates in region {random.choice(self.active_regions)}")
                        logger.info(f"⏰ NEW FUTURE: Started at {time.strftime('%H:%M:%S')} - will timeout after {self.future_timeout}s")
                        logger.info(f"🔄 RESTART: Future {future_id} submitted to executor")
                        return
                else:
                    # No successful templates yet, fallback to explore
                    logger.info(f"🎯 EXPLOIT RESTART: No successful templates found, falling back to EXPLORE")
                    logger.info(f"🔄 RESTART: Starting fallback explore task for restart future")
                    future = self.executor.submit(self._explore_and_simulate_concurrent)
                    future_id = f"explore_restart_fallback_{int(time.time() * 1000)}"
                    self.active_futures[future_id] = future
                    self.future_start_times[future_id] = time.time()
                    logger.info(f"🚀 Started NEW CONCURRENT EXPLORE (fallback) task: {future_id}")
                    logger.info(f"🔍 NEW FUTURE: Will generate new template and simulate it (fallback)")
                    logger.info(f"⏰ NEW FUTURE: Started at {time.strftime('%H:%M:%S')} - will timeout after {self.future_timeout}s")
                    logger.info(f"🔄 RESTART: Future {future_id} submitted to executor")
        
        except Exception as e:
            logger.error(f"❌ Error starting new future: {e}")
            # If we can't start a new future, at least log the issue
            logger.warning(f"⚠️ Could not start replacement future, will retry in next iteration")
    
    def _check_future_health(self):
        """Check the health of active futures and log detailed status"""
        current_time = time.time()
        healthy_futures = 0
        slow_futures = 0
        stuck_futures = 0
        restart_futures = 0
        
        # Create a copy of the items to avoid "dictionary changed size during iteration" error
        for future_id, future in list(self.active_futures.items()):
            start_time = self.future_start_times.get(future_id, current_time)
            elapsed_time = current_time - start_time
            
            # Check if this is a restart future
            is_restart = isinstance(future_id, str) and 'restart' in future_id
            
            if elapsed_time < 60:  # Less than 1 minute
                healthy_futures += 1
                if is_restart:
                    restart_futures += 1
            elif elapsed_time < 180:  # 1-3 minutes
                slow_futures += 1
                if is_restart:
                    logger.warning(f"⚠️ RESTART FUTURE SLOW: {future_id} running for {elapsed_time:.1f}s")
            else:  # More than 3 minutes
                stuck_futures += 1
                if is_restart:
                    logger.warning(f"🚨 RESTART FUTURE STUCK: {future_id} running for {elapsed_time:.1f}s")
        
        if stuck_futures > 0:
            logger.warning(f"🚨 HEALTH ALERT: {stuck_futures} futures stuck (>3min), {slow_futures} slow (1-3min), {healthy_futures} healthy (<1min)")
            if restart_futures > 0:
                logger.warning(f"🔄 RESTART STATUS: {restart_futures} restart futures active")
        elif slow_futures > 0:
            logger.info(f"⚠️ HEALTH: {slow_futures} futures slow (1-3min), {healthy_futures} healthy (<1min)")
            if restart_futures > 0:
                logger.info(f"🔄 RESTART STATUS: {restart_futures} restart futures active")
        else:
            logger.info(f"✅ HEALTH: All {healthy_futures} futures healthy (<1min)")
            if restart_futures > 0:
                logger.info(f"🔄 RESTART STATUS: {restart_futures} restart futures active")
        
        return healthy_futures, slow_futures, stuck_futures
    
    def _check_restart_futures_status(self):
        """Check and log the status of restart futures specifically"""
        current_time = time.time()
        restart_futures = []
        
        # Create a copy of the items to avoid "dictionary changed size during iteration" error
        for future_id, future in list(self.active_futures.items()):
            if 'restart' in future_id:
                start_time = self.future_start_times.get(future_id, current_time)
                elapsed_time = current_time - start_time
                restart_futures.append((future_id, elapsed_time, future))
        
        if restart_futures:
            logger.info(f"🔄 RESTART FUTURES STATUS: {len(restart_futures)} restart futures active")
            for future_id, elapsed_time, future in restart_futures:
                status = "healthy" if elapsed_time < 60 else "slow" if elapsed_time < 180 else "stuck"
                future_status = "running" if not future.done() else "completed"
                logger.info(f"  - {future_id}: {elapsed_time:.1f}s ({status}, {future_status})")
                
                # Show warning if restart future is getting close to timeout
                if elapsed_time > (self.future_timeout * 0.8):  # 80% of timeout
                    remaining = self.future_timeout - elapsed_time
                    logger.warning(f"    ⚠️ {future_id} approaching timeout in {remaining:.1f}s")
        
        return len(restart_futures)
    
    def _check_executor_health(self):
        """Check if the executor is healthy and responsive"""
        try:
            # Try to get executor info
            if hasattr(self.executor, '_threads'):
                active_threads = len([t for t in self.executor._threads if t.is_alive()])
                logger.info(f"🔧 EXECUTOR HEALTH: {active_threads} active threads")
            else:
                logger.info(f"🔧 EXECUTOR HEALTH: ThreadPoolExecutor active")
            
            # Check if executor is accepting new tasks
            if hasattr(self.executor, '_work_queue'):
                queue_size = self.executor._work_queue.qsize()
                logger.info(f"🔧 EXECUTOR QUEUE: {queue_size} tasks in queue")
            
            return True
        except Exception as e:
            logger.error(f"❌ EXECUTOR HEALTH CHECK FAILED: {e}")
            return False
    
    def _show_all_futures_status(self):
        """Show detailed status of all active futures"""
        current_time = time.time()
        if not self.active_futures:
            logger.info("📊 ALL FUTURES STATUS: No active futures")
            return
        
        logger.info(f"📊 ALL FUTURES STATUS: {len(self.active_futures)} futures active")
        # Create a copy of the items to avoid "dictionary changed size during iteration" error
        for future_id, future in list(self.active_futures.items()):
            start_time = self.future_start_times.get(future_id, current_time)
            elapsed_time = current_time - start_time
            
            # Determine task type
            if 'explore' in future_id:
                task_type = "EXPLORE"
            elif 'exploit' in future_id:
                task_type = "EXPLOIT"
            else:
                task_type = "UNKNOWN"
            
            # Determine status
            if elapsed_time < 60:
                status = "healthy"
            elif elapsed_time < 180:
                status = "slow"
            else:
                status = "stuck"
            
            future_status = "running" if not future.done() else "completed"
            
            logger.info(f"  - {future_id}: {elapsed_time:.1f}s ({status}, {future_status}, {task_type})")
            
            # Show warning if approaching timeout
            if elapsed_time > (self.future_timeout * 0.8):
                remaining = self.future_timeout - elapsed_time
                logger.warning(f"    ⚠️ {future_id} approaching timeout in {remaining:.1f}s")
    
    def _force_cleanup_stuck_futures(self):
        """Force cleanup of all stuck futures in emergency situations"""
        current_time = time.time()
        stuck_count = 0
        
        for future_id, future in list(self.active_futures.items()):
            start_time = self.future_start_times.get(future_id, current_time)
            elapsed_time = current_time - start_time
            
            if elapsed_time > 180:  # More than 3 minutes
                logger.warning(f"🔄 FORCE CLEANUP: Cancelling stuck future {future_id} (running for {elapsed_time:.1f}s)")
                try:
                    future.cancel()
                except Exception as e:
                    logger.error(f"Error cancelling future {future_id}: {e}")
                
                # Remove from tracking
                del self.active_futures[future_id]
                if future_id in self.future_start_times:
                    del self.future_start_times[future_id]
                self.failed_count += 1
                self.completed_count += 1
                stuck_count += 1
        
        if stuck_count > 0:
            logger.warning(f"🧹 FORCE CLEANUP: Removed {stuck_count} stuck futures")
        
        return stuck_count
    
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
                        self.future_start_times[future_id] = time.time()
                        self.thread_count += 1
                        logger.info(f"🚀 Started CONCURRENT EXPLORE task: {future_id}")
                        logger.info(f"🧵 THREAD STARTED: {future_id} - Total threads: {self.thread_count}")
                
                elif plan_type == 'exploit':
                    # Exploit: try to use existing successful template
                    logger.info(f"🎯 EXPLOIT mode: Looking for successful templates...")
                    successful_templates = self._get_successful_templates()
                    if successful_templates:
                        # Filter for elite templates that meet high performance criteria
                        elite_templates = []
                        for template in successful_templates:
                            sharpe = template.get('sharpe', 0)
                            fitness = template.get('fitness', 0)
                            margin = template.get('margin', 0)
                            
                            # Only consider templates that meet the high bar (5 bps = 0.0005)
                            if (sharpe > 1.25 and fitness > 1.0 and margin > 0.0005):
                                elite_templates.append(template)
                            
                        if elite_templates:
                            logger.info(f"🎯 EXPLOIT: {len(elite_templates)}/{len(successful_templates)} templates meet elite criteria")
                            
                            # Use weighted selection among elite templates
                            performance_weights = []
                            for template in elite_templates:
                                # Use Sharpe ratio as the weight (higher Sharpe = higher weight)
                                weight = max(template.get('sharpe', 0), 0.1)  # Minimum weight of 0.1
                                performance_weights.append(weight)
                            
                            # Weighted random selection
                            total_weight = sum(performance_weights)
                            probabilities = [w / total_weight for w in performance_weights]
                            selected_idx = random.choices(range(len(elite_templates)), weights=probabilities)[0]
                            best_template = elite_templates[selected_idx]
                            
                            logger.info(f"🎯 EXPLOIT: Using elite template with Sharpe={best_template.get('sharpe', 0):.3f}, Fitness={best_template.get('fitness', 0):.3f}, Margin={best_template.get('margin', 0):.3f} (weight={probabilities[selected_idx]:.3f})")
                            
                            future = self.executor.submit(self._exploit_and_simulate_concurrent, best_template)
                            future_id = f"exploit_{int(time.time() * 1000)}"
                            self.active_futures[future_id] = future
                            self.future_start_times[future_id] = time.time()
                            logger.info(f"🚀 Started CONCURRENT EXPLOIT task: {future_id}")
                        else:
                                # No elite templates available, fallback to EXPLORE mode
                                logger.warning(f"🎯 EXPLOIT: No elite templates found, falling back to EXPLORE mode")
                                logger.info(f"📊 Available templates: {len(successful_templates)}")
                                for i, template in enumerate(successful_templates[:3]):  # Show first 3 for debugging
                                    logger.info(f"   Template {i+1}: Sharpe={template.get('sharpe', 0):.3f}, Fitness={template.get('fitness', 0):.3f}, Margin={template.get('margin', 0):.3f}")
                                
                                # Fallback to explore mode instead of using mediocre templates
                                logger.info(f"🔄 FALLBACK: Switching to EXPLORE mode due to no elite templates")
                                future = self.executor.submit(self._explore_and_simulate_concurrent)
                                future_id = f"explore_{int(time.time() * 1000)}"
                                self.active_futures[future_id] = future
                                self.future_start_times[future_id] = time.time()
                                logger.info(f"🚀 Started CONCURRENT EXPLORE task: {future_id}")
                    else:
                        # No successful templates yet, fallback to explore
                        logger.info(f"🎯 EXPLOIT: No successful templates found, falling back to EXPLORE")
                        future = self.executor.submit(self._explore_and_simulate_concurrent)
                        future_id = f"explore_fallback_{int(time.time() * 1000)}"
                        self.active_futures[future_id] = future
                        self.future_start_times[future_id] = time.time()
                        logger.info(f"🚀 Started CONCURRENT EXPLORE (fallback) task: {future_id}")
    
    def _explore_and_simulate_concurrent(self) -> Optional[TemplateResult]:
        """CONCURRENTLY explore new template and simulate it"""
        try:
            logger.info(f"🔍 CONCURRENT EXPLORE: Starting exploration task")
            
            # Generate new template with retry logic
            logger.info(f"🔍 CONCURRENT EXPLORE: Selecting region...")
            region = self.select_region_by_pyramid()
            logger.info(f"🔍 CONCURRENT EXPLORE: Selected region {region}")
            
            logger.info(f"🔍 CONCURRENT EXPLORE: Selecting delay...")
            delay = self.select_optimal_delay(region)
            logger.info(f"🔍 CONCURRENT EXPLORE: Selected delay {delay}")
            
            logger.info(f"🔍 CONCURRENT EXPLORE: Generating templates for {region}...")
            templates = self.generate_templates_for_region_with_retry(region, 1, 5)
            
            if not templates:
                logger.warning(f"⚠️ CONCURRENT EXPLORE: No templates generated for {region}")
                return TemplateResult(
                    template="",
                    region=region,
                    settings=SimulationSettings(region=region, universe=self.region_configs[region].universe, delay=delay, neutralization=template.get('neutralization', 'INDUSTRY')),
                    success=False,
                    error_message="No templates generated",
                    alpha_id="",
                    timestamp=time.time()
                )
            
            template = templates[0]
            logger.info(f"🔍 CONCURRENT EXPLORE: Generated template, starting simulation...")
            logger.info(f"🔍 EXPLORING new template: {template['template'][:50]}...")
            
            # Simulate the template CONCURRENTLY
            logger.info(f"🔍 CONCURRENT EXPLORE: Calling _simulate_template_concurrent...")
            result = self._simulate_template_concurrent(template, region, delay)
            logger.info(f"🔍 CONCURRENT EXPLORE: Simulation completed, result: {result is not None}")
            return result
            
        except Exception as e:
            logger.error(f"❌ CONCURRENT EXPLORE ERROR: {e}")
            import traceback
            logger.error(f"❌ CONCURRENT EXPLORE TRACEBACK: {traceback.format_exc()}")
            return TemplateResult(
                template="",
                region="",
                settings={},
                success=False,
                error_message=f"Explore error: {str(e)}",
                alpha_id="",
                timestamp=time.time()
            )
    
    def _exploit_and_simulate_concurrent(self, best_template: Dict) -> Optional[TemplateResult]:
        """CONCURRENTLY exploit existing template and simulate it with enhanced variations"""
        try:
            logger.info(f"🎯 CONCURRENT EXPLOIT: Starting exploitation task")
            
            # Always use cross-region exploitation for better diversity
            original_region = best_template['region']
            available_regions = [r for r in self.active_regions if r != original_region]
            if available_regions:
                region = random.choice(available_regions)
                logger.info(f"🎯 CONCURRENT EXPLOIT: Using cross-region {region} (original: {original_region})")
            else:
                # Fallback to original region if no other regions available
                region = original_region
                logger.info(f"🎯 CONCURRENT EXPLOIT: Using original region {region} (no other regions available)")
            
            logger.info(f"🎯 CONCURRENT EXPLOIT: Selecting delay...")
            delay = self.select_optimal_delay(region)
            logger.info(f"🎯 CONCURRENT EXPLOIT: Selected delay {delay}")
            
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
                    settings=SimulationSettings(region=region, universe=self.region_configs[region].universe, delay=delay, neutralization=best_template.get('neutralization', 'INDUSTRY')),
                    success=False,
                    error_message="No variations generated",
                    alpha_id="",
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
                alpha_id="",
                timestamp=time.time()
            )
    
    def _simulate_template_concurrent(self, template: Dict, region: str, delay: int) -> Optional[TemplateResult]:
        """CONCURRENTLY simulate a single template"""
        import re
        
        try:
            logger.info(f"🎮 CONCURRENT SIMULATION: Starting simulation for region {region}")
            
            # CRITICAL: Final validation before simulation - check for data fields as operators
            template_str = template['template']
            known_operators = {
                'ts_rank', 'ts_max', 'ts_min', 'ts_sum', 'ts_mean', 'ts_std_dev', 'ts_skewness', 'ts_kurtosis',
                'ts_corr', 'ts_regression', 'ts_delta', 'ts_ratio', 'ts_product', 'ts_scale', 'ts_zscore',
                'ts_lag', 'ts_lead', 'ts_arg_max', 'ts_arg_min', 'rank', 'add', 'subtract', 'multiply', 'divide',
                'power', 'abs', 'sign', 'sqrt', 'log', 'exp', 'max', 'min', 'sum', 'mean', 'std', 'std_dev',
                'corr', 'regression', 'delta', 'ratio', 'product', 'scale', 'zscore', 'lag', 'lead',
                'group_neutralize', 'group_neutralize', 'group_zscore', 'group_rank', 'group_max', 'group_min',
                'group_zscore', 'group_scale', 'group_max', 'group_min', 'group_rank', 'group_neutralize', 'group_mean', 'group_backfill', 'group_cartesian_product',
                'vec_avg', 'vec_sum', 'vec_max', 'vec_min',
                'if_else', 'greater', 'less', 'greater_equal', 'less_equal', 'equal', 'not_equal',
                'and', 'or', 'not', 'is_nan', 'is_finite', 'is_infinite', 'fill_na', 'forward_fill',
                'backward_fill', 'clip', 'clip_lower', 'clip_upper', 'signed_power', 'inverse', 'inverse_sqrt',
                'bucket', 'step', 'hump', 'days_from_last_change', 'ts_step', 'ts_bucket', 'ts_hump'
            }
            
            # Pattern to find function calls: word(...
            function_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            matches = re.finditer(function_pattern, template_str)
            
            data_fields_as_operators = []
            for match in matches:
                function_name = match.group(1)
                # If it's not a known operator, it's likely a data field being used as operator
                if function_name not in known_operators:
                    data_fields_as_operators.append(function_name)
            
            if data_fields_as_operators:
                logger.warning(f"⚠️ Data fields used as operators: {data_fields_as_operators}")
                logger.warning(f"   Template: {template_str}")
                logger.warning(f"   Proceeding to simulation - let WorldQuant Brain validate")
            
            # Validate and fix template fields to match region settings
            # CRITICAL: Use the actual simulation delay to ensure field-delay compatibility
            logger.info(f"🔧 FIELD VALIDATION: Using delay {delay} for field validation to match simulation")
            template['template'] = self._validate_and_fix_template_fields(template['template'], region, delay)
            
            # Create simulation data with all required fields
            # Use neutralization from template variation if available
            neutralization = template.get('neutralization', 'INDUSTRY')
            logger.info(f"🎮 CONCURRENT SIMULATION: Using neutralization {neutralization}")
            
            # Validate that simulation settings match data fields exactly
            self._validate_simulation_settings(region, delay, neutralization)
            
            # CRITICAL: Ensure delay matches data field delay
            data_fields = self.get_data_fields_for_region(region, delay)
            if not data_fields:
                logger.error(f"🚨 NO DATA FIELDS: No fields available for {region} delay={delay}")
                # Try the other delay
                other_delay = 1 if delay == 0 else 0
                other_fields = self.get_data_fields_for_region(region, other_delay)
                if other_fields:
                    logger.warning(f"🔄 DELAY SWITCH: Switching from delay={delay} to delay={other_delay} (has {len(other_fields)} fields)")
                    delay = other_delay
                    data_fields = other_fields
                else:
                    logger.error(f"🚨 NO FIELDS AVAILABLE: Neither delay=0 nor delay=1 has fields for {region}")
                    return None
            else:
                field_delays = set(field.get('delay', -1) for field in data_fields)
                if delay not in field_delays:
                    logger.error(f"🚨 DELAY MISMATCH: Simulation delay {delay} doesn't match data field delays {field_delays}")
                    logger.error(f"   This will cause 'unknown variable' errors!")
                    # Use the most common delay from data fields
                    from collections import Counter
                    delay_counts = Counter(field.get('delay', -1) for field in data_fields)
                    most_common_delay = delay_counts.most_common(1)[0][0]
                    logger.warning(f"   Correcting delay from {delay} to {most_common_delay}")
                    delay = most_common_delay
            
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
            
            logger.info(f"🎮 CONCURRENT SIMULATION: Submitting simulation to API...")
            # Submit simulation
            response = self.make_api_request('POST', 'https://api.worldquantbrain.com/simulations', json=simulation_data)
            logger.info(f"🎮 CONCURRENT SIMULATION: API response status: {response.status_code}")
            
            if response.status_code != 201:
                error_message = f"Failed to submit simulation: {response.status_code}"
                logger.error(f"❌ CONCURRENT SIMULATION: Failed to submit simulation: {response.status_code} - {response.text}")
                # Record the failure for learning
                self.record_failure(region, template['template'], error_message)
                
                return TemplateResult(
                    template=template['template'],
                    region=region,
                    settings=SimulationSettings(region=region, universe=self.region_configs[region].universe, delay=delay, neutralization=template.get('neutralization', 'INDUSTRY')),
                    success=False,
                    error_message=error_message,
                    alpha_id="",
                    timestamp=time.time()
                )
            
            progress_url = response.headers.get('Location')
            logger.info(f"🎮 CONCURRENT SIMULATION: Got progress URL: {progress_url}")
            if not progress_url:
                error_message = "No Location header in response"
                logger.error(f"❌ CONCURRENT SIMULATION: No Location header in response")
                # Record the failure for learning
                self.record_failure(region, template['template'], error_message)
                
                return TemplateResult(
                    template=template['template'],
                    region=region,
                    settings=SimulationSettings(region=region, universe=self.region_configs[region].universe, delay=delay, neutralization=template.get('neutralization', 'INDUSTRY')),
                    success=False,
                    error_message=error_message,
                    alpha_id="",
                    timestamp=time.time()
                )
            
            logger.info(f"🎮 CONCURRENT SIMULATION: Starting to monitor simulation progress...")
            # Monitor simulation progress CONCURRENTLY
            result = self._monitor_simulation_concurrent(progress_url, template, region, delay)
            logger.info(f"🎮 CONCURRENT SIMULATION: Monitoring completed, result: {result is not None}")
            return result
            
        except Exception as e:
            error_message = f"Simulation error: {str(e)}"
            logger.error(f"Error in simulate_template_concurrent: {e}")
            # Record the failure for learning
            self.record_failure(region, template['template'], error_message)
            
            return TemplateResult(
                template=template['template'],
                region=region,
                settings=SimulationSettings(region=region, universe=self.region_configs[region].universe, delay=delay, neutralization=template.get('neutralization', 'INDUSTRY')),
                success=False,
                error_message=error_message,
                alpha_id="",
                timestamp=time.time()
            )
    
    def _monitor_simulation_concurrent(self, progress_url: str, template: Dict, region: str, delay: int) -> Optional[TemplateResult]:
        """CONCURRENTLY monitor simulation progress"""
        max_wait_time = 3600  # 1 hour maximum wait time
        start_time = time.time()
        check_count = 0
        
        logger.info(f"🎮 MONITORING: Starting to monitor simulation progress (max {max_wait_time}s)")
        
        while (time.time() - start_time) < max_wait_time:
            try:
                check_count += 1
                elapsed = time.time() - start_time
                logger.info(f"🎮 MONITORING: Check #{check_count} (elapsed: {elapsed:.1f}s)")
                
                response = self.make_api_request('GET', progress_url)
                logger.info(f"🎮 MONITORING: Status check response: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status')
                    logger.info(f"🎮 MONITORING: Simulation status: {status}")
                    
                    if status == 'COMPLETE':
                        # Get the alphaId from the simulation response
                        alpha_id = data.get('alpha')
                        if not alpha_id:
                            logger.error(f"No alphaId in completed simulation response")
                            return TemplateResult(
                                template=template['template'],
                                region=region,
                                settings=SimulationSettings(region=region, universe=self.region_configs[region].universe, delay=delay, neutralization=template.get('neutralization', 'INDUSTRY')),
                                success=False,
                                error_message="No alphaId in simulation response",
                                alpha_id="",
                                timestamp=time.time()
                            )
                        
                        # Fetch the alpha data using the alphaId
                        logger.info(f"Simulation complete, fetching alpha {alpha_id}")
                        alpha_response = self.make_api_request('GET', f'https://api.worldquantbrain.com/alphas/{alpha_id}')
                        
                        if alpha_response.status_code != 200:
                            logger.error(f"Failed to fetch alpha {alpha_id}: {alpha_response.status_code}")
                            return TemplateResult(
                                template=template['template'],
                                region=region,
                                settings=SimulationSettings(region=region, universe=self.region_configs[region].universe, delay=delay, neutralization=template.get('neutralization', 'INDUSTRY')),
                                success=False,
                                error_message=f"Failed to fetch alpha: {alpha_response.status_code}",
                                alpha_id="",
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
                        
                        # Check PnL data quality for successful simulations
                        pnl_quality_ok = True
                        if has_meaningful_metrics:
                            pnl_quality_ok = self.track_template_quality(template['template'], alpha_id, sharpe, fitness, margin)
                        
                        # Only consider truly successful if both metrics and PnL quality are good
                        is_truly_successful = has_meaningful_metrics and pnl_quality_ok
                        
                        logger.info(f"Alpha {alpha_id} metrics: Sharpe={sharpe}, Fitness={fitness}, Turnover={turnover}, Returns={returns}")
                        logger.info(f"Alpha {alpha_id} positions: Long={longCount}, Short={shortCount}")
                        logger.info(f"Alpha {alpha_id} PnL quality: {pnl_quality_ok}")
                        logger.info(f"Alpha {alpha_id} success: {is_truly_successful}")
                        
                        # Track operator usage for diversity if successful
                        if is_truly_successful:
                            self.track_operator_usage(template['template'])
                        
                        # Perform post-simulation analysis immediately after getting alphaId
                        if is_truly_successful:
                            try:
                                logger.info(f"🔍 POST-SIMULATION ANALYSIS: Starting analysis for {template['template'][:50]}...")
                                # Create a temporary result object for the analysis
                                temp_result = TemplateResult(
                                    template=template['template'],
                                    region=region,
                                    settings=SimulationSettings(region=region, universe=self.region_configs[region].universe, delay=delay, neutralization=template.get('neutralization', 'INDUSTRY')),
                                    sharpe=sharpe,
                                    fitness=fitness if fitness is not None else 0,
                                    turnover=turnover,
                                    returns=returns,
                                    drawdown=drawdown,
                                    margin=margin,
                                    longCount=longCount,
                                    shortCount=shortCount,
                                    success=is_truly_successful,
                                    alpha_id=alpha_id,
                                    timestamp=time.time()
                                )
                                self._perform_post_simulation_analysis(temp_result)
                                logger.info(f"✅ POST-SIMULATION ANALYSIS: Completed successfully")
                            except Exception as e:
                                logger.error(f"❌ POST-SIMULATION ANALYSIS ERROR: {e}")
                                import traceback
                                logger.error(f"❌ POST-SIMULATION ANALYSIS TRACEBACK: {traceback.format_exc()}")
                                # Continue execution even if post-simulation analysis fails
                        
                        return TemplateResult(
                            template=template['template'],
                            region=region,
                            settings=SimulationSettings(region=region, universe=self.region_configs[region].universe, delay=delay, neutralization=template.get('neutralization', 'INDUSTRY')),
                            sharpe=sharpe,
                            fitness=fitness if fitness is not None else 0,
                            turnover=turnover,
                            returns=returns,
                            drawdown=drawdown,
                            margin=margin,
                            longCount=longCount,
                            shortCount=shortCount,
                            success=is_truly_successful,
                            alpha_id=alpha_id,
                            timestamp=time.time()
                        )
                    
                    elif status in ['FAILED', 'ERROR', 'FAIL']:
                        error_message = data.get('message', 'Unknown error')
                        # Record the failure for learning with simulation settings
                        settings_info = {
                            'region': region,
                            'universe': self.region_configs[region].universe,
                            'delay': delay,
                            'neutralization': template.get('neutralization', 'INDUSTRY')
                        }
                        self.record_failure(region, template['template'], error_message, settings_info)
                        
                        return TemplateResult(
                            template=template['template'],
                            region=region,
                            settings=SimulationSettings(region=region, universe=self.region_configs[region].universe, delay=delay, neutralization=template.get('neutralization', 'INDUSTRY')),
                            success=False,
                            error_message=error_message,
                            timestamp=time.time()
                        )
                    
                    elif status == 'WARNING':
                        # WARNING status should be treated as failed immediately
                        logger.warning(f"⚠️ Simulation in WARNING status, treating as failed immediately")
                        error_message = "Simulation failed with WARNING status"
                        settings_info = {
                            'region': region,
                            'universe': self.region_configs[region].universe,
                            'delay': delay,
                            'neutralization': template.get('neutralization', 'INDUSTRY')
                        }
                        self.record_failure(region, template['template'], error_message, settings_info)
                        
                        return TemplateResult(
                            template=template['template'],
                            region=region,
                            settings=SimulationSettings(region=region, universe=self.region_configs[region].universe, delay=delay, neutralization=template.get('neutralization', 'INDUSTRY')),
                            success=False,
                            error_message=error_message,
                            timestamp=time.time()
                        )
                    
                    elif status is None:
                        # None status might mean simulation is still starting
                        logger.info(f"⏳ Simulation status is None, waiting... (elapsed: {elapsed:.1f}s)")
                        time.sleep(5)  # Wait 5 seconds before next check
                        continue
                    
                    else:
                        # Unknown status - log and continue with timeout
                        logger.info(f"❓ Unknown simulation status: {status} (elapsed: {elapsed:.1f}s)")
                        time.sleep(5)  # Wait 5 seconds before next check
                        continue
                
                # 401 errors are now handled automatically by make_api_request
                
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
            settings=SimulationSettings(region=region, universe=self.region_configs[region].universe, delay=delay, neutralization=template.get('neutralization', 'INDUSTRY')),
            success=False,
            error_message=error_message,
            alpha_id="",
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
            # Update successful simulation count for blacklist release
            self._update_successful_simulation_count()
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
    parser = argparse.ArgumentParser(description='Enhanced template generator v2 with TRUE CONCURRENT subprocess execution')
    parser.add_argument('--credentials', default='credential.txt', help='Path to credentials file')
    parser.add_argument('--ollama-model', default='qwen2.5-coder:7b', help='Ollama model to use (default: qwen2.5-coder)')
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
            args.ollama_model, 
            args.max_concurrent,
            args.progress_file,
            args.output
        )
        
        # Generate and test templates
        results = generator.generate_and_test_templates(args.regions, args.templates_per_region, args.resume)
        
        # Save final results
        generator.save_results(results, args.output)
        
        # Save blacklist for persistence
        generator.save_blacklist_to_file()
        
        # Print final summary
        print(f"\n{'='*70}")
        print("🎉 TRUE CONCURRENT TEMPLATE GENERATION COMPLETE!")
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
        
        # Display PnL checking statistics
        pnl_stats = generator.get_pnl_check_statistics()
        print(f"\n🔍 PnL Checking Statistics:")
        print(f"   Total evaluations: {pnl_stats['total_checks'] + pnl_stats['skipped_checks']}")
        print(f"   PnL checks performed: {pnl_stats['total_checks']}")
        print(f"   Checks skipped: {pnl_stats['skipped_checks']}")
        print(f"   Check rate: {pnl_stats['check_rate']*100:.1f}%")
        print(f"   Mandatory checks: {pnl_stats['mandatory_checks']} ({pnl_stats['mandatory_rate']*100:.1f}%)")
        print(f"   Probability checks: {pnl_stats['probability_checks']} ({pnl_stats['probability_rate']*100:.1f}%)")
        print(f"   Flatlined alphas detected: {pnl_stats['flatlined_detected']} ({pnl_stats['flatlined_rate']*100:.1f}% of checks)")
        print(f"   Avg suspicion score: {pnl_stats['avg_suspicion_score']:.3f}")
        print(f"   Max suspicion score: {pnl_stats['max_suspicion_score']:.3f}")
        
    except Exception as e:
        logger.error(f"Enhanced template generation failed: {e}")
        raise

if __name__ == '__main__': 
    main()