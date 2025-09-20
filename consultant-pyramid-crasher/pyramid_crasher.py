#!/usr/bin/env python3
"""
Pyramid Crasher - 3 Concurrent Simulations for Pyramid Cracking
- Runs alongside consultant-templates-api
- Uses 3 concurrent simulation strategies to crack each pyramid
- Integrates with existing template generation system
- Implements specialized pyramid-cracking algorithms
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
import queue
import signal
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pyramid_crasher.log')
    ]
)
logger = logging.getLogger(__name__)

class PyramidStrategy(Enum):
    """Pyramid cracking strategies"""
    AGGREGATE_BREAKER = "aggregate_breaker"  # Break through aggregation layers
    CORRELATION_HUNTER = "correlation_hunter"  # Find hidden correlations
    VOLATILITY_EXPLOITER = "volatility_exploiter"  # Exploit volatility patterns

@dataclass
class PyramidResult:
    """Result of a pyramid cracking simulation"""
    strategy: str
    region: str
    delay: int
    template: str
    success: bool
    sharpe: float
    fitness: float
    turnover: float
    pnl: float
    error_message: str = ""
    timestamp: float = 0.0
    pyramid_level: int = 0
    breakthrough_score: float = 0.0

@dataclass
class RegionConfig:
    """Configuration for a region"""
    name: str
    universe: str
    pyramid_multiplier: float
    max_trade: bool = False
    neutralization_options: List[str] = None

    def __post_init__(self):
        if self.neutralization_options is None:
            self.neutralization_options = ['INDUSTRY', 'SECTOR', 'SUBSECTOR']

class PyramidCrasher:
    """Main pyramid crasher with 3 concurrent simulation strategies"""
    
    def __init__(self, credentials_path: str, max_concurrent: int = 3, 
                 progress_file: str = "pyramid_progress.json", results_file: str = "pyramid_results.json"):
        """Initialize the pyramid crasher with 3 concurrent strategies"""
        self.sess = requests.Session()
        self.credentials_path = credentials_path
        self.max_concurrent = min(max_concurrent, 3)  # Limit to 3 concurrent strategies
        self.progress_file = progress_file
        self.results_file = results_file
        
        # 3 concurrent simulation strategies
        self.strategies = [
            PyramidStrategy.AGGREGATE_BREAKER,
            PyramidStrategy.CORRELATION_HUNTER, 
            PyramidStrategy.VOLATILITY_EXPLOITER
        ]
        
        # TRUE CONCURRENT execution using ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
        self.active_futures = {}  # Track active Future objects
        self.completed_count = 0
        self.successful_count = 0
        self.failed_count = 0
        
        # Results storage
        self.pyramid_results = []
        self.all_results = {
            'metadata': {
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'strategies': [s.value for s in self.strategies],
                'max_concurrent': self.max_concurrent,
                'version': '1.0'
            },
            'pyramid_results': {},
            'breakthrough_analysis': {}
        }
        
        # Region configurations with pyramid multipliers
        self.region_configs = {
            "USA": RegionConfig("USA", "TOP3000", 1.8),
            "GLB": RegionConfig("GLB", "TOP3000", 1.5), 
            "EUR": RegionConfig("EUR", "TOP2500", 1.7),
            "ASI": RegionConfig("ASI", "MINVOL1M", 1.5, max_trade=True),
            "CHN": RegionConfig("CHN", "TOP2000U", 1.8, max_trade=True)
        }
        
        # Define regions list
        self.regions = list(self.region_configs.keys())
        
        # Pyramid theme multipliers (delay=0, delay=1) for each region
        self.pyramid_multipliers = {
            "USA": {"0": 1.8, "1": 1.2},
            "GLB": {"0": 1.0, "1": 1.5},
            "EUR": {"0": 1.7, "1": 1.4},
            "ASI": {"0": 1.0, "1": 1.5},
            "CHN": {"0": 1.0, "1": 1.8}
        }
        
        # Load operators and data fields
        self.operators = self.load_operators()
        self.data_fields = {}
        
        # Breakthrough tracking
        self.breakthrough_threshold = 2.0  # Minimum Sharpe for breakthrough
        self.best_breakthrough = 0.0
        self.breakthrough_count = 0
        
        self.setup_auth()
        self.load_progress()
    
    def setup_auth(self):
        """Setup authentication for WorldQuant Brain"""
        try:
            with open(self.credentials_path, 'r') as f:
                content = f.read().strip()
                
                # Try JSON format first (array format)
                if content.startswith('[') and content.endswith(']'):
                    credentials = json.loads(content)
                    if len(credentials) != 2:
                        raise ValueError("JSON credentials must have exactly 2 elements")
                    username, password = credentials
                else:
                    # Try two-line format
                    lines = content.split('\n')
                    if len(lines) >= 2:
                        username = lines[0].strip()
                        password = lines[1].strip()
                    else:
                        raise ValueError("Credential file must have at least 2 lines")
            
            self.sess.auth = HTTPBasicAuth(username, password)
            logger.info("✅ Authentication setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup authentication: {e}")
            raise
    
    def load_operators(self) -> List[Dict]:
        """Load operators from WorldQuant Brain"""
        try:
            response = self.sess.get('https://platform.worldquantbrain.com/static/operators.json')
            response.raise_for_status()
            operators = response.json()
            logger.info(f"Loaded {len(operators)} operators")
            return operators
        except Exception as e:
            logger.error(f"Failed to load operators: {e}")
            return []
    
    def get_data_fields_for_region(self, region: str, delay: int) -> List[Dict]:
        """Get data fields for a specific region and delay"""
        cache_key = f"data_fields_cache_{region}_{delay}.json"
        
        if cache_key in self.data_fields:
            return self.data_fields[cache_key]
        
        try:
            if os.path.exists(cache_key):
                with open(cache_key, 'r') as f:
                    fields = json.load(f)
                    self.data_fields[cache_key] = fields
                    logger.info(f"Loaded {len(fields)} cached data fields for {region} delay={delay}")
                    return fields
            
            # Fetch from API
            response = self.sess.get(f'https://platform.worldquantbrain.com/static/data-fields-{region.lower()}-{delay}.json')
            response.raise_for_status()
            fields = response.json()
            
            # Cache the results
            with open(cache_key, 'w') as f:
                json.dump(fields, f)
            
            self.data_fields[cache_key] = fields
            logger.info(f"Fetched and cached {len(fields)} data fields for {region} delay={delay}")
            return fields
            
        except Exception as e:
            logger.error(f"Failed to get data fields for {region} delay={delay}: {e}")
            return []
    
    def select_region_by_pyramid(self) -> str:
        """Select region based on pyramid multipliers"""
        # Weight regions by their pyramid multipliers
        weights = []
        for region in self.regions:
            multipliers = self.pyramid_multipliers.get(region, {"0": 1.0, "1": 1.0})
            # Use the higher multiplier as weight
            max_mult = max(multipliers.values())
            weights.append(max_mult)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(self.regions)
        
        rand_val = random.uniform(0, total_weight)
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand_val <= cumulative:
                return self.regions[i]
        
        return random.choice(self.regions)
    
    def select_optimal_delay(self, region: str) -> int:
        """Select delay based on pyramid multipliers and region constraints"""
        multipliers = self.pyramid_multipliers.get(region, {"0": 1.0, "1": 1.0})
        
        # For ASI, CHN, and GLB, only delay=1 is available
        if region in ["ASI", "CHN", "GLB"]:
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
            return 0
        else:
            return 1
    
    def generate_pyramid_template(self, strategy: PyramidStrategy, region: str, delay: int) -> Dict:
        """Generate a pyramid-cracking template based on strategy"""
        data_fields = self.get_data_fields_for_region(region, delay)
        if not data_fields:
            return None
        
        # Select relevant operators for the strategy
        strategy_operators = self.get_strategy_operators(strategy)
        
        # Generate template based on strategy
        if strategy == PyramidStrategy.AGGREGATE_BREAKER:
            template = self._generate_aggregate_breaker_template(data_fields, strategy_operators)
        elif strategy == PyramidStrategy.CORRELATION_HUNTER:
            template = self._generate_correlation_hunter_template(data_fields, strategy_operators)
        elif strategy == PyramidStrategy.VOLATILITY_EXPLOITER:
            template = self._generate_volatility_exploiter_template(data_fields, strategy_operators)
        else:
            template = self._generate_generic_pyramid_template(data_fields, strategy_operators)
        
        return {
            'template': template,
            'region': region,
            'strategy': strategy.value,
            'operators_used': strategy_operators,
            'fields_used': self.extract_fields_from_template(template),
            'neutralization': 'INDUSTRY'
        }
    
    def get_strategy_operators(self, strategy: PyramidStrategy) -> List[str]:
        """Get operators relevant to each strategy"""
        if strategy == PyramidStrategy.AGGREGATE_BREAKER:
            return ['ts_rank', 'ts_mean', 'ts_std', 'ts_delta', 'rank', 'scale', 'winsorize']
        elif strategy == PyramidStrategy.CORRELATION_HUNTER:
            return ['ts_corr', 'ts_rank', 'ts_mean', 'ts_std', 'rank', 'scale', 'group_neutralize']
        elif strategy == PyramidStrategy.VOLATILITY_EXPLOITER:
            return ['ts_std', 'ts_delta', 'ts_rank', 'ts_mean', 'rank', 'scale', 'winsorize']
        else:
            return ['ts_rank', 'ts_mean', 'ts_std', 'rank', 'scale']
    
    def _generate_aggregate_breaker_template(self, data_fields: List[Dict], operators: List[str]) -> str:
        """Generate template that breaks through aggregation layers"""
        # Select 2-3 fields for aggregation breaking
        selected_fields = random.sample(data_fields, min(3, len(data_fields)))
        field_names = [field['id'] for field in selected_fields]
        
        # Create multi-layer aggregation breaking template
        base_field = field_names[0]
        if len(field_names) > 1:
            secondary_field = field_names[1]
        else:
            secondary_field = base_field
        
        # Template: rank(ts_rank(scale(ts_mean(field1, 20)), 5)) - ts_rank(scale(ts_mean(field2, 10)), 3)
        template = f"subtract(rank(ts_rank(scale(ts_mean({base_field}, 20)), 5)), ts_rank(scale(ts_mean({secondary_field}, 10)), 3))"
        
        return template
    
    def _generate_correlation_hunter_template(self, data_fields: List[Dict], operators: List[str]) -> str:
        """Generate template that hunts for hidden correlations"""
        # Select 2-3 fields for correlation analysis
        selected_fields = random.sample(data_fields, min(3, len(data_fields)))
        field_names = [field['id'] for field in selected_fields]
        
        base_field = field_names[0]
        if len(field_names) > 1:
            correlation_field = field_names[1]
        else:
            correlation_field = base_field
        
        # Template: ts_corr(field1, field2, 20) * rank(ts_rank(scale(field1), 5))
        template = f"multiply(ts_corr({base_field}, {correlation_field}, 20), rank(ts_rank(scale({base_field}), 5)))"
        
        return template
    
    def _generate_volatility_exploiter_template(self, data_fields: List[Dict], operators: List[str]) -> str:
        """Generate template that exploits volatility patterns"""
        # Select 2-3 fields for volatility analysis
        selected_fields = random.sample(data_fields, min(3, len(data_fields)))
        field_names = [field['id'] for field in selected_fields]
        
        base_field = field_names[0]
        if len(field_names) > 1:
            volatility_field = field_names[1]
        else:
            volatility_field = base_field
        
        # Template: rank(ts_std(field1, 20)) - rank(ts_std(field2, 10))
        template = f"subtract(rank(ts_std({base_field}, 20)), rank(ts_std({volatility_field}, 10)))"
        
        return template
    
    def _generate_generic_pyramid_template(self, data_fields: List[Dict], operators: List[str]) -> str:
        """Generate a generic pyramid-cracking template"""
        selected_fields = random.sample(data_fields, min(2, len(data_fields)))
        field_names = [field['id'] for field in selected_fields]
        
        base_field = field_names[0]
        if len(field_names) > 1:
            secondary_field = field_names[1]
        else:
            secondary_field = base_field
        
        # Generic template: rank(ts_rank(scale(field1), 5)) - rank(ts_rank(scale(field2), 3))
        template = f"subtract(rank(ts_rank(scale({base_field}), 5)), rank(ts_rank(scale({secondary_field}), 3)))"
        
        return template
    
    def extract_fields_from_template(self, template: str) -> List[str]:
        """Extract field names from a template"""
        # Simple field extraction - look for common field patterns
        field_patterns = [
            r'fnd\d+_[a-zA-Z0-9_]+',  # fnd28_field_name
            r'[a-zA-Z_][a-zA-Z0-9_]*',  # cash_st, volume, etc.
        ]
        
        fields = []
        for pattern in field_patterns:
            matches = re.findall(pattern, template)
            fields.extend(matches)
        
        # Filter out common operators
        common_operators = {'max', 'min', 'log', 'abs', 'scale', 'rank', 'ts_rank', 'ts_mean', 'ts_std', 'ts_delta', 'ts_corr', 'divide', 'multiply', 'add', 'subtract', 'if_else', 'winsorize', 'group_neutralize'}
        fields = [f for f in fields if f not in common_operators and len(f) > 2]
        
        return list(set(fields))
    
    def simulate_pyramid_template(self, template: Dict, region: str, delay: int) -> PyramidResult:
        """Simulate a pyramid-cracking template"""
        try:
            # Create simulation data
            simulation_data = {
                'type': 'REGULAR',
                'settings': {
                    'instrumentType': 'EQUITY',
                    'region': region,
                    'universe': self.region_configs[region].universe,
                    'delay': delay,
                    'neutralization': template.get('neutralization', 'INDUSTRY'),
                    'decay': 0.01,
                    'maxTrade': self.region_configs[region].max_trade
                },
                'expression': template['template']
            }
            
            # Submit simulation
            response = self.sess.post(
                'https://platform.worldquantbrain.com/static/simulations.json',
                json=simulation_data
            )
            response.raise_for_status()
            
            result_data = response.json()
            
            # Extract results
            success = result_data.get('success', False)
            sharpe = result_data.get('sharpe', 0.0)
            fitness = result_data.get('fitness', 0.0)
            turnover = result_data.get('turnover', 0.0)
            pnl = result_data.get('pnl', 0.0)
            
            # Calculate breakthrough score
            breakthrough_score = self.calculate_breakthrough_score(sharpe, fitness, turnover)
            
            return PyramidResult(
                strategy=template['strategy'],
                region=region,
                delay=delay,
                template=template['template'],
                success=success,
                sharpe=sharpe,
                fitness=fitness,
                turnover=turnover,
                pnl=pnl,
                timestamp=time.time(),
                pyramid_level=self.get_pyramid_level(region),
                breakthrough_score=breakthrough_score
            )
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return PyramidResult(
                strategy=template['strategy'],
                region=region,
                delay=delay,
                template=template['template'],
                success=False,
                sharpe=0.0,
                fitness=0.0,
                turnover=0.0,
                pnl=0.0,
                error_message=str(e),
                timestamp=time.time(),
                pyramid_level=self.get_pyramid_level(region),
                breakthrough_score=0.0
            )
    
    def calculate_breakthrough_score(self, sharpe: float, fitness: float, turnover: float) -> float:
        """Calculate breakthrough score for pyramid cracking"""
        # Weighted combination of metrics
        sharpe_weight = 0.5
        fitness_weight = 0.3
        turnover_weight = 0.2
        
        # Normalize turnover (lower is better, so invert)
        turnover_score = max(0, 1 - turnover)
        
        breakthrough_score = (
            sharpe_weight * sharpe +
            fitness_weight * fitness +
            turnover_weight * turnover_score
        )
        
        return breakthrough_score
    
    def get_pyramid_level(self, region: str) -> int:
        """Get pyramid level for region (higher = more challenging)"""
        level_map = {
            "USA": 3,  # Highest level
            "EUR": 3,  # Highest level
            "CHN": 2,  # Medium level
            "GLB": 2,  # Medium level
            "ASI": 1   # Lowest level
        }
        return level_map.get(region, 1)
    
    def run_concurrent_pyramid_cracking(self, regions: List[str] = None, iterations: int = 100):
        """Run 3 concurrent pyramid cracking simulations"""
        if regions is None:
            regions = self.regions
        
        logger.info(f"🚀 Starting 3 concurrent pyramid cracking simulations...")
        logger.info(f"🎯 Strategies: {[s.value for s in self.strategies]}")
        logger.info(f"🌍 Regions: {regions}")
        logger.info(f"🔄 Iterations: {iterations}")
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            iteration = 0
            while iteration < iterations:
                # Fill available slots with concurrent tasks
                self._fill_available_slots_concurrent(regions)
                
                # Process completed futures
                self._process_completed_futures()
                
                # Update progress
                if iteration % 10 == 0:
                    self._display_progress()
                
                iteration += 1
                time.sleep(0.1)  # Small delay to prevent busy waiting
            
            # Wait for remaining futures to complete
            self._wait_for_futures_completion()
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Received interrupt signal. Stopping gracefully...")
            self._wait_for_futures_completion()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        return self.all_results
    
    def _fill_available_slots_concurrent(self, regions: List[str]):
        """Fill available slots with concurrent pyramid cracking tasks"""
        available_slots = self.max_concurrent - len(self.active_futures)
        
        if available_slots > 0:
            logger.info(f"🎯 Filling {available_slots} available slots with CONCURRENT pyramid tasks...")
            
            for i in range(available_slots):
                strategy = self.strategies[i % len(self.strategies)]
                region = random.choice(regions)
                
                # Submit concurrent task
                future = self.executor.submit(self._crack_pyramid_concurrent, strategy, region)
                future_id = f"{strategy.value}_{int(time.time() * 1000)}"
                self.active_futures[future_id] = future
                logger.info(f"🚀 Started CONCURRENT {strategy.value.upper()} task: {future_id}")
    
    def _crack_pyramid_concurrent(self, strategy: PyramidStrategy, region: str) -> Optional[PyramidResult]:
        """Concurrently crack a pyramid using the specified strategy"""
        try:
            delay = self.select_optimal_delay(region)
            
            # Generate pyramid-cracking template
            template = self.generate_pyramid_template(strategy, region, delay)
            if not template:
                logger.warning(f"No template generated for {strategy.value} in {region}")
                return None
            
            logger.info(f"🔍 {strategy.value.upper()}: {template['template'][:50]}...")
            
            # Simulate the template
            result = self.simulate_pyramid_template(template, region, delay)
            
            # Check for breakthrough
            if result.success and result.breakthrough_score >= self.breakthrough_threshold:
                self.breakthrough_count += 1
                if result.breakthrough_score > self.best_breakthrough:
                    self.best_breakthrough = result.breakthrough_score
                logger.info(f"💥 BREAKTHROUGH! {strategy.value}: Sharpe={result.sharpe:.3f}, Score={result.breakthrough_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {strategy.value} pyramid cracking: {e}")
            return None
    
    def _process_completed_futures(self):
        """Process completed futures and update results"""
        completed_futures = []
        
        for future_id, future in self.active_futures.items():
            if future.done():
                completed_futures.append(future_id)
                
                try:
                    result = future.result()
                    if result:
                        self.pyramid_results.append(result)
                        
                        # Add to results by region
                        region = result.region
                        if region not in self.all_results['pyramid_results']:
                            self.all_results['pyramid_results'][region] = []
                        self.all_results['pyramid_results'][region].append(asdict(result))
                        
                        # Update counters
                        if result.success:
                            self.successful_count += 1
                        else:
                            self.failed_count += 1
                        
                        self.completed_count += 1
                        
                        # Save progress
                        self.save_progress()
                
                except Exception as e:
                    logger.error(f"Error processing future {future_id}: {e}")
                    self.failed_count += 1
                    self.completed_count += 1
        
        # Remove completed futures
        for future_id in completed_futures:
            del self.active_futures[future_id]
    
    def _wait_for_futures_completion(self):
        """Wait for all active futures to complete"""
        logger.info("⏳ Waiting for active futures to complete...")
        
        while self.active_futures:
            self._process_completed_futures()
            time.sleep(0.1)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        logger.info("\n🛑 Received interrupt signal. Stopping gracefully...")
        self._wait_for_futures_completion()
    
    def _display_progress(self):
        """Display current progress"""
        total = self.completed_count + len(self.active_futures)
        success_rate = (self.successful_count / self.completed_count * 100) if self.completed_count > 0 else 0
        
        print(f"\r🔄 Progress: {self.completed_count}/{total} completed | "
              f"✅ {self.successful_count} successful | "
              f"❌ {self.failed_count} failed | "
              f"📈 {success_rate:.1f}% success rate | "
              f"💥 {self.breakthrough_count} breakthroughs | "
              f"🏆 Best: {self.best_breakthrough:.3f}", end="")
        sys.stdout.flush()
    
    def save_progress(self):
        """Save current progress to file"""
        try:
            progress_data = {
                'completed_count': self.completed_count,
                'successful_count': self.successful_count,
                'failed_count': self.failed_count,
                'breakthrough_count': self.breakthrough_count,
                'best_breakthrough': self.best_breakthrough,
                'timestamp': time.time()
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def load_progress(self):
        """Load previous progress from file"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                self.completed_count = progress_data.get('completed_count', 0)
                self.successful_count = progress_data.get('successful_count', 0)
                self.failed_count = progress_data.get('failed_count', 0)
                self.breakthrough_count = progress_data.get('breakthrough_count', 0)
                self.best_breakthrough = progress_data.get('best_breakthrough', 0.0)
                
                logger.info(f"📁 Loaded previous progress: {self.completed_count} completed, {self.breakthrough_count} breakthroughs")
                
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
    
    def analyze_breakthroughs(self) -> Dict:
        """Analyze breakthrough results"""
        successful_results = [r for r in self.pyramid_results if r.success]
        breakthrough_results = [r for r in successful_results if r.breakthrough_score >= self.breakthrough_threshold]
        
        analysis = {
            'total_simulations': len(self.pyramid_results),
            'successful_simulations': len(successful_results),
            'breakthrough_simulations': len(breakthrough_results),
            'breakthrough_rate': len(breakthrough_results) / len(successful_results) if successful_results else 0,
            'strategy_performance': {},
            'region_performance': {},
            'best_breakthroughs': []
        }
        
        # Analyze by strategy
        for strategy in self.strategies:
            strategy_results = [r for r in breakthrough_results if r.strategy == strategy.value]
            if strategy_results:
                analysis['strategy_performance'][strategy.value] = {
                    'count': len(strategy_results),
                    'avg_breakthrough_score': np.mean([r.breakthrough_score for r in strategy_results]),
                    'max_breakthrough_score': max([r.breakthrough_score for r in strategy_results]),
                    'avg_sharpe': np.mean([r.sharpe for r in strategy_results])
                }
        
        # Analyze by region
        for region in self.regions:
            region_results = [r for r in breakthrough_results if r.region == region]
            if region_results:
                analysis['region_performance'][region] = {
                    'count': len(region_results),
                    'avg_breakthrough_score': np.mean([r.breakthrough_score for r in region_results]),
                    'max_breakthrough_score': max([r.breakthrough_score for r in region_results]),
                    'avg_sharpe': np.mean([r.sharpe for r in region_results])
                }
        
        # Get best breakthroughs
        best_breakthroughs = sorted(breakthrough_results, key=lambda x: x.breakthrough_score, reverse=True)[:10]
        analysis['best_breakthroughs'] = [asdict(r) for r in best_breakthroughs]
        
        return analysis
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            filename = self.results_file
            
        try:
            # Add breakthrough analysis to results
            results['breakthrough_analysis'] = self.analyze_breakthroughs()
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Pyramid Crasher - 3 Concurrent Simulations for Pyramid Cracking')
    parser.add_argument('--credentials', default='credential.txt', help='Path to credentials file')
    parser.add_argument('--output', default='pyramid_results.json', help='Output filename')
    parser.add_argument('--progress-file', default='pyramid_progress.json', help='Progress file')
    parser.add_argument('--regions', nargs='+', help='Regions to process (default: all)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations to run')
    parser.add_argument('--max-concurrent', type=int, default=3, help='Maximum concurrent simulations (default: 3)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    
    args = parser.parse_args()
    
    try:
        # Initialize pyramid crasher
        crasher = PyramidCrasher(
            args.credentials,
            args.max_concurrent,
            args.progress_file,
            args.output
        )
        
        # Run concurrent pyramid cracking
        results = crasher.run_concurrent_pyramid_cracking(args.regions, args.iterations)
        
        # Save final results
        crasher.save_results(results, args.output)
        
        # Print final summary
        print(f"\n{'='*70}")
        print("🎉 PYRAMID CRACKING COMPLETE!")
        print(f"{'='*70}")
        
        total_simulations = crasher.completed_count
        successful_sims = crasher.successful_count
        breakthrough_sims = crasher.breakthrough_count
        
        print(f"📊 Final Statistics:")
        print(f"   Total concurrent simulations: {total_simulations}")
        print(f"   Successful simulations: {successful_sims}")
        print(f"   Breakthrough simulations: {breakthrough_sims}")
        print(f"   Success rate: {successful_sims/total_simulations*100:.1f}%" if total_simulations > 0 else "   Success rate: N/A")
        print(f"   Breakthrough rate: {breakthrough_sims/successful_sims*100:.1f}%" if successful_sims > 0 else "   Breakthrough rate: N/A")
        print(f"   Best breakthrough score: {crasher.best_breakthrough:.3f}")
        print(f"   Results saved to: {args.output}")
        print(f"   Progress saved to: {args.progress_file}")
        print(f"   Strategies used: {[s.value for s in crasher.strategies]}")
        print(f"   Max Concurrent: {crasher.max_concurrent}")
        
    except Exception as e:
        logger.error(f"Pyramid cracking failed: {e}")
        raise

if __name__ == '__main__':
    main()
