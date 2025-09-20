#!/usr/bin/env python3
"""
Integrated Orchestrator - Coordinates Pyramid Crasher with Templates API
- Runs 3 concurrent pyramid cracking simulations alongside consultant-templates-api
- Coordinates between template generation and pyramid cracking
- Provides unified results and analysis
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

# Import from pyramid crasher
from pyramid_crasher import PyramidCrasher, PyramidStrategy, PyramidResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('integrated_orchestrator.log')
    ]
)
logger = logging.getLogger(__name__)

class IntegratedOrchestrator:
    """Orchestrates both template generation and pyramid cracking"""
    
    def __init__(self, credentials_path: str, deepseek_api_key: str, 
                 max_concurrent: int = 8, progress_file: str = "integrated_progress.json", 
                 results_file: str = "integrated_results.json"):
        """Initialize the integrated orchestrator"""
        self.credentials_path = credentials_path
        self.deepseek_api_key = deepseek_api_key
        self.max_concurrent = max_concurrent
        self.progress_file = progress_file
        self.results_file = results_file
        
        # Initialize pyramid crasher (3 concurrent strategies)
        self.pyramid_crasher = PyramidCrasher(
            credentials_path=credentials_path,
            max_concurrent=3,  # 3 pyramid cracking strategies
            progress_file="pyramid_progress.json",
            results_file="pyramid_results.json"
        )
        
        # Template generation slots (5 concurrent)
        self.template_slots = 5
        self.pyramid_slots = 3
        
        # TRUE CONCURRENT execution using ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
        self.active_futures = {}  # Track active Future objects
        self.completed_count = 0
        self.successful_count = 0
        self.failed_count = 0
        
        # Results storage
        self.all_results = {
            'metadata': {
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'max_concurrent': self.max_concurrent,
                'template_slots': self.template_slots,
                'pyramid_slots': self.pyramid_slots,
                'version': '1.0'
            },
            'template_results': {},
            'pyramid_results': {},
            'integrated_analysis': {}
        }
        
        # Region configurations
        self.region_configs = {
            "USA": {"universe": "TOP3000", "pyramid_multiplier": 1.8},
            "GLB": {"universe": "TOP3000", "pyramid_multiplier": 1.5},
            "EUR": {"universe": "TOP2500", "pyramid_multiplier": 1.7},
            "ASI": {"universe": "MINVOL1M", "pyramid_multiplier": 1.5, "max_trade": True},
            "CHN": {"universe": "TOP2000U", "pyramid_multiplier": 1.8, "max_trade": True}
        }
        
        self.regions = list(self.region_configs.keys())
        
        # Setup session
        self.sess = requests.Session()
        
        # Setup authentication
        self.setup_auth()
        
        # Load previous progress
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
    
    def run_integrated_simulation(self, regions: List[str] = None, iterations: int = 100):
        """Run integrated simulation with both template generation and pyramid cracking"""
        if regions is None:
            regions = self.regions
        
        logger.info(f"🚀 Starting Integrated Simulation...")
        logger.info(f"🎯 Template slots: {self.template_slots}")
        logger.info(f"💥 Pyramid slots: {self.pyramid_slots}")
        logger.info(f"🌍 Regions: {regions}")
        logger.info(f"🔄 Iterations: {iterations}")
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            iteration = 0
            while iteration < iterations:
                # Fill available slots with concurrent tasks
                self._fill_available_slots_integrated(regions)
                
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
    
    def _fill_available_slots_integrated(self, regions: List[str]):
        """Fill available slots with integrated tasks"""
        available_slots = self.max_concurrent - len(self.active_futures)
        
        if available_slots > 0:
            logger.info(f"🎯 Filling {available_slots} available slots with INTEGRATED tasks...")
            
            for i in range(available_slots):
                # Alternate between template generation and pyramid cracking
                if i < self.template_slots:
                    # Template generation slot
                    region = random.choice(regions)
                    future = self.executor.submit(self._generate_template_concurrent, region)
                    future_id = f"template_{int(time.time() * 1000)}"
                    self.active_futures[future_id] = future
                    logger.info(f"🚀 Started CONCURRENT TEMPLATE task: {future_id}")
                else:
                    # Pyramid cracking slot
                    strategy = random.choice(list(PyramidStrategy))
                    region = random.choice(regions)
                    future = self.executor.submit(self._crack_pyramid_integrated, strategy, region)
                    future_id = f"pyramid_{strategy.value}_{int(time.time() * 1000)}"
                    self.active_futures[future_id] = future
                    logger.info(f"🚀 Started CONCURRENT PYRAMID task: {future_id}")
    
    def _generate_template_concurrent(self, region: str) -> Optional[Dict]:
        """Generate a template concurrently"""
        try:
            # This would integrate with the existing template generation system
            # For now, we'll use a simplified version
            delay = self.pyramid_crasher.select_optimal_delay(region)
            
            # Generate a simple template
            template = self._generate_simple_template(region, delay)
            if not template:
                return None
            
            # Simulate the template
            result = self._simulate_template(template, region, delay)
            
            return {
                'type': 'template',
                'region': region,
                'template': template,
                'result': result,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in template generation: {e}")
            return None
    
    def _crack_pyramid_integrated(self, strategy: PyramidStrategy, region: str) -> Optional[Dict]:
        """Crack pyramid using integrated approach"""
        try:
            # Use the pyramid crasher's method
            delay = self.pyramid_crasher.select_optimal_delay(region)
            
            # Generate pyramid-cracking template
            template = self.pyramid_crasher.generate_pyramid_template(strategy, region, delay)
            if not template:
                return None
            
            # Simulate the template
            result = self.pyramid_crasher.simulate_pyramid_template(template, region, delay)
            
            return {
                'type': 'pyramid',
                'strategy': strategy.value,
                'region': region,
                'template': template,
                'result': result,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in pyramid cracking: {e}")
            return None
    
    def _generate_simple_template(self, region: str, delay: int) -> str:
        """Generate a simple template for testing"""
        # This is a placeholder - in real implementation, this would use the full template generation system
        data_fields = self.pyramid_crasher.get_data_fields_for_region(region, delay)
        if not data_fields:
            return None
        
        # Select a random field
        field = random.choice(data_fields)
        field_name = field['id']
        
        # Generate a simple template
        template = f"rank(ts_rank(scale({field_name}), 5))"
        
        return template
    
    def _simulate_template(self, template: str, region: str, delay: int) -> Dict:
        """Simulate a template"""
        try:
            # Create simulation data
            simulation_data = {
                'type': 'REGULAR',
                'settings': {
                    'instrumentType': 'EQUITY',
                    'region': region,
                    'universe': self.region_configs[region]['universe'],
                    'delay': delay,
                    'neutralization': 'INDUSTRY',
                    'decay': 0.01,
                    'maxTrade': self.region_configs[region].get('max_trade', False)
                },
                'expression': template
            }
            
            # Submit simulation
            response = self.pyramid_crasher.sess.post(
                'https://platform.worldquantbrain.com/static/simulations.json',
                json=simulation_data
            )
            response.raise_for_status()
            
            result_data = response.json()
            
            return {
                'success': result_data.get('success', False),
                'sharpe': result_data.get('sharpe', 0.0),
                'fitness': result_data.get('fitness', 0.0),
                'turnover': result_data.get('turnover', 0.0),
                'pnl': result_data.get('pnl', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {
                'success': False,
                'sharpe': 0.0,
                'fitness': 0.0,
                'turnover': 0.0,
                'pnl': 0.0,
                'error': str(e)
            }
    
    def _process_completed_futures(self):
        """Process completed futures and update results"""
        completed_futures = []
        
        for future_id, future in self.active_futures.items():
            if future.done():
                completed_futures.append(future_id)
                
                try:
                    result = future.result()
                    if result:
                        # Add to results by type and region
                        result_type = result['type']
                        region = result['region']
                        
                        if result_type not in self.all_results:
                            self.all_results[result_type] = {}
                        if region not in self.all_results[result_type]:
                            self.all_results[result_type][region] = []
                        
                        self.all_results[result_type][region].append(result)
                        
                        # Update counters
                        if result.get('result', {}).get('success', False):
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
              f"📈 {success_rate:.1f}% success rate", end="")
        sys.stdout.flush()
    
    def save_progress(self):
        """Save current progress to file"""
        try:
            progress_data = {
                'completed_count': self.completed_count,
                'successful_count': self.successful_count,
                'failed_count': self.failed_count,
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
                
                logger.info(f"📁 Loaded previous progress: {self.completed_count} completed")
                
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            filename = self.results_file
            
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Integrated Orchestrator - Coordinates Pyramid Crasher with Templates API')
    parser.add_argument('--credentials', default='credential.txt', help='Path to credentials file')
    parser.add_argument('--deepseek-key', required=True, help='DeepSeek API key')
    parser.add_argument('--output', default='integrated_results.json', help='Output filename')
    parser.add_argument('--progress-file', default='integrated_progress.json', help='Progress file')
    parser.add_argument('--regions', nargs='+', help='Regions to process (default: all)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations to run')
    parser.add_argument('--max-concurrent', type=int, default=8, help='Maximum concurrent simulations (default: 8)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        orchestrator = IntegratedOrchestrator(
            args.credentials,
            args.deepseek_key,
            args.max_concurrent,
            args.progress_file,
            args.output
        )
        
        # Run integrated simulation
        results = orchestrator.run_integrated_simulation(args.regions, args.iterations)
        
        # Save final results
        orchestrator.save_results(results, args.output)
        
        # Print final summary
        print(f"\n{'='*70}")
        print("🎉 INTEGRATED SIMULATION COMPLETE!")
        print(f"{'='*70}")
        
        total_simulations = orchestrator.completed_count
        successful_sims = orchestrator.successful_count
        
        print(f"📊 Final Statistics:")
        print(f"   Total concurrent simulations: {total_simulations}")
        print(f"   Successful simulations: {successful_sims}")
        print(f"   Failed simulations: {total_simulations - successful_sims}")
        print(f"   Success rate: {successful_sims/total_simulations*100:.1f}%" if total_simulations > 0 else "   Success rate: N/A")
        print(f"   Results saved to: {args.output}")
        print(f"   Progress saved to: {args.progress_file}")
        print(f"   Max Concurrent: {orchestrator.max_concurrent}")
        
    except Exception as e:
        logger.error(f"Integrated simulation failed: {e}")
        raise

if __name__ == '__main__':
    main()
