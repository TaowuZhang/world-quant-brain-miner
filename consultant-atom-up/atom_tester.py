#!/usr/bin/env python3
"""
Atom-Based Alpha Testing System for WorldQuant Brain
- Tests atoms (single dataset alphas) across all cached data fields
- Comprehensive statistical analysis and PnL tracking
- Stores detailed information for statistical analysis
"""

import argparse
import requests
import json
import os
import random
import time
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from requests.auth import HTTPBasicAuth
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime
import threading
import sys
import math
import subprocess
from collections import defaultdict, Counter
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('atom_tester.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AtomTestResult:
    """Result of an atom test"""
    atom_id: str
    expression: str
    dataset_id: str
    dataset_name: str
    region: str
    universe: str
    delay: int
    neutralization: str
    status: str  # 'success', 'failed', 'error'
    sharpe_ratio: Optional[float] = None
    returns: Optional[float] = None
    max_drawdown: Optional[float] = None
    hit_ratio: Optional[float] = None
    pnl_data: Optional[Dict] = None
    error_message: Optional[str] = None
    test_timestamp: str = None
    execution_time: Optional[float] = None

@dataclass
class AtomStatistics:
    """Statistical analysis of atom performance"""
    dataset_id: str
    dataset_name: str
    total_tests: int
    successful_tests: int
    failed_tests: int
    avg_sharpe: Optional[float] = None
    max_sharpe: Optional[float] = None
    min_sharpe: Optional[float] = None
    avg_returns: Optional[float] = None
    avg_max_drawdown: Optional[float] = None
    avg_hit_ratio: Optional[float] = None
    best_atom: Optional[str] = None
    worst_atom: Optional[str] = None
    success_rate: Optional[float] = None

class AtomTester:
    """Main class for testing atoms"""
    
    def __init__(self, credential_file: str = "credential.txt"):
        """Initialize the atom tester"""
        self.credential_file = credential_file
        self.results: List[AtomTestResult] = []
        self.statistics: Dict[str, AtomStatistics] = {}
        self.operators = []
        self.data_fields = {}
        self.regions = ['USA', 'EUR', 'ASI', 'CHN', 'GLB']
        self.universes = ['TOP3000', 'TOP2000', 'TOP1000']
        self.neutralizations = ['INDUSTRY', 'SUBINDUSTRY', 'SECTOR', 'COUNTRY', 'NONE']
        
        # Load credentials and setup session
        self._load_credentials()
        
        # Setup session
        self.sess = requests.Session()
        
        # Setup authentication
        self._setup_auth()
        
        # Load operators and data fields
        self._load_operators()
        self._load_data_fields()
        
        # Results storage
        self.results_file = "atom_test_results.json"
        self.statistics_file = "atom_statistics.json"
        
    def _load_credentials(self):
        """Load WorldQuant Brain credentials"""
        try:
            with open(self.credential_file, 'r') as f:
                content = f.read().strip()
                
                # Try JSON format first (array format)
                if content.startswith('[') and content.endswith(']'):
                    import json
                    credentials = json.loads(content)
                    if len(credentials) == 2:
                        self.username = credentials[0].strip()
                        self.password = credentials[1].strip()
                    else:
                        raise ValueError("JSON credentials must have exactly 2 elements")
                else:
                    # Try two-line format
                    lines = content.split('\n')
                    if len(lines) >= 2:
                        self.username = lines[0].strip()
                        self.password = lines[1].strip()
                    else:
                        raise ValueError("Credential file must have at least 2 lines")
                        
            logger.info(f"✅ Credentials loaded for user: {self.username}")
        except Exception as e:
            logger.error(f"❌ Failed to load credentials: {e}")
            sys.exit(1)
    
    def _setup_auth(self):
        """Setup authentication for WorldQuant Brain API"""
        try:
            # Authenticate with WorldQuant Brain
            auth_response = self.sess.post(
                'https://api.worldquantbrain.com/authentication',
                auth=HTTPBasicAuth(self.username, self.password)
            )
            
            if auth_response.status_code == 201:
                logger.info("✅ Authentication successful")
            else:
                logger.error(f"❌ Authentication failed: {auth_response.status_code}")
                raise Exception("Authentication failed")
                
        except Exception as e:
            logger.error(f"❌ Failed to setup authentication: {e}")
            raise
    
    def _load_operators(self):
        """Load operators from operatorRAW.json"""
        try:
            with open('operatorRAW.json', 'r', encoding='utf-8') as f:
                self.operators = json.load(f)
            logger.info(f"✅ Loaded {len(self.operators)} operators")
        except Exception as e:
            logger.error(f"❌ Failed to load operators: {e}")
            sys.exit(1)
    
    def _load_data_fields(self):
        """Load all cached data fields with API fetching if cache is missing"""
        # Define regions and their available delays
        region_delays = {
            'USA': [0, 1],
            'GLB': [1],  # GLB only has delay=1
            'EUR': [0, 1],
            'ASI': [1],  # ASI only has delay=1
            'CHN': [1]   # CHN only has delay=1
        }
        
        for region, delays in region_delays.items():
            for delay in delays:
                cache_file = f"data_fields_cache_{region}_{delay}.json"
                
                if os.path.exists(cache_file):
                    # Load from cache
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        cache_key = f"{region}_{delay}"
                        self.data_fields[cache_key] = data
                        logger.info(f"✅ Loaded {len(data)} cached fields from {cache_file}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to load {cache_file}: {e}")
                else:
                    # Fetch from API if cache doesn't exist
                    logger.info(f"📡 No cache found for {region} delay={delay}, fetching from API...")
                    fields = self._fetch_data_fields_from_api(region, delay)
                    if fields:
                        cache_key = f"{region}_{delay}"
                        self.data_fields[cache_key] = fields
                        logger.info(f"✅ Fetched {len(fields)} fields for {region} delay={delay}")
                    else:
                        logger.warning(f"⚠️  No fields fetched for {region} delay={delay}")
        
        total_fields = sum(len(fields) for fields in self.data_fields.values())
        logger.info(f"✅ Total data fields loaded: {total_fields}")
    
    def _fetch_data_fields_from_api(self, region: str, delay: int) -> List[Dict]:
        """Fetch data fields from WorldQuant Brain API"""
        try:
            # Region configurations
            region_configs = {
                "USA": {"universe": "TOP3000"},
                "GLB": {"universe": "TOP3000"},
                "EUR": {"universe": "TOP2500"},
                "ASI": {"universe": "MINVOL1M"},
                "CHN": {"universe": "TOP2000U"}
            }
            
            config = region_configs.get(region, {"universe": "TOP3000"})
            
            # Get available datasets from multiple categories with better error handling
            categories = ['fundamental', 'analyst', 'model', 'news', 'alternative']
            all_dataset_ids = []
            
            for category in categories:
                try:
                    datasets_params = {
                        'category': category,
                        'delay': delay,
                        'instrumentType': 'EQUITY',
                        'region': region,
                        'universe': config['universe'],
                        'limit': 20
                    }
                    
                    logger.info(f"Getting {category} datasets for region {region}")
                    datasets_response = self.sess.get(
                        'https://api.worldquantbrain.com/data-sets', 
                        params=datasets_params,
                        timeout=10  # Reduced timeout
                    )
                    
                    if datasets_response.status_code == 200:
                        datasets_data = datasets_response.json()
                        available_datasets = datasets_data.get('results', [])
                        category_dataset_ids = [ds.get('id') for ds in available_datasets if ds.get('id')]
                        all_dataset_ids.extend(category_dataset_ids)
                        logger.info(f"Found {len(category_dataset_ids)} {category} datasets for region {region}")
                    else:
                        logger.warning(f"Failed to get {category} datasets for region {region}: {datasets_response.status_code}")
                        
                except Exception as e:
                    logger.warning(f"Error getting {category} datasets for region {region}: {e}")
                    continue
            
            # Remove duplicates and use the combined list
            dataset_ids = list(set(all_dataset_ids))
            
            if not dataset_ids:
                logger.warning(f"No datasets found for region {region}, using fallback datasets")
                # Use more reliable fallback datasets
                fallback_datasets = {
                    'USA': ['fundamental6', 'fundamental2', 'analyst4', 'model16'],
                    'GLB': ['fundamental6', 'fundamental2', 'analyst4', 'model16'],
                    'EUR': ['fundamental6', 'fundamental2', 'analyst4', 'model16'],
                    'ASI': ['fundamental6', 'fundamental2'],  # Fewer datasets for ASI
                    'CHN': ['fundamental6', 'fundamental2']
                }
                dataset_ids = fallback_datasets.get(region, ['fundamental6', 'fundamental2'])
            
            logger.info(f"Total unique datasets for region {region}: {len(dataset_ids)}")
            
            # Get fields from datasets with pagination and better error handling
            all_fields = []
            max_datasets = min(5, len(dataset_ids))  # Reduced to 5 datasets for faster execution
            
            for dataset in dataset_ids[:max_datasets]:
                try:
                    dataset_fields = []
                    page = 1
                    max_pages = 3  # Reduced to 3 pages per dataset
                    
                    while page <= max_pages:
                        try:
                            params = {
                                'dataset.id': dataset,
                                'delay': delay,
                                'instrumentType': 'EQUITY',
                                'region': region,
                                'universe': config['universe'],
                                'limit': 50,
                                'page': page
                            }
                            
                            response = self.sess.get(
                                'https://api.worldquantbrain.com/data-fields', 
                                params=params,
                                timeout=10  # Reduced timeout
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                fields = data.get('results', [])
                                if not fields:  # No more fields on this page
                                    break
                                dataset_fields.extend(fields)
                                logger.info(f"Found {len(fields)} fields in dataset {dataset} page {page}")
                                page += 1
                            else:
                                logger.warning(f"Failed to get fields from dataset {dataset} page {page}: {response.status_code}")
                                break
                                
                        except Exception as e:
                            logger.warning(f"Error getting fields from dataset {dataset} page {page}: {e}")
                            break
                    
                    all_fields.extend(dataset_fields)
                    logger.info(f"Total fields from dataset {dataset}: {len(dataset_fields)}")
                    
                except Exception as e:
                    logger.warning(f"Error processing dataset {dataset}: {e}")
                    continue
            
            # Remove duplicates
            unique_fields = {field['id']: field for field in all_fields}.values()
            field_list = list(unique_fields)
            logger.info(f"Total unique fields for region {region}: {len(field_list)} (from {max_datasets} datasets)")
            
            # Cache the fetched data
            cache_file = f"data_fields_cache_{region}_{delay}.json"
            try:
                with open(cache_file, 'w') as f:
                    json.dump(field_list, f, indent=2)
                logger.info(f"Cached {len(field_list)} fields to {cache_file}")
            except Exception as cache_error:
                logger.warning(f"Failed to cache data fields: {cache_error}")
            
            return field_list
            
        except Exception as e:
            logger.error(f"Failed to fetch data fields for region {region}: {e}")
            return []
    
    def _get_simple_operators(self) -> List[Dict]:
        """Get operators suitable for atom creation"""
        # Filter operators that work well with single inputs
        simple_ops = []
        for op in self.operators:
            op_name = op['name']
            definition = op.get('definition', '')
            
            # Skip operators that require multiple inputs or are too complex
            if any(skip in op_name.lower() for skip in ['corr', 'co_', 'ts_', 'regress', 'combine']):
                continue
            
            # Include basic mathematical and statistical operators
            if any(include in op_name.lower() for include in ['rank', 'abs', 'log', 'sqrt', 'pow', 'sign', 'max', 'min', 'mean', 'std', 'sum', 'prod']):
                simple_ops.append(op)
            elif op_name in ['add', 'subtract', 'multiply', 'divide', 'neg', 'reciprocal']:
                simple_ops.append(op)
        
        return simple_ops
    
    def _generate_atom_expression(self, data_field: Dict, operator: Dict) -> str:
        """Generate an atom expression using a single data field and operator"""
        field_id = data_field['id']
        op_name = operator['name']
        definition = operator.get('definition', '')
        
        # Handle different operator patterns
        if 'rank' in op_name.lower():
            return f"rank({field_id})"
        elif 'abs' in op_name.lower():
            return f"abs({field_id})"
        elif 'log' in op_name.lower():
            return f"log({field_id})"
        elif 'sqrt' in op_name.lower():
            return f"sqrt({field_id})"
        elif 'sign' in op_name.lower():
            return f"sign({field_id})"
        elif 'neg' in op_name.lower():
            return f"neg({field_id})"
        elif 'reciprocal' in op_name.lower():
            return f"reciprocal({field_id})"
        elif 'max' in op_name.lower():
            return f"max({field_id}, 20)"  # Add a reasonable lookback
        elif 'min' in op_name.lower():
            return f"min({field_id}, 20)"
        elif 'mean' in op_name.lower():
            return f"mean({field_id}, 20)"
        elif 'std' in op_name.lower():
            return f"std({field_id}, 20)"
        elif 'sum' in op_name.lower():
            return f"sum({field_id}, 20)"
        elif 'prod' in op_name.lower():
            return f"prod({field_id}, 20)"
        else:
            # Default: just use the field directly
            return field_id
    
    def _test_atom(self, expression: str, region: str, universe: str, delay: int, neutralization: str) -> AtomTestResult:
        """Test a single atom expression"""
        start_time = time.time()
        
        try:
            # Create simulation
            simulation_data = {
                "type": "REGULAR",
                "settings": {
                    "instrumentType": "EQUITY",
                    "region": region,
                    "universe": universe,
                    "delay": delay,
                    "neutralization": neutralization,
                    "decay": 0,
                    "truncation": 0.08,
                    "pasteurization": "ON",
                    "unitHandling": "VERIFY",
                    "nanHandling": "OFF",
                    "maxTrade": "OFF",
                    "language": "FASTEXPR",
                    "visualization": False,
                    "testPeriod": "P5Y0M0D"
                },
                "regular": expression
            }
            
            # Submit simulation
            response = self.sess.post(
                "https://api.worldquantbrain.com/simulations",
                json=simulation_data,
                timeout=30
            )
            
            # Handle authentication errors
            if response.status_code == 401:
                logger.info("Session expired, re-authenticating...")
                self._setup_auth()
                response = self.sess.post(
                    "https://api.worldquantbrain.com/simulations",
                    json=simulation_data,
                    timeout=30
                )
            
            if response.status_code != 201:
                return AtomTestResult(
                    atom_id="",
                    expression=expression,
                    dataset_id="",
                    dataset_name="",
                    region=region,
                    universe=universe,
                    delay=delay,
                    neutralization=neutralization,
                    status="failed",
                    error_message=f"Alpha creation failed: {response.status_code} - {response.text}",
                    test_timestamp=datetime.now().isoformat(),
                    execution_time=time.time() - start_time
                )
            
            # Get progress URL from Location header
            progress_url = response.headers.get('Location')
            if not progress_url:
                return AtomTestResult(
                    atom_id="",
                    expression=expression,
                    dataset_id="",
                    dataset_name="",
                    region=region,
                    universe=universe,
                    delay=delay,
                    neutralization=neutralization,
                    status="failed",
                    error_message="No Location header in simulation response",
                    test_timestamp=datetime.now().isoformat(),
                    execution_time=time.time() - start_time
                )
            
            # Monitor simulation progress
            max_wait_time = 180  # 3 minutes maximum wait time (reduced from 5)
            simulation_start_time = time.time()
            alpha_id = None
            sim_data = None
            check_count = 0
            last_progress = None
            stuck_count = 0
            
            logger.info(f"🔍 Monitoring simulation progress for: {expression[:50]}...")
            
            while (time.time() - simulation_start_time) < max_wait_time:
                try:
                    check_count += 1
                    progress_response = self.sess.get(progress_url, timeout=10)  # Reduced timeout for progress checks
                    
                    if progress_response.status_code == 200:
                        progress_data = progress_response.json()
                        progress_status = progress_data.get('progress')
                        
                        # Check if progress is stuck
                        if progress_status == last_progress:
                            stuck_count += 1
                            if stuck_count > 30:  # If stuck for 30 checks (60 seconds), timeout
                                logger.warning(f"⚠️ Simulation appears stuck at {progress_status}, timing out...")
                                break
                        else:
                            stuck_count = 0
                            last_progress = progress_status
                        
                        # Log progress every 10 checks
                        if check_count % 10 == 0:
                            logger.info(f"⏳ Simulation check {check_count}: status={progress_status}")
                        
                        if progress_status == 'DONE':
                            alpha_id = progress_data.get('alphaId')
                            if alpha_id:
                                logger.info(f"✅ Simulation complete, fetching alpha {alpha_id}")
                                # Fetch the alpha data
                                alpha_response = self.sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}', timeout=30)
                                
                                if alpha_response.status_code == 200:
                                    sim_data = alpha_response.json()
                                    logger.info(f"✅ Alpha data fetched successfully")
                                    break
                                else:
                                    logger.error(f"❌ Failed to fetch alpha {alpha_id}: {alpha_response.status_code}")
                                    return AtomTestResult(
                                        atom_id=alpha_id,
                                        expression=expression,
                                        dataset_id="",
                                        dataset_name="",
                                        region=region,
                                        universe=universe,
                                        delay=delay,
                                        neutralization=neutralization,
                                        status="error",
                                        error_message=f"Failed to fetch alpha {alpha_id}: {alpha_response.status_code}",
                                        test_timestamp=datetime.now().isoformat(),
                                        execution_time=time.time() - start_time
                                    )
                        elif progress_status == 'ERROR':
                            error_message = progress_data.get('message', 'Unknown simulation error')
                            logger.error(f"❌ Simulation error: {error_message}")
                            return AtomTestResult(
                                atom_id="",
                                expression=expression,
                                dataset_id="",
                                dataset_name="",
                                region=region,
                                universe=universe,
                                delay=delay,
                                neutralization=neutralization,
                                status="failed",
                                error_message=f"Simulation error: {error_message}",
                                test_timestamp=datetime.now().isoformat(),
                                execution_time=time.time() - start_time
                            )
                        elif progress_status is None:
                            # Simulation failed or became invalid
                            logger.warning(f"⚠️ Simulation status became None, simulation likely failed")
                            return AtomTestResult(
                                atom_id="",
                                expression=expression,
                                dataset_id="",
                                dataset_name="",
                                region=region,
                                universe=universe,
                                delay=delay,
                                neutralization=neutralization,
                                status="failed",
                                error_message="Simulation failed - status became None",
                                test_timestamp=datetime.now().isoformat(),
                                execution_time=time.time() - start_time
                            )
                    
                    time.sleep(2)  # Wait 2 seconds before checking again
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error checking simulation progress: {e}")
                    time.sleep(2)
            
            # Check if simulation completed successfully
            if alpha_id is None or sim_data is None:
                return AtomTestResult(
                    atom_id="",
                    expression=expression,
                    dataset_id="",
                    dataset_name="",
                    region=region,
                    universe=universe,
                    delay=delay,
                    neutralization=neutralization,
                    status="error",
                    error_message="Simulation timed out or failed to complete",
                    test_timestamp=datetime.now().isoformat(),
                    execution_time=time.time() - start_time
                )
            
            # If we reach here, simulation was successful and sim_data is available
            # Extract performance metrics
            sharpe_ratio = sim_data.get('sharpeRatio', 0)
            returns = sim_data.get('returns', 0)
            max_drawdown = sim_data.get('maxDrawdown', 0)
            hit_ratio = sim_data.get('hitRatio', 0)
            
            # Get PnL data
            pnl_response = self.sess.get(
                f"https://api.worldquantbrain.com/alphas/{alpha_id}/recordsets/pnl",
                timeout=30
            )
            
            pnl_data = None
            if pnl_response.status_code == 200:
                pnl_data = pnl_response.json()
            
            return AtomTestResult(
                atom_id=alpha_id,
                expression=expression,
                dataset_id="",
                dataset_name="",
                region=region,
                universe=universe,
                delay=delay,
                neutralization=neutralization,
                status="success",
                sharpe_ratio=sharpe_ratio,
                returns=returns,
                max_drawdown=max_drawdown,
                hit_ratio=hit_ratio,
                pnl_data=pnl_data,
                test_timestamp=datetime.now().isoformat(),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return AtomTestResult(
                atom_id="",
                expression=expression,
                dataset_id="",
                dataset_name="",
                region=region,
                universe=universe,
                delay=delay,
                neutralization=neutralization,
                status="error",
                error_message=str(e),
                test_timestamp=datetime.now().isoformat(),
                execution_time=time.time() - start_time
            )
    
    def _calculate_statistics(self):
        """Calculate comprehensive statistics for all tested atoms"""
        logger.info("📊 Calculating atom statistics...")
        
        # Group results by dataset
        dataset_results = defaultdict(list)
        
        for result in self.results:
            if result.status == "success":
                # Extract dataset from expression (simplified)
                dataset_id = "unknown"
                dataset_name = "unknown"
                
                # Try to extract dataset from expression
                for region_key, fields in self.data_fields.items():
                    for field in fields:
                        if field['id'] in result.expression:
                            dataset_id = field.get('dataset', {}).get('id', 'unknown') if isinstance(field.get('dataset'), dict) else 'unknown'
                            dataset_name = field.get('dataset', {}).get('name', 'unknown') if isinstance(field.get('dataset'), dict) else 'unknown'
                            break
                
                dataset_results[dataset_id].append({
                    'result': result,
                    'dataset_name': dataset_name
                })
        
        # Calculate statistics for each dataset
        for dataset_id, results in dataset_results.items():
            if not results:
                continue
            
            dataset_name = results[0]['dataset_name']
            sharpe_ratios = [r['result'].sharpe_ratio for r in results if r['result'].sharpe_ratio is not None]
            returns = [r['result'].returns for r in results if r['result'].returns is not None]
            max_drawdowns = [r['result'].max_drawdown for r in results if r['result'].max_drawdown is not None]
            hit_ratios = [r['result'].hit_ratio for r in results if r['result'].hit_ratio is not None]
            
            # Find best and worst atoms
            best_atom = max(results, key=lambda x: x['result'].sharpe_ratio or -999)
            worst_atom = min(results, key=lambda x: x['result'].sharpe_ratio or 999)
            
            self.statistics[dataset_id] = AtomStatistics(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                total_tests=len(results),
                successful_tests=len([r for r in results if r['result'].status == "success"]),
                failed_tests=len([r for r in results if r['result'].status != "success"]),
                avg_sharpe=statistics.mean(sharpe_ratios) if sharpe_ratios else None,
                max_sharpe=max(sharpe_ratios) if sharpe_ratios else None,
                min_sharpe=min(sharpe_ratios) if sharpe_ratios else None,
                avg_returns=statistics.mean(returns) if returns else None,
                avg_max_drawdown=statistics.mean(max_drawdowns) if max_drawdowns else None,
                avg_hit_ratio=statistics.mean(hit_ratios) if hit_ratios else None,
                best_atom=best_atom['result'].expression,
                worst_atom=worst_atom['result'].expression,
                success_rate=len([r for r in results if r['result'].status == "success"]) / len(results) if results else 0
            )
    
    def _save_results(self):
        """Save results and statistics to files"""
        # Save detailed results
        results_data = [asdict(result) for result in self.results]
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats_data = {k: asdict(v) for k, v in self.statistics.items()}
        with open(self.statistics_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to {self.results_file}")
        logger.info(f"💾 Statistics saved to {self.statistics_file}")
    
    def _load_previous_results(self):
        """Load previous results if they exist"""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                
                self.results = [AtomTestResult(**result) for result in results_data]
                logger.info(f"📂 Loaded {len(self.results)} previous results")
                
                if os.path.exists(self.statistics_file):
                    with open(self.statistics_file, 'r', encoding='utf-8') as f:
                        stats_data = json.load(f)
                    
                    self.statistics = {k: AtomStatistics(**v) for k, v in stats_data.items()}
                    logger.info(f"📂 Loaded statistics for {len(self.statistics)} datasets")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to load previous results: {e}")
    
    def run_atom_tests(self, max_tests: int = 100, max_workers: int = 4):
        """Run comprehensive atom tests"""
        logger.info("🚀 Starting atom testing system...")
        
        # Load previous results
        self._load_previous_results()
        
        # Get simple operators
        simple_operators = self._get_simple_operators()
        logger.info(f"🔧 Using {len(simple_operators)} simple operators for atom generation")
        
        # Generate test cases
        test_cases = []
        
        for region_key, fields in self.data_fields.items():
            if not fields:
                continue
            
            # Parse region and delay from key (format: "region_delay")
            if '_' in region_key:
                region, delay_str = region_key.split('_', 1)
                delay = int(delay_str)
            else:
                region = region_key
                delay = 1
            
            # Sample fields to avoid too many tests
            sampled_fields = random.sample(fields, min(20, len(fields)))
            
            for field in sampled_fields:
                for operator in simple_operators[:10]:  # Limit operators per field
                    expression = self._generate_atom_expression(field, operator)
                    
                    # Extract dataset info safely
                    dataset_id = field.get('dataset', {}).get('id', 'unknown') if isinstance(field.get('dataset'), dict) else 'unknown'
                    dataset_name = field.get('dataset', {}).get('name', 'unknown') if isinstance(field.get('dataset'), dict) else 'unknown'
                    
                    # Test different configurations
                    for universe in self.universes[:2]:  # Limit universes
                        for test_delay in [delay]:  # Use the delay from the data
                            for neutralization in self.neutralizations[:3]:  # Limit neutralizations
                                test_cases.append({
                                    'expression': expression,
                                    'region': region,
                                    'universe': universe,
                                    'delay': test_delay,
                                    'neutralization': neutralization,
                                    'dataset_id': dataset_id,
                                    'dataset_name': dataset_name
                                })
        
        # Limit total test cases
        test_cases = test_cases[:max_tests]
        logger.info(f"🎯 Generated {len(test_cases)} test cases")
        
        # Run tests with threading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for i, test_case in enumerate(test_cases):
                future = executor.submit(
                    self._test_atom,
                    test_case['expression'],
                    test_case['region'],
                    test_case['universe'],
                    test_case['delay'],
                    test_case['neutralization']
                )
                futures.append((future, test_case, i))
            
            # Process results as they complete
            for future, test_case, test_num in futures:
                try:
                    result = future.result(timeout=200)  # 3.5 minutes to allow for simulation timeout
                    
                    # Add dataset information
                    result.dataset_id = test_case['dataset_id']
                    result.dataset_name = test_case['dataset_name']
                    
                    self.results.append(result)
                    
                    if result.status == "success":
                        logger.info(f"✅ Test {test_num+1}/{len(test_cases)}: {result.expression} - Sharpe: {result.sharpe_ratio:.3f}")
                    else:
                        logger.warning(f"❌ Test {test_num+1}/{len(test_cases)}: {result.expression} - {result.status}")
                    
                    # Save results periodically
                    if len(self.results) % 10 == 0:
                        self._save_results()
                        
                except Exception as e:
                    import traceback
                    logger.error(f"💥 Test {test_num+1} failed with exception: {type(e).__name__}: {str(e)}")
                    logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        
        # Calculate final statistics
        self._calculate_statistics()
        self._save_results()
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print comprehensive summary of results"""
        logger.info("\n" + "="*80)
        logger.info("📊 ATOM TESTING SUMMARY")
        logger.info("="*80)
        
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.status == "success"])
        failed_tests = len([r for r in self.results if r.status == "failed"])
        error_tests = len([r for r in self.results if r.status == "error"])
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        logger.info(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        logger.info(f"Errors: {error_tests} ({error_tests/total_tests*100:.1f}%)")
        
        if successful_tests > 0:
            sharpe_ratios = [r.sharpe_ratio for r in self.results if r.sharpe_ratio is not None]
            if sharpe_ratios:
                logger.info(f"\nSharpe Ratio Statistics:")
                logger.info(f"  Average: {statistics.mean(sharpe_ratios):.3f}")
                logger.info(f"  Maximum: {max(sharpe_ratios):.3f}")
                logger.info(f"  Minimum: {min(sharpe_ratios):.3f}")
                
                # Top performers
                top_results = sorted([r for r in self.results if r.sharpe_ratio is not None], 
                                   key=lambda x: x.sharpe_ratio, reverse=True)[:5]
                
                logger.info(f"\n🏆 Top 5 Atom Performers:")
                for i, result in enumerate(top_results, 1):
                    logger.info(f"  {i}. {result.expression}")
                    logger.info(f"     Dataset: {result.dataset_name}")
                    logger.info(f"     Sharpe: {result.sharpe_ratio:.3f}, Returns: {result.returns:.3f}")
        
        logger.info(f"\n📈 Dataset Performance Summary:")
        for dataset_id, stats in self.statistics.items():
            logger.info(f"  {dataset_id}: {stats.dataset_name}")
            logger.info(f"    Tests: {stats.total_tests}, Success Rate: {stats.success_rate:.1%}")
            if stats.avg_sharpe is not None:
                logger.info(f"    Avg Sharpe: {stats.avg_sharpe:.3f}, Max Sharpe: {stats.max_sharpe:.3f}")
        
        logger.info("="*80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Atom-Based Alpha Testing System")
    parser.add_argument("--max-tests", type=int, default=100, help="Maximum number of tests to run")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of concurrent workers")
    parser.add_argument("--credential-file", type=str, default="credential.txt", help="Credential file path")
    
    args = parser.parse_args()
    
    # Create atom tester
    tester = AtomTester(args.credential_file)
    
    # Run tests
    tester.run_atom_tests(max_tests=args.max_tests, max_workers=args.max_workers)

if __name__ == "__main__":
    main()
