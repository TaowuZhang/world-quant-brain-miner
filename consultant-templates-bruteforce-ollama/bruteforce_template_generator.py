#!/usr/bin/env python3
"""
Bruteforce Template Generator for WorldQuant Brain
- Generates ONE atom/light-weight template at a time using Ollama
- Tests each template across ALL data fields and ALL regions
- Uses similar thread management from consultant-templates-ollama
- Supports custom template JSON input
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
            self.stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
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
        logging.FileHandler('bruteforce_template_generator.log', encoding='utf-8')
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
    neutralization_options: List[str] = None
    
    def __post_init__(self):
        if self.neutralization_options is None:
            # Region-specific neutralization options based on WorldQuant Brain API availability
            if self.region == "USA":
                self.neutralization_options = ["INDUSTRY", "SUBINDUSTRY", "SECTOR", "NONE"]
            elif self.region == "EUR":
                self.neutralization_options = ["INDUSTRY", "SUBINDUSTRY", "SECTOR", "NONE"]
            elif self.region == "CHN":
                # CHN region has limited neutralization options
                self.neutralization_options = ["INDUSTRY", "SUBINDUSTRY", "SECTOR", "NONE"]
            elif self.region == "GLB":
                self.neutralization_options = ["INDUSTRY", "SUBINDUSTRY", "SECTOR", "COUNTRY", "NONE"]
            elif self.region == "ASI":
                # ASI region has limited neutralization options
                self.neutralization_options = ["INDUSTRY", "SUBINDUSTRY", "SECTOR", "NONE"]
            else:
                # Default fallback
                self.neutralization_options = ["INDUSTRY", "NONE"]

@dataclass
class SimulationSettings:
    """Settings for simulation"""
    region: str
    universe: str
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
class BruteforceResult:
    """Result of a bruteforce simulation"""
    template: str
    region: str
    data_field: str
    neutralization: str
    success: bool
    sharpe: float = 0.0
    returns: float = 0.0
    max_drawdown: float = 0.0
    margin: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    error_message: str = ""
    simulation_time: float = 0.0

class BruteforceTemplateGenerator:
    def __init__(self, credentials_path: str, ollama_model: str = "llama3.1", max_concurrent: int = 8, target_dataset: str = None):
        """Initialize the bruteforce template generator"""
        self.sess = requests.Session()
        self.credentials_path = credentials_path
        self.ollama_model = ollama_model
        self.max_concurrent = min(max_concurrent, 8)  # WorldQuant Brain limit is 8
        self.target_dataset = target_dataset  # Specific dataset to test
        
        # Thread management similar to consultant-templates-ollama
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
        self.active_futures = {}
        self.future_start_times = {}
        self.future_timeout = 300  # 5 minutes timeout
        self.completed_count = 0
        self.successful_count = 0
        self.failed_count = 0
        
        # Results tracking
        self.results = []
        self.results_file = "bruteforce_results.json"
        self.progress_file = "bruteforce_progress.json"
        
        # Track templates that have VECTOR field issues
        self.templates_with_vector_issues = set()
        
        # Success criteria
        self.success_criteria = {
            'min_sharpe': 1.25,
            'min_margin_bps': 10,  # 10 basis points
            'min_fitness': 1.0
        }
        
        # Template validation
        self.template_validation_attempts = 3
        self.test_data_fields = ['close', 'open', 'high', 'low', 'volume']  # Simple test fields
        
        # Load operators
        self.operators = self.load_operators()
        
        # Region configurations (matching consultant-templates-api)
        self.regions = {
            "USA": RegionConfig("USA", "TOP3000", 1, False),
            "EUR": RegionConfig("EUR", "TOP2500", 1, False),
            "CHN": RegionConfig("CHN", "TOP2000U", 1, True),
            "GLB": RegionConfig("GLB", "TOP3000", 1, False),
            "ASI": RegionConfig("ASI", "MINVOL1M", 1, True)
        }
        
        self.setup_auth()
        
    def load_operators(self) -> List[Dict]:
        """Load operators from operatorRAW.json"""
        try:
            # Try to load from consultant-templates-ollama directory first
            operator_files = [
                'operatorRAW.json',
                '../consultant-templates-ollama/operatorRAW.json',
                '../../consultant-templates-ollama/operatorRAW.json'
            ]
            
            for operator_file in operator_files:
                if os.path.exists(operator_file):
                    with open(operator_file, 'r') as f:
                        operators = json.load(f)
                    logger.info(f"📊 Loaded {len(operators)} operators from {operator_file}")
                    return operators
            
            # If no operator file found, create a basic set
            logger.warning("⚠️ No operatorRAW.json found, using basic operator set")
            return self._get_basic_operators()
            
        except Exception as e:
            logger.error(f"❌ Failed to load operators: {e}")
            return self._get_basic_operators()
    
    def _get_basic_operators(self) -> List[Dict]:
        """Get basic operators if operatorRAW.json is not available"""
        return [
            {"name": "add", "description": "Add all inputs (at least 2 inputs required)", "definition": "add(x, y, filter = false), x + y"},
            {"name": "subtract", "description": "x-y. If filter = true, filter all input NaN to 0 before subtracting", "definition": "subtract(x, y, filter=false), x - y"},
            {"name": "multiply", "description": "Multiply all inputs (at least 2 inputs required)", "definition": "multiply(x, y, filter = false), x * y"},
            {"name": "divide", "description": "x/y. If filter = true, filter all input NaN to 0 before dividing", "definition": "divide(x, y, filter=false), x / y"},
            {"name": "abs", "description": "Absolute value of x", "definition": "abs(x)"},
            {"name": "log", "description": "Natural logarithm", "definition": "log(x)"},
            {"name": "sign", "description": "Sign of x", "definition": "sign(x)"},
            {"name": "rank", "description": "Rank of x within each group", "definition": "rank(x, d)"},
            {"name": "ts_rank", "description": "Time series rank of x over d days", "definition": "ts_rank(x, d)"},
            {"name": "ts_delta", "description": "Time series delta of x over d days", "definition": "ts_delta(x, d)"},
            {"name": "ts_mean", "description": "Time series mean of x over d days", "definition": "ts_mean(x, d)"},
            {"name": "ts_std_dev", "description": "Time series standard deviation of x over d days", "definition": "ts_std_dev(x, d)"},
            {"name": "ts_zscore", "description": "Time series z-score of x over d days", "definition": "ts_zscore(x, d)"},
            {"name": "ts_corr", "description": "Time series correlation between x and y over d days", "definition": "ts_corr(x, y, d)"},
            {"name": "ts_regression", "description": "Time series regression of y on x over d days", "definition": "ts_regression(y, x, d)"},
            {"name": "group_normalize", "description": "Normalize x within each group", "definition": "group_normalize(x, group)"},
            {"name": "winsorize", "description": "Winsorize x to remove outliers", "definition": "winsorize(x, std=3)"},
            {"name": "sma", "description": "Simple moving average of x over d days", "definition": "sma(x, d)"},
            {"name": "ema", "description": "Exponential moving average of x over d days", "definition": "ema(x, d)"},
            {"name": "max", "description": "Maximum of all inputs", "definition": "max(x, y, ...)"},
            {"name": "min", "description": "Minimum of all inputs", "definition": "min(x, y, ...)"}
        ]
        
    def check_success_criteria(self, result: BruteforceResult) -> bool:
        """Check if result meets success criteria"""
        if not result.success:
            return False
        
        margin_bps = result.margin * 10000  # Convert to basis points
        
        meets_criteria = (
            result.sharpe > self.success_criteria['min_sharpe'] and
            margin_bps > self.success_criteria['min_margin_bps'] and
            result.fitness > self.success_criteria['min_fitness']
        )
        
        if meets_criteria:
            logger.info(f"🏆 SUCCESS CRITERIA MET: Sharpe={result.sharpe:.3f}, Margin={margin_bps:.1f}bps, Fitness={result.fitness:.3f}")
        
        return meets_criteria
    
    def validate_template_with_test_data(self, template: str) -> bool:
        """Validate template with simple test data fields"""
        try:
            # Test with simple data fields
            test_template = template
            for field in self.test_data_fields:
                if f"{{{field}}}" in test_template:
                    test_template = test_template.replace(f"{{{field}}}", field)
            
            # Basic syntax validation
            if any(op in test_template for op in ['>', '<', '>=', '<=', '==', '!=', '&&', '||', '%']):
                logger.warning(f"🚫 Template contains forbidden operators: {template}")
                return False
            
            # Check for basic structure
            if len(test_template.strip()) < 5:
                logger.warning(f"🚫 Template too short: {template}")
                return False
            
            # Check operator count (should be <= 5)
            operator_count = self._count_operators_in_template(template)
            if operator_count > 5:
                logger.warning(f"🚫 Template has too many operators ({operator_count} > 5): {template}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"🚫 Template validation failed: {e}")
            return False
    
    def _count_operators_in_template(self, template: str) -> int:
        """Count the number of operators used in a template"""
        operator_count = 0
        for op in self.operators:
            if op['name'] in template:
                operator_count += template.count(op['name'])
        return operator_count
    
    def regenerate_template_if_needed(self, original_template: str, attempt: int) -> Optional[str]:
        """Regenerate template if validation fails"""
        if attempt >= self.template_validation_attempts:
            logger.error(f"❌ Max validation attempts reached for template: {original_template}")
            return None
        
        logger.info(f"🔄 Regenerating template (attempt {attempt + 1}/{self.template_validation_attempts})")
        
        # Try different prompts for regeneration
        regeneration_prompts = [
            "Generate a simple alpha expression using basic price data (open, high, low, close)",
            "Create a lightweight alpha using volume and price data",
            "Generate a simple momentum alpha expression",
            "Create a basic mean reversion alpha",
            "Generate a simple volatility-based alpha"
        ]
        
        prompt = regeneration_prompts[attempt % len(regeneration_prompts)]
        new_template = self.call_ollama_api(prompt)
        
        if new_template and self.validate_template_with_test_data(new_template):
            logger.info(f"✅ Regenerated valid template: {new_template}")
            return new_template
        else:
            logger.warning(f"⚠️ Regenerated template failed validation, trying again...")
            return self.regenerate_template_if_needed(original_template, attempt + 1)
    
    def save_progress(self):
        """Save current progress to file for continuation"""
        try:
            progress_data = {
                'timestamp': time.time(),
                'total_simulations': len(self.results),
                'successful_simulations': len([r for r in self.results if r.success]),
                'failed_simulations': len([r for r in self.results if not r.success]),
                'success_criteria_met': len([r for r in self.results if self.check_success_criteria(r)]),
                'current_batch': getattr(self, 'current_batch', 0),
                'current_template': getattr(self, 'current_template', 0),
                'templates_with_vector_issues': list(self.templates_with_vector_issues),
                'resume_info': {
                    'can_resume': True,
                    'last_saved': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    'total_simulations': len(self.results),
                    'success_rate': len([r for r in self.results if r.success]) / max(len(self.results), 1) * 100
                },
                'results': [asdict(r) for r in self.results]
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            logger.info(f"💾 Progress saved to {self.progress_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save progress: {e}")
    
    def load_progress(self) -> bool:
        """Load progress from file for continuation"""
        try:
            if not os.path.exists(self.progress_file):
                logger.info("📁 No progress file found, starting fresh")
                return False
            
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # Restore results
            self.results = []
            for result_data in progress_data.get('results', []):
                result = BruteforceResult(**result_data)
                self.results.append(result)
            
            # Restore batch and template tracking
            self.current_batch = progress_data.get('current_batch', 0)
            self.current_template = progress_data.get('current_template', 0)
            
            # Restore VECTOR field issues tracking
            self.templates_with_vector_issues = set(progress_data.get('templates_with_vector_issues', []))
            
            # Calculate statistics
            total_simulations = len(self.results)
            successful_simulations = len([r for r in self.results if r.success])
            failed_simulations = len([r for r in self.results if not r.success])
            success_criteria_met = len([r for r in self.results if self.check_success_criteria(r)])
            
            logger.info(f"📁 Loaded progress:")
            logger.info(f"   📊 Total simulations: {total_simulations}")
            logger.info(f"   ✅ Successful: {successful_simulations}")
            logger.info(f"   ❌ Failed: {failed_simulations}")
            logger.info(f"   🏆 Success criteria met: {success_criteria_met}")
            logger.info(f"   📦 Current batch: {self.current_batch}")
            logger.info(f"   📝 Current template: {self.current_template}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load progress: {e}")
            return False
    
    def can_resume(self) -> bool:
        """Check if progress can be resumed"""
        try:
            if not os.path.exists(self.progress_file):
                return False
            
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # Check if resume is supported
            resume_info = progress_data.get('resume_info', {})
            return resume_info.get('can_resume', False)
            
        except Exception as e:
            logger.error(f"❌ Failed to check resume capability: {e}")
            return False
        
    def setup_auth(self):
        """Setup authentication from credentials file"""
        try:
            with open(self.credentials_path, 'r') as f:
                credentials = json.load(f)
            
            # Handle both JSON format {"username": "...", "password": "..."} and array format ["username", "password"]
            if isinstance(credentials, dict):
                username = credentials['username']
                password = credentials['password']
            elif isinstance(credentials, list):
                username = credentials[0]
                password = credentials[1]
            else:
                raise ValueError("Invalid credentials format")
            
            # Authenticate with WorldQuant Brain using the same method as consultant-templates-ollama
            auth_response = self.sess.post(
                'https://api.worldquantbrain.com/authentication',
                auth=HTTPBasicAuth(username, password)
            )
            
            if auth_response.status_code == 201:
                logger.info("✅ Authentication successful")
            else:
                logger.error(f"❌ Authentication failed: {auth_response.status_code}")
                raise Exception("Authentication failed")
                
        except Exception as e:
            logger.error(f"❌ Failed to setup authentication: {e}")
            raise

    def get_data_fields_for_region(self, region: str, delay: int = 1) -> List[Dict]:
        """Get all data fields for a specific region with pagination and caching"""
        try:
            # Check if we have cached data fields
            cache_key = f"{region}_{delay}"
            cache_file = f"data_fields_cache_{cache_key}.json"
            
            if os.path.exists(cache_file):
                logger.info(f"📁 Loading cached data fields for {region} delay={delay}")
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    logger.info(f"📊 Loaded {len(cached_data)} cached fields for {region} delay={delay}")
                    
                    # Filter by target dataset if specified
                    if self.target_dataset:
                        filtered_data = [field for field in cached_data if self.target_dataset in field.get('id', '')]
                        logger.info(f"🎯 Filtered to {len(filtered_data)} fields for dataset {self.target_dataset}")
                        return filtered_data
                    
                    return cached_data
            
            logger.info(f"🌐 No cache found for {region} delay={delay}, fetching from API...")
            config = self.regions[region]
            
            # First get available datasets from multiple categories
            categories = ['fundamental', 'analyst', 'model', 'news', 'alternative']
            all_dataset_ids = []
            
            for category in categories:
                try:
                    datasets_params = {
                        'category': category,
                        'delay': delay,
                        'instrumentType': 'EQUITY',
                        'region': region,
                        'universe': config.universe,
                        'limit': 20
                    }
                    
                    logger.info(f"📊 Getting {category} datasets for region {region}")
                    datasets_response = self.sess.get('https://api.worldquantbrain.com/data-sets', params=datasets_params)
                    
                    if datasets_response.status_code == 200:
                        datasets_data = datasets_response.json()
                        available_datasets = datasets_data.get('results', [])
                        category_dataset_ids = [ds.get('id') for ds in available_datasets if ds.get('id')]
                        all_dataset_ids.extend(category_dataset_ids)
                        logger.info(f"📊 Found {len(category_dataset_ids)} {category} datasets for region {region}")
                    else:
                        logger.warning(f"⚠️ Failed to get {category} datasets for region {region}: {datasets_response.status_code}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get datasets for category {category}: {e}")
            
            # Remove duplicates and use the combined list
            dataset_ids = list(set(all_dataset_ids))
            
            # Filter by target dataset if specified
            if self.target_dataset:
                dataset_ids = [ds for ds in dataset_ids if self.target_dataset in ds]
                logger.info(f"🎯 Filtered datasets to {len(dataset_ids)} for target dataset {self.target_dataset}")
                if not dataset_ids:
                    logger.warning(f"⚠️ No datasets found matching {self.target_dataset} for region {region}")
                    return []
            
            if not dataset_ids:
                logger.warning(f"⚠️ No datasets found for region {region}, using fallback datasets")
                dataset_ids = ['fundamental6', 'fundamental2', 'analyst4', 'model16', 'model51', 'news12']
            
            logger.info(f"📊 Total unique datasets for region {region}: {len(dataset_ids)}")
            
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
                        'limit': 50,  # 50 fields per page
                        'page': page
                    }
                    
                    response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params)
                    if response.status_code == 200:
                        data = response.json()
                        fields = data.get('results', [])
                        if not fields:  # No more fields on this page
                            break
                        dataset_fields.extend(fields)
                        logger.info(f"📄 Found {len(fields)} fields in dataset {dataset} page {page}")
                        page += 1
                    else:
                        logger.warning(f"⚠️ Failed to get fields from dataset {dataset} page {page}")
                        break
                
                all_fields.extend(dataset_fields)
                logger.info(f"📊 Total fields from dataset {dataset}: {len(dataset_fields)}")
            
            # Remove duplicates
            unique_fields = {field['id']: field for field in all_fields}.values()
            field_list = list(unique_fields)
            logger.info(f"📊 Total unique fields for region {region}: {len(field_list)} (from {max_datasets} datasets)")
            
            # Cache the fetched data
            try:
                with open(cache_file, 'w') as f:
                    json.dump(field_list, f, indent=2)
                logger.info(f"💾 Cached {len(field_list)} fields for {region} delay={delay}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cache data fields: {e}")
            
            return field_list
            
        except Exception as e:
            logger.error(f"❌ Failed to get data fields for region {region}: {e}")
            return []

    def get_sample_data_fields(self) -> List[str]:
        """Get sample data fields from all regions for prompt generation"""
        sample_fields = []
        
        # Try to get fields from different regions
        for region in ["USA", "EUR", "CHN"]:
            try:
                fields = self.get_data_fields_for_region(region)
                for field in fields[:5]:  # Take first 5 from each region
                    if field['id'] not in sample_fields:
                        sample_fields.append(field['id'])
                if len(sample_fields) >= 15:  # Enough samples
                    break
            except:
                continue
        
        # Fallback to common field names if no API data available
        if not sample_fields:
            sample_fields = [
                "close", "open", "high", "low", "volume", "returns", "vwap",
                "adv20", "adv5", "adv30", "adv60", "adv120", "adv180", "adv240",
                "market_cap", "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda"
            ]
        
        return sample_fields

    def operator_supports_event_inputs(self, operator_name: str) -> bool:
        """Check if an operator supports event inputs based on its definition"""
        # Known operators that don't support event inputs (based on WorldQuant Brain API errors)
        operators_no_event_support = {
            'ts_mean', 'ts_std', 'ts_rank', 'ts_argmax', 'ts_argmin', 'ts_sum', 
            'ts_min', 'ts_max', 'ts_median', 'ts_quantile', 'ts_skew', 'ts_kurt',
            'ts_delta', 'ts_delay', 'ts_lag', 'ts_corr', 'ts_cov', 'ts_regression',
            'ts_scale', 'ts_decay_linear', 'ts_decay_exponential', 'ts_ema',
            'ts_wma', 'ts_sma', 'ts_stddev', 'ts_variance', 'ts_zscore'
        }
        
        # Time series operators typically don't support event inputs
        if operator_name in operators_no_event_support:
            return False
        
        # Check operator definition for explicit event support
        for op in self.operators:
            if op['name'] == operator_name:
                definition = op.get('definition', '').lower()
                description = op.get('description', '').lower()
                # Check if the operator explicitly mentions event support
                if 'event' in definition or 'event' in description:
                    return True
                # Some operators explicitly don't support events
                if 'does not support event' in description or 'no event' in description:
                    return False
                # Default to supporting events for non-time-series operators
                return True
        return True  # Default to supporting events if operator not found

    def filter_data_fields_for_operator(self, data_fields: List[Dict], operator_name: str) -> List[Dict]:
        """Filter data fields based on operator event support"""
        if self.operator_supports_event_inputs(operator_name):
            return data_fields  # Return all fields if operator supports events
        
        # Filter out fields containing 'event' if operator doesn't support events
        filtered_fields = []
        for field in data_fields:
            field_id = field.get('id', '').lower()
            if 'event' not in field_id:
                filtered_fields.append(field)
        
        logger.info(f"🔍 Filtered data fields for operator '{operator_name}': {len(data_fields)} → {len(filtered_fields)} (removed {len(data_fields) - len(filtered_fields)} event fields)")
        return filtered_fields

    def is_vector_field_error(self, error_message: str) -> bool:
        """Check if error message indicates VECTOR field incompatibility or any field type incompatibility"""
        vector_error_patterns = [
            "VECTOR",
            "vector",
            "does not support VECTOR",
            "VECTOR field",
            "vector field",
            "VECTOR data",
            "vector data",
            "does not support event inputs",
            "event inputs",
            "event field",
            "event data"
        ]
        
        error_lower = error_message.lower()
        return any(pattern.lower() in error_lower for pattern in vector_error_patterns)

    def filter_vector_fields_for_template(self, data_fields: List[Dict], template: str) -> List[Dict]:
        """Filter out VECTOR fields for templates that have field type issues"""
        if template in self.templates_with_vector_issues:
            filtered_fields = []
            vector_count = 0
            for field in data_fields:
                field_type = field.get('type', '').upper()
                if field_type != 'VECTOR':
                    filtered_fields.append(field)
                else:
                    vector_count += 1
            
            logger.error(f"🚫🚫🚫 SKIPPING ALL VECTOR FIELDS for template '{template[:30]}...': {len(data_fields)} → {len(filtered_fields)} (removed {vector_count} VECTOR fields)")
            logger.error(f"🚫🚫🚫 This template has field type issues and will skip VECTOR fields for the rest of the run!")
            return filtered_fields
        
        return data_fields

    def extract_operators_from_template(self, template: str) -> List[str]:
        """Extract operator names from a template string"""
        operators_found = []
        for op in self.operators:
            if op['name'] in template:
                operators_found.append(op['name'])
        return operators_found

    def call_ollama_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call Ollama API to generate a single atom/light-weight template"""
        # Select a subset of operators for the prompt (limit to 5 for simpler templates)
        selected_operators = random.sample(self.operators, min(5, len(self.operators)))
        
        # Get some sample data fields for the prompt
        sample_data_fields = self.get_sample_data_fields()
        
        # Format operators for the prompt
        operators_desc = []
        for op in selected_operators:
            operators_desc.append(f"- {op['name']}: {op['description']} (Definition: {op['definition']})")
        
        # Format data fields for the prompt
        fields_desc = []
        for field in sample_data_fields[:10]:  # Show first 10 fields as examples
            fields_desc.append(f"- {field}")
        
        # Define JSON schema for structured output
        json_schema = {
            "type": "object",
            "properties": {
                "template": {
                    "type": "string",
                    "description": "A single alpha expression template"
                }
            },
            "required": ["template"],
            "additionalProperties": False
        }
        
        system_prompt = f"""You are an expert in quantitative finance and WorldQuant Brain alpha expressions. 
        Generate ONE simple, atomic alpha expression template that is lightweight and efficient. But if it is too orthodox, everybody would use it.
        Focus on basic mathematical operations and simple data field combinations.
        
        Available Operators (USE ONLY THESE - MAX 5 OPERATORS):
        {chr(10).join(operators_desc)}
        
        Available Data Fields (USE ONLY THESE - These are REAL field names from WorldQuant Brain):
        {chr(10).join(fields_desc)}
        
        CRITICAL RULES:
        1. Use ONLY the operators listed above - do NOT invent new operators
        2. Use ONLY the data field names listed above - do NOT invent field names
        3. Use MAXIMUM 5 operators in your template - keep it simple and atomic
        4. Use proper function syntax: operator(field_name, parameter) or operator(field1, field2, parameter)
        5. NO comparison operators like >, <, >=, <=, ==, !=, &&, ||, %
        6. Use realistic parameter values (e.g., 20, 60, 120 for time periods)
        7. Focus on atomic, lightweight expressions with minimal complexity
        8. Field names must match EXACTLY as shown in the Available Data Fields list
        
        ALWAYS respond with valid JSON format containing a single template.
        Example format: {{"template": "ts_rank(ts_delta(close, 1), 20)"}}"""
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 Ollama API call attempt {attempt + 1}/{max_retries}")
                
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
                        "num_predict": 500  # Shorter for atomic templates
                    }
                )
                
                content = response['message']['content']
                logger.info("✅ Ollama API call successful")
                
                # Parse and validate structured output
                try:
                    parsed_json = json.loads(content)
                    if 'template' in parsed_json and isinstance(parsed_json['template'], str):
                        logger.info(f"✅ Structured output validation successful")
                        return parsed_json['template']
                    else:
                        logger.error("Invalid structured output: missing 'template' key or not a string")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return None
                except json.JSONDecodeError as e:
                    logger.error(f"Structured output failed JSON validation: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Ollama API call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None

    def generate_multiple_templates(self, count: int = 4) -> List[str]:
        """Generate multiple atomic templates using Ollama with validation"""
        prompts = [
            "Generate a simple alpha expression using basic price data (open, high, low, close)",
            "Create a lightweight alpha using volume and price data",
            "Generate a simple momentum alpha expression",
            "Create a basic mean reversion alpha",
            "Generate a simple volatility-based alpha",
            "Create a basic technical indicator alpha",
            "Generate a simple cross-sectional alpha",
            "Create a basic time-series alpha",
            "Generate a simple correlation-based alpha",
            "Create a basic regression alpha",
            "Generate a simple factor-based alpha",
            "Create a basic statistical alpha"
        ]
        
        templates = []
        selected_prompts = random.sample(prompts, min(count, len(prompts)))
        
        logger.info(f"🎯 Generating {count} atomic templates with validation...")
        
        for i, prompt in enumerate(selected_prompts):
            logger.info(f"📝 Template {i+1}/{count}: {prompt}")
            template = self.call_ollama_api(prompt)
            
            if template:
                # Validate template
                if self.validate_template_with_test_data(template):
                    templates.append(template)
                    logger.info(f"✅ Generated and validated template {i+1}: {template}")
                else:
                    # Try to regenerate
                    logger.warning(f"⚠️ Template {i+1} failed validation, attempting regeneration...")
                    regenerated = self.regenerate_template_if_needed(template, 0)
                    if regenerated:
                        templates.append(regenerated)
                        logger.info(f"✅ Regenerated valid template {i+1}: {regenerated}")
                    else:
                        logger.error(f"❌ Failed to generate valid template {i+1}")
            else:
                logger.error(f"❌ Failed to generate template {i+1}")
        
        logger.info(f"🎯 Generated {len(templates)}/{count} validated templates successfully")
        return templates

    def _validate_data_field_exists(self, data_field: str, region: str) -> bool:
        """Check if a data field exists in the region's available fields"""
        try:
            # Get data fields for the region
            region_fields = self.get_data_fields_for_region(region)
            
            # Check if the data field exists
            for field in region_fields:
                if field.get('id') == data_field:
                    return True
            
            logger.warning(f"⚠️ Data field '{data_field}' not found in region {region}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error validating data field '{data_field}': {e}")
            return False

    def _process_template(self, template: str, data_field: str) -> str:
        """Process template to substitute data field placeholders"""
        # Common placeholders that need to be replaced
        placeholders = {
            'x': data_field,
            'high': data_field,
            'low': data_field,
            'close': data_field,
            'open': data_field,
            'volume': data_field,
            'returns': data_field,
            'DATA_FIELD': data_field,
            'DATA_FIELD1': data_field,
            'DATA_FIELD2': data_field,
            'DATA_FIELD3': data_field,
            'DATA_FIELD4': data_field
        }
        
        processed = template
        for placeholder, actual_field in placeholders.items():
            processed = processed.replace(placeholder, actual_field)
        
        return processed

    def _regenerate_template_on_error(self, failed_template: str, error_message: str, region: str, data_field: str) -> str:
        """Regenerate a template using Ollama when simulation fails, feeding error message back"""
        try:
            # Get available operators for the prompt (limit to 5 for simpler templates)
            operators_desc = []
            for op in self.operators[:5]:  # Use first 5 operators
                operators_desc.append(f"- {op['name']}: {op.get('description', 'No description')}")
            
            # Get sample data fields for the prompt
            sample_data_fields = self.get_sample_data_fields()
            fields_desc = []
            for field in sample_data_fields[:10]:  # Show first 10 fields as examples
                fields_desc.append(f"- {field}")
            
            # Get region-specific neutralization options
            region_neutralizations = self.regions[region].neutralization_options
            
            # Create error-aware prompt
            prompt = f"""The previous alpha expression template failed with this error:
ERROR: {error_message}

FAILED TEMPLATE: {failed_template}

Available Operators (USE ONLY THESE - MAX 5 OPERATORS):
{chr(10).join(operators_desc)}

Available Data Fields (USE ONLY THESE - These are REAL field names from WorldQuant Brain):
{chr(10).join(fields_desc)}

Please generate a NEW alpha expression template that:
1. Avoids the error that caused the previous template to fail
2. Uses proper WorldQuant Brain syntax
3. Uses the data field: {data_field}
4. Is suitable for region: {region}
5. Uses only the operators listed above
6. Uses only the data field names listed above
7. Note: Region {region} supports these neutralization options: {', '.join(region_neutralizations)}

Requirements:
- Must be a complete, valid alpha expression
- Avoid the specific error mentioned above
- Use proper syntax and parentheses
- Use ONLY the operators and data fields listed above
- Return ONLY the alpha expression, no explanations

Generate a new template:"""

            logger.info(f"🤖 Regenerating template due to error: {error_message[:200]}...")
            logger.info(f"🔍 Full error details: {error_message}")
            
            # Call Ollama to regenerate template
            response = self.call_ollama_api(prompt)
            if response:
                # Extract template from response (remove any JSON wrapper if present)
                new_template = response.strip()
                if new_template.startswith('{'):
                    try:
                        data = json.loads(new_template)
                        if 'template' in data:
                            new_template = data['template']
                        elif 'templates' in data and data['templates']:
                            new_template = data['templates'][0]
                    except:
                        pass
                
                # Clean up the template
                new_template = new_template.strip().strip('"').strip("'")
                logger.info(f"🔄 Generated new template: {new_template}")
                return new_template
            else:
                logger.warning("⚠️ Failed to get response from Ollama, using fallback template")
                return f"ts_rank({data_field}, 20)"  # Simple fallback
                
        except Exception as e:
            logger.error(f"❌ Failed to regenerate template: {e}")
            return f"ts_rank({data_field}, 20)"  # Simple fallback

    def simulate_template(self, template: str, region: str, data_field: str, neutralization: str, max_retries: int = 3, use_ollama: bool = True) -> BruteforceResult:
        """Simulate a single template with retry mechanism that regenerates templates on failure"""
        start_time = time.time()
        current_template = template
        
        # Check if this template has field type issues and skip VECTOR fields
        if template in self.templates_with_vector_issues:
            # Get the data field type to check if it's VECTOR
            try:
                region_fields = self.get_data_fields_for_region(region)
                for field in region_fields:
                    if field.get('id') == data_field:
                        field_type = field.get('type', '').upper()
                        if field_type == 'VECTOR':
                            logger.error(f"🚫🚫🚫 SKIPPING VECTOR FIELD '{data_field}' for template with field type issues!")
                            logger.error(f"🚫🚫🚫 This template will skip ALL VECTOR fields for the rest of the run!")
                            return BruteforceResult(
                                template=template,
                                region=region,
                                data_field=data_field,
                                neutralization=neutralization,
                                success=False,
                                error_message=f"Skipped VECTOR field '{data_field}' due to template field type issues",
                                simulation_time=0.0
                            )
                        break
            except Exception as e:
                logger.warning(f"⚠️ Could not check field type for {data_field}: {e}")
        
        # Validate data field exists before simulation
        if not self._validate_data_field_exists(data_field, region):
            logger.info(f"⏭️ Skipping simulation - data field '{data_field}' not found in region {region}")
            return BruteforceResult(
                template=template,
                region=region,
                data_field=data_field,
                neutralization=neutralization,
                success=False,
                error_message=f"Data field '{data_field}' not found in region {region}",
                simulation_time=0.0
            )
        
        for attempt in range(max_retries + 1):
            try:
                # Process template to substitute data field
                processed_template = self._process_template(current_template, data_field)
                logger.info(f"🔄 Template processing (attempt {attempt + 1}): {current_template} -> {processed_template}")
                
                # Create simulation settings
                settings = SimulationSettings(
                    region=region,
                    universe=self.regions[region].universe,
                    delay=1,
                    neutralization=neutralization,
                    maxTrade="ON" if self.regions[region].max_trade else "OFF"
                )
                
                # Prepare simulation data with proper format (matching consultant-templates-ollama)
                simulation_data = {
                    'type': 'REGULAR',
                    'settings': asdict(settings),
                    'regular': processed_template
                }
                
                # Submit simulation using the correct API endpoint
                submit_url = "https://api.worldquantbrain.com/simulations"
                response = self.sess.post(submit_url, json=simulation_data)
                
                if response.status_code != 201:
                    error_message = f"Failed to submit simulation: {response.status_code}"
                    logger.error(f"❌ Simulation submission failed: {response.status_code} - {response.text}")
                    if attempt < max_retries:
                        logger.info(f"🔄 Attempt {attempt + 1} failed, regenerating template...")
                        current_template = self._regenerate_template_on_error(current_template, error_message, region, data_field)
                        continue
                    return BruteforceResult(
                        template=current_template,
                        region=region,
                        data_field=data_field,
                        neutralization=neutralization,
                        success=False,
                        error_message=error_message,
                        simulation_time=time.time() - start_time
                    )
                
                # Get progress URL from Location header
                progress_url = response.headers.get('Location')
                if not progress_url:
                    error_message = "No Location header in response"
                    logger.error(f"❌ No Location header in response")
                    if attempt < max_retries:
                        logger.info(f"🔄 Attempt {attempt + 1} failed, regenerating template...")
                        current_template = self._regenerate_template_on_error(current_template, error_message, region, data_field)
                        continue
                    return BruteforceResult(
                        template=current_template,
                        region=region,
                        data_field=data_field,
                        neutralization=neutralization,
                        success=False,
                        error_message=error_message,
                        simulation_time=time.time() - start_time
                    )
                
                logger.info(f"🚀 Started simulation for template: {current_template[:50]}... (Progress URL: {progress_url})")
                
                # Monitor simulation
                result = self._monitor_simulation(progress_url, current_template, region, data_field, neutralization)
                result.simulation_time = time.time() - start_time
                
                # If simulation failed, try to regenerate template (only if Ollama is enabled)
                if not result.success and attempt < max_retries:
                    if use_ollama:
                        logger.info(f"🔄 Simulation failed (attempt {attempt + 1}), regenerating template...")
                        current_template = self._regenerate_template_on_error(current_template, result.error_message, region, data_field)
                        continue
                    else:
                        logger.info(f"🔄 Simulation failed (attempt {attempt + 1}), no regeneration for custom templates...")
                        continue
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Simulation failed for template {current_template[:50]}...: {e}")
                if attempt < max_retries:
                    logger.info(f"🔄 Attempt {attempt + 1} failed, regenerating template...")
                    current_template = self._regenerate_template_on_error(current_template, str(e), region, data_field)
                    continue
                return BruteforceResult(
                    template=current_template,
                    region=region,
                    data_field=data_field,
                    neutralization=neutralization,
                    success=False,
                    error_message=str(e),
                    simulation_time=time.time() - start_time
                )
        
        # If we get here, all retries failed
        return BruteforceResult(
            template=current_template,
            region=region,
            data_field=data_field,
            neutralization=neutralization,
            success=False,
            error_message=f"All {max_retries + 1} attempts failed",
            simulation_time=time.time() - start_time
        )

    def _monitor_simulation(self, progress_url: str, template: str, region: str, data_field: str, neutralization: str) -> BruteforceResult:
        """Monitor a simulation until completion using progress URL"""
        max_wait_time = 300  # 5 minutes
        check_interval = 5  # Check every 5 seconds
        start_time = time.time()
        check_count = 0
        
        logger.info(f"🎮 MONITORING: Starting to monitor simulation progress (max {max_wait_time}s)")
        
        while time.time() - start_time < max_wait_time:
            try:
                check_count += 1
                elapsed = time.time() - start_time
                logger.info(f"🎮 MONITORING: Check #{check_count} (elapsed: {elapsed:.1f}s)")
                
                response = self.sess.get(progress_url)
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
                            return BruteforceResult(
                                template=template,
                                region=region,
                                data_field=data_field,
                                neutralization=neutralization,
                                success=False,
                                error_message="No alphaId in simulation response"
                            )
                        
                        # Fetch the alpha data using the alphaId
                        logger.info(f"Simulation complete, fetching alpha {alpha_id}")
                        alpha_response = self.sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}')
                        
                        if alpha_response.status_code == 200:
                            alpha_data = alpha_response.json()
                            
                            # Check if alpha_data is valid
                            if not isinstance(alpha_data, dict):
                                logger.error(f"❌ Invalid alpha data format: {type(alpha_data)}")
                                return BruteforceResult(
                                    template=template,
                                    region=region,
                                    data_field=data_field,
                                    neutralization=neutralization,
                                    success=False,
                                    error_message=f"Invalid alpha data format: {type(alpha_data)}"
                                )
                            
                            # Log the alpha data structure for debugging
                            logger.debug(f"Alpha data structure: {list(alpha_data.keys())}")
                            
                            # Extract performance metrics
                            sharpe = alpha_data.get('sharpe', 0.0)
                            returns = alpha_data.get('returns', 0.0)
                            max_drawdown = alpha_data.get('maxDrawdown', 0.0)
                            margin = alpha_data.get('margin', 0.0)
                            fitness = alpha_data.get('fitness', 0.0)
                            turnover = alpha_data.get('turnover', 0.0)
                            
                            # Check if we have valid performance data
                            # If all key metrics are 0.0, it likely means the simulation didn't complete properly
                            has_valid_data = any([
                                sharpe != 0.0,
                                returns != 0.0,
                                margin != 0.0,
                                fitness != 0.0,
                                turnover != 0.0
                            ])
                            
                            # Additional check: ensure we have at least some meaningful data
                            # A successful simulation should have at least sharpe or returns > 0
                            has_meaningful_data = sharpe != 0.0 or returns != 0.0
                            
                            if not has_valid_data or not has_meaningful_data:
                                logger.warning(f"⚠️ Simulation completed but returned zero/invalid metrics - treating as failure")
                                logger.warning(f"   Sharpe: {sharpe}, Returns: {returns}, Margin: {margin}, Fitness: {fitness}")
                                logger.warning(f"   Alpha data keys: {list(alpha_data.keys())}")
                                return BruteforceResult(
                                    template=template,
                                    region=region,
                                    data_field=data_field,
                                    neutralization=neutralization,
                                    success=False,
                                    error_message="Simulation completed but returned zero/invalid performance metrics"
                                )
                            
                            return BruteforceResult(
                                template=template,
                                region=region,
                                data_field=data_field,
                                neutralization=neutralization,
                                success=True,
                                sharpe=sharpe,
                                returns=returns,
                                max_drawdown=max_drawdown,
                                margin=margin,
                                fitness=fitness,
                                turnover=turnover
                            )
                        else:
                            logger.error(f"Failed to fetch alpha data: {alpha_response.status_code}")
                            return BruteforceResult(
                                template=template,
                                region=region,
                                data_field=data_field,
                                neutralization=neutralization,
                                success=False,
                                error_message=f"Failed to fetch alpha data: {alpha_response.status_code}"
                            )
                    
                    elif status in ['FAILED', 'ERROR', 'WARNING']:
                        # Get detailed error information
                        if status == 'WARNING':
                            error_msg = data.get('error', f'Simulation warning: {data.get("message", "Unknown warning")}')
                        else:
                            error_msg = data.get('error', f'Simulation {status.lower()}')
                        
                        # Also check the 'message' field for error details
                        message_error = data.get('message', '')
                        if message_error:
                            error_msg = message_error
                        
                        error_details = data.get('errorDetails', '')
                        error_type = data.get('errorType', '')
                        error_code = data.get('errorCode', '')
                        
                        # Combine all error information
                        full_error = f"{error_msg}"
                        if error_details:
                            full_error += f" | Details: {error_details}"
                        if error_type:
                            full_error += f" | Type: {error_type}"
                        if error_code:
                            full_error += f" | Code: {error_code}"
                        
                        # Check if this is a field type error (VECTOR, event, etc.)
                        if self.is_vector_field_error(full_error):
                            logger.error(f"🚫🚫🚫 FIELD TYPE ERROR DETECTED! Template '{template[:30]}...' will skip ALL VECTOR fields for the rest of the run!")
                            logger.error(f"🚫🚫🚫 ERROR: {full_error}")
                            logger.error(f"🚫🚫🚫 This template is now marked to skip VECTOR fields permanently!")
                            self.templates_with_vector_issues.add(template)
                        
                        # Log the full error details
                        if status == 'WARNING':
                            logger.warning(f"⚠️ Simulation warning (treated as failure): {full_error}")
                        else:
                            logger.error(f"❌ Simulation {status.lower()}: {full_error}")
                        logger.error(f"❌ Full error data: {data}")
                        
                        return BruteforceResult(
                            template=template,
                            region=region,
                            data_field=data_field,
                            neutralization=neutralization,
                            success=False,
                            error_message=full_error
                        )
                    
                    # Still running, wait and check again
                    time.sleep(check_interval)
                else:
                    logger.warning(f"⚠️ Status check failed: {response.status_code}")
                    time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"❌ Error monitoring simulation: {e}")
                return BruteforceResult(
                    template=template,
                    region=region,
                    data_field=data_field,
                    neutralization=neutralization,
                    success=False,
                    error_message=f"Monitoring error: {str(e)}"
                )
        
        # Timeout
        logger.warning(f"⏰ Simulation timeout after {max_wait_time}s")
        return BruteforceResult(
            template=template,
            region=region,
            data_field=data_field,
            neutralization=neutralization,
            success=False,
            error_message="Simulation timeout"
        )

    def bruteforce_multiple_templates(self, templates: List[str], use_ollama: bool = True) -> List[BruteforceResult]:
        """Bruteforce test multiple templates with region-specific field matching"""
        logger.info(f"🎯 Starting bruteforce test for {len(templates)} templates")
        
        all_results = []
        
        # For custom templates, we need to match templates to their correct regions
        # Since templates are now generated per region, we need to determine which region each template belongs to
        region_order = ["USA", "EUR", "ASI", "CHN", "GLB"]
        
        # Create simulation tasks for each template with region-specific matching
        simulation_tasks = []
        for i, template in enumerate(templates):
            logger.info(f"📝 Template {i+1}: {template}")
            
            # Determine which region this template belongs to by checking which region has the data field
            template_region = None
            for region in region_order:
                if region in self.regions:
                    try:
                        region_fields = self.get_data_fields_for_region(region)
                        # Check if any field in this template exists in this region
                        for field in region_fields:
                            if field['id'] in template:
                                template_region = region
                                break
                        if template_region:
                            break
                    except:
                        continue
            
            if not template_region:
                logger.warning(f"⚠️ Could not determine region for template: {template}")
                continue
            
            logger.info(f"🎯 Template {i+1} belongs to region: {template_region}")
            
            # Get data fields for this specific region
            try:
                data_fields = self.get_data_fields_for_region(template_region)
                logger.info(f"📊 Region {template_region}: {len(data_fields)} data fields")
                
                # Extract operators from template and filter data fields accordingly
                template_ops = self.extract_operators_from_template(template)
                logger.info(f"🔍 Template uses operators: {template_ops}")
                
                # Filter data fields based on operators in this template
                if template_ops:
                    # Check if any operator in the template doesn't support events
                    needs_event_filtering = False
                    for op in template_ops:
                        if not self.operator_supports_event_inputs(op):
                            needs_event_filtering = True
                            break
                    
                    if needs_event_filtering:
                        # Apply filtering using the first operator found that doesn't support events
                        data_fields = self.filter_data_fields_for_operator(data_fields, template_ops[0])
                
                # Filter out VECTOR fields if template has VECTOR field issues
                data_fields = self.filter_vector_fields_for_template(data_fields, template)
                
                # Create tasks for this template in its specific region
                region_tasks = []
                for data_field in data_fields:
                    # Skip VECTOR fields if template has field type issues
                    if template in self.templates_with_vector_issues:
                        field_type = data_field.get('type', '').upper()
                        if field_type == 'VECTOR':
                            logger.error(f"🚫🚫🚫 SKIPPING VECTOR FIELD '{data_field['id']}' - template has field type issues!")
                            continue
                    
                    for neutralization in self.regions[template_region].neutralization_options:
                        region_tasks.append((template, template_region, data_field['id'], neutralization))
                
                # Split tasks into 2 subprocesses
                mid_point = len(region_tasks) // 2
                subprocess_1_tasks = region_tasks[:mid_point]
                subprocess_2_tasks = region_tasks[mid_point:]
                
                simulation_tasks.append({
                    'template': template,
                    'template_id': i + 1,
                    'region': template_region,
                    'subprocess_1': subprocess_1_tasks,
                    'subprocess_2': subprocess_2_tasks
                })
                
                logger.info(f"📊 Template {i+1} ({template_region}): {len(subprocess_1_tasks)} tasks for subprocess 1, {len(subprocess_2_tasks)} tasks for subprocess 2")
                
            except Exception as e:
                logger.error(f"❌ Failed to get data fields for region {template_region}: {e}")
                continue
        
        logger.info(f"🚀 Created {len(simulation_tasks)} template groups with 2 subprocesses each")
        
        # Execute simulations with thread management
        self._execute_multiple_templates_concurrent(simulation_tasks, all_results, use_ollama)
        
        # Save final progress after all templates
        self.save_progress()
        
        return all_results

    def _execute_multiple_templates_concurrent(self, simulation_tasks: List[Dict], all_results: List[BruteforceResult], use_ollama: bool = True):
        """Execute multiple templates with 2 subprocesses each using thread management"""
        logger.info(f"🚀 Starting concurrent execution of {len(simulation_tasks)} templates with 2 subprocesses each")
        
        # Create futures for all subprocesses
        all_futures = []
        
        for template_group in simulation_tasks:
            template = template_group['template']
            template_id = template_group['template_id']
            
            # Submit subprocess 1
            if template_group['subprocess_1']:
                future_1 = self.executor.submit(
                    self._execute_subprocess_tasks,
                    template_group['subprocess_1'],
                    f"Template_{template_id}_Subprocess_1",
                    use_ollama
                )
                all_futures.append(future_1)
                logger.info(f"🚀 Started subprocess 1 for template {template_id}: {template[:30]}...")
            
            # Submit subprocess 2
            if template_group['subprocess_2']:
                future_2 = self.executor.submit(
                    self._execute_subprocess_tasks,
                    template_group['subprocess_2'],
                    f"Template_{template_id}_Subprocess_2",
                    use_ollama
                )
                all_futures.append(future_2)
                logger.info(f"🚀 Started subprocess 2 for template {template_id}: {template[:30]}...")
        
        # Wait for all futures to complete
        logger.info(f"⏳ Waiting for {len(all_futures)} subprocesses to complete...")
        
        for future in as_completed(all_futures):
            try:
                subprocess_results = future.result()
                all_results.extend(subprocess_results)
                logger.info(f"✅ Subprocess completed with {len(subprocess_results)} results")
            except Exception as e:
                logger.error(f"❌ Subprocess failed: {e}")
        
        logger.info(f"✅ Completed all {len(simulation_tasks)} templates with 2 subprocesses each")

    def _execute_subprocess_tasks(self, tasks: List[Tuple], subprocess_name: str, use_ollama: bool = True) -> List[BruteforceResult]:
        """Execute a batch of tasks for a subprocess"""
        results = []
        logger.info(f"🔄 {subprocess_name}: Starting {len(tasks)} tasks")
        
        for i, (template, region, data_field, neutralization) in enumerate(tasks):
            try:
                result = self.simulate_template(template, region, data_field, neutralization, use_ollama=use_ollama)
                results.append(result)
                
                # Save progress after each simulation (only successful ones)
                if result.success:
                    self.results.append(result)
                    self.save_progress()
                
                if result.success:
                    logger.info(f"✅ {subprocess_name}: Task {i+1}/{len(tasks)} - {template[:30]}... in {region} (Sharpe: {result.sharpe:.3f})")
                    
                    # Check if result meets success criteria for neutralization expansion
                    if self.check_success_criteria(result):
                        logger.info(f"🏆 SUCCESS CRITERIA MET! Expanding neutralization options for {template[:30]}...")
                        expansion_results = self._expand_neutralization_options(template, region, data_field, use_ollama)
                        results.extend(expansion_results)
                        
                        # Save progress after neutralization expansion
                        self.results.extend(expansion_results)
                        self.save_progress()
                else:
                    logger.info(f"❌ {subprocess_name}: Task {i+1}/{len(tasks)} - {template[:30]}... in {region} - {result.error_message}")
                    
            except Exception as e:
                logger.error(f"❌ {subprocess_name}: Task {i+1}/{len(tasks)} failed: {e}")
                error_result = BruteforceResult(
                    template=template,
                    region=region,
                    data_field=data_field,
                    neutralization=neutralization,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
                
                # Don't save failed simulations to progress file
                # self.results.append(error_result)  # Removed - only save successful simulations
                # self.save_progress()  # Removed - only save successful simulations
        
        logger.info(f"🏁 {subprocess_name}: Completed {len(tasks)} tasks with {len([r for r in results if r.success])} successes")
        return results
    
    def _expand_neutralization_options(self, template: str, region: str, data_field: str, use_ollama: bool = True) -> List[BruteforceResult]:
        """Run all neutralization options for successful templates"""
        expansion_results = []
        all_neutralizations = self.regions[region].neutralization_options
        
        logger.info(f"🔄 Expanding neutralization options for {template[:30]}... in {region}")
        
        for neutralization in all_neutralizations:
            try:
                result = self.simulate_template(template, region, data_field, neutralization, use_ollama=use_ollama)
                expansion_results.append(result)
                
                # Save progress after each neutralization expansion simulation (only successful ones)
                if result.success:
                    self.results.append(result)
                    self.save_progress()
                
                if result.success:
                    logger.info(f"✅ Neutralization expansion: {neutralization} - Sharpe: {result.sharpe:.3f}")
                else:
                    logger.info(f"❌ Neutralization expansion: {neutralization} - {result.error_message}")
                    
            except Exception as e:
                logger.error(f"❌ Neutralization expansion failed for {neutralization}: {e}")
                error_result = BruteforceResult(
                    template=template,
                    region=region,
                    data_field=data_field,
                    neutralization=neutralization,
                    success=False,
                    error_message=str(e)
                )
                expansion_results.append(error_result)
                
                # Don't save failed neutralization expansion to progress file
                # self.results.append(error_result)  # Removed - only save successful simulations
                # self.save_progress()  # Removed - only save successful simulations
        
        logger.info(f"🏁 Neutralization expansion completed: {len(expansion_results)} additional tests")
        return expansion_results

    def load_custom_templates(self, custom_template_file: str = "custom_templates.json") -> List[str]:
        """Load custom templates from JSON file and generate variations with anl/fnd data fields"""
        try:
            with open(custom_template_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            templates = []
            for template_data in data.get('templates', []):
                template_pattern = template_data['template']
                data_field_types = template_data.get('data_field_types', ['anl', 'fnd'])
                
                # Generate templates for each region separately to ensure field-region matching
                for region in ["USA", "EUR", "ASI", "CHN", "GLB"]:
                    try:
                        region_fields = self.get_data_fields_for_region(region)
                        anl_fnd_fields = []
                        
                        for field in region_fields:
                            field_id = field.get('id', '').lower()
                            if any(prefix in field_id for prefix in data_field_types):
                                anl_fnd_fields.append(field['id'])
                        
                        logger.info(f"📊 Found {len(anl_fnd_fields)} {data_field_types} fields for region {region}")
                        
                        # Generate template variations with region-specific data fields
                        for field in anl_fnd_fields[:2]:  # Limit to first 2 fields per region to avoid too many combinations
                            custom_template = template_pattern.replace('{data_field}', field)
                            templates.append(custom_template)
                            logger.info(f"🔧 Generated custom template for {region}: {custom_template[:50]}...")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to get fields for region {region}: {e}")
                        continue
            
            return templates
            
        except Exception as e:
            logger.error(f"❌ Failed to load custom templates: {e}")
            return []

    def load_nws77_template(self, template_file: str = "nws77_template.json") -> List[str]:
        """Load and process the nws77 template with variable substitution"""
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            base_template = template_data['template']
            variables = template_data['variables']
            
            # Get all possible combinations
            dataprocess_options = variables['dataprocess_op']['options']
            ts_statistical_options = variables['ts_Statistical_op']['options']
            
            templates = []
            
            # Generate all combinations of the template variables
            for dataprocess_op in dataprocess_options:
                for ts_statistical_op in ts_statistical_options:
                    # Replace variable placeholders
                    template = base_template.replace('<dataprocess_op/>', dataprocess_op)
                    template = template.replace('<ts_Statistical_op/>', ts_statistical_op)
                    template = template.replace('<nws77/>', 'nws77_sentiment_impact_projection')  # Use the specific field
                    
                    templates.append(template)
                    logger.info(f"🔧 Generated nws77 template: {template[:50]}...")
            
            logger.info(f"📊 Generated {len(templates)} nws77 template variations")
            return templates
            
        except Exception as e:
            logger.error(f"❌ Failed to load nws77 template: {e}")
            return []

    def _execute_simulations_concurrent(self, simulation_tasks: List[Tuple], all_results: List[BruteforceResult]):
        """Execute simulations concurrently using thread management (legacy method)"""
        task_index = 0
        
        while task_index < len(simulation_tasks) or self.active_futures:
            # Fill available slots
            while len(self.active_futures) < self.max_concurrent and task_index < len(simulation_tasks):
                template, region, data_field, neutralization = simulation_tasks[task_index]
                
                future = self.executor.submit(
                    self.simulate_template, 
                    template, region, data_field, neutralization
                )
                
                future_id = f"sim_{task_index}_{int(time.time() * 1000)}"
                self.active_futures[future_id] = future
                self.future_start_times[future_id] = time.time()
                
                logger.info(f"🚀 Started simulation {future_id}: {template[:30]}... in {region}")
                task_index += 1
            
            # Process completed futures
            self._process_completed_futures(all_results)
            
            # Wait a bit before next iteration
            time.sleep(1)
        
        logger.info(f"✅ Completed all {len(simulation_tasks)} simulations")

    def _process_completed_futures(self, all_results: List[BruteforceResult]):
        """Process completed futures and update results"""
        completed_futures = []
        timed_out_futures = []
        current_time = time.time()
        
        for future_id, future in self.active_futures.items():
            # Check for timeout
            start_time = self.future_start_times.get(future_id, current_time)
            elapsed_time = current_time - start_time
            
            if elapsed_time > self.future_timeout:
                timed_out_futures.append(future_id)
                logger.warning(f"⏰ TIMEOUT: Future {future_id} has been running for {elapsed_time:.1f}s")
                continue
            
            if future.done():
                completed_futures.append(future_id)
                try:
                    result = future.result()
                    all_results.append(result)
                    self.completed_count += 1
                    
                    if result.success:
                        self.successful_count += 1
                        logger.info(f"✅ SUCCESS: {result.template[:30]}... in {result.region} (Sharpe: {result.sharpe:.3f})")
                    else:
                        self.failed_count += 1
                        logger.info(f"❌ FAILED: {result.template[:30]}... in {result.region} - {result.error_message}")
                        
                except Exception as e:
                    self.failed_count += 1
                    logger.error(f"❌ ERROR processing future {future_id}: {e}")
        
        # Remove completed and timed out futures
        for future_id in completed_futures + timed_out_futures:
            if future_id in self.active_futures:
                del self.active_futures[future_id]
            if future_id in self.future_start_times:
                del self.future_start_times[future_id]

    def load_custom_template(self, template_file: str) -> Optional[str]:
        """Load a custom template from JSON file"""
        try:
            with open(template_file, 'r') as f:
                data = json.load(f)
            
            if 'template' in data:
                template = data['template']
                logger.info(f"✅ Loaded custom template: {template}")
                return template
            else:
                logger.error("❌ Template file missing 'template' field")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load custom template: {e}")
            return None

    def save_results(self, results: List[BruteforceResult]):
        """Save results to JSON file"""
        try:
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'total_simulations': len(results),
                'successful_simulations': len([r for r in results if r.success]),
                'failed_simulations': len([r for r in results if not r.success]),
                'results': [asdict(r) for r in results]
            }
            
            with open(self.results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"💾 Saved {len(results)} results to {self.results_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

    def run_bruteforce(self, custom_template_file: str = None, max_batches: int = 3, resume: bool = False):
        """Run the bruteforce template testing with 4 templates per batch and 2 subprocesses each"""
        logger.info("🚀 Starting Bruteforce Template Generator (4 templates per batch, 2 subprocesses each)")
        
        # Load existing progress if resuming
        if resume:
            if self.load_progress():
                logger.info("📁 Resumed from previous progress")
                # Start from the next batch if we were in the middle of one
                start_batch = self.current_batch
                if start_batch > 0:
                    logger.info(f"📦 Resuming from batch {start_batch + 1}")
            else:
                logger.info("📁 Starting fresh")
                start_batch = 0
        else:
            start_batch = 0
        
        all_results = self.results.copy()  # Start with existing results if resuming
        
        if custom_template_file:
            # Test custom templates (multiple templates mode) - NO OLLAMA NEEDED
            if custom_template_file == "custom_templates.json":
                # Load custom templates with anl/fnd field variations
                templates = self.load_custom_templates(custom_template_file)
                if templates:
                    logger.info(f"🎯 Testing {len(templates)} custom templates with anl/fnd fields (NO OLLAMA)")
                    logger.info("🚀 Starting direct bruteforce testing without AI generation...")
                    # Direct bruteforce without any AI generation
                    results = self.bruteforce_multiple_templates(templates, use_ollama=False)
                    all_results.extend(results)
                    self.results = all_results
                    self.save_results(all_results)
                else:
                    logger.error("❌ No custom templates loaded")
            elif custom_template_file == "nws77_template.json":
                # Load nws77 template with variable substitution
                templates = self.load_nws77_template(custom_template_file)
                if templates:
                    logger.info(f"🎯 Testing {len(templates)} nws77 template variations (NO OLLAMA)")
                    logger.info("🚀 Starting direct bruteforce testing without AI generation...")
                    # Direct bruteforce without any AI generation
                    results = self.bruteforce_multiple_templates(templates, use_ollama=False)
                    all_results.extend(results)
                    self.results = all_results
                    self.save_results(all_results)
                else:
                    logger.error("❌ No nws77 templates loaded")
            else:
                # Test single custom template (legacy mode)
                template = self.load_custom_template(custom_template_file)
                if template:
                    logger.info(f"🎯 Testing custom template: {template} (NO OLLAMA)")
                    logger.info("🚀 Starting direct bruteforce testing without AI generation...")
                    # Direct bruteforce without any AI generation
                    results = self.bruteforce_multiple_templates([template], use_ollama=False)
                    all_results.extend(results)
                    self.results = all_results
                    self.save_results(all_results)
        else:
            # Generate and test multiple batches of 4 templates each
            for batch in range(start_batch, max_batches):
                self.current_batch = batch
                logger.info(f"\n🔄 === BATCH {batch+1}/{max_batches} (4 templates, 2 subprocesses each) ===")
                
                # Generate 4 templates
                templates = self.generate_multiple_templates(4)
                if templates:
                    logger.info(f"🎯 Generated {len(templates)} templates for batch {batch+1}")
                    
                    # Save progress after template generation
                    self.save_progress()
                    
                    results = self.bruteforce_multiple_templates(templates)
                    all_results.extend(results)
                    self.results = all_results
                    
                    # Save progress after each batch
                    self.save_progress()
                    self.save_results(all_results)
                    
                    # Check for success criteria met
                    success_criteria_met = len([r for r in all_results if self.check_success_criteria(r)])
                    logger.info(f"🏆 Success criteria met: {success_criteria_met} results")
                else:
                    logger.error(f"❌ Failed to generate templates for batch {batch+1}")
        
        # Final summary
        successful = len([r for r in all_results if r.success])
        failed = len([r for r in all_results if not r.success])
        success_criteria_met = len([r for r in all_results if self.check_success_criteria(r)])
        
        logger.info(f"\n🏆 BRUTEFORCE COMPLETE")
        logger.info(f"📊 Total simulations: {len(all_results)}")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"🏆 Success criteria met: {success_criteria_met}")
        
        if successful > 0:
            best_result = max([r for r in all_results if r.success], key=lambda x: x.sharpe)
            logger.info(f"🏆 Best result: Sharpe {best_result.sharpe:.3f} in {best_result.region}")
            
            # Show best results per template
            unique_templates = list(set([r.template for r in all_results if r.success]))
            logger.info(f"📊 Tested {len(unique_templates)} unique templates")
            
            for template in unique_templates[:5]:  # Show top 5 templates
                template_results = [r for r in all_results if r.template == template and r.success]
                if template_results:
                    best_template_result = max(template_results, key=lambda x: x.sharpe)
                    logger.info(f"🎯 Template: {template[:50]}... - Best Sharpe: {best_template_result.sharpe:.3f} in {best_template_result.region}")

def main():
    parser = argparse.ArgumentParser(description='Bruteforce Template Generator for WorldQuant Brain (4 templates per batch, 2 subprocesses each)')
    parser.add_argument('--credentials', required=True, help='Path to credentials JSON file')
    parser.add_argument('--ollama-model', default='llama3.1', help='Ollama model to use')
    parser.add_argument('--max-concurrent', type=int, default=8, help='Maximum concurrent simulations')
    parser.add_argument('--custom-template', help='Path to custom template JSON file')
    parser.add_argument('--max-batches', type=int, default=3, help='Maximum number of batches (4 templates per batch)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    parser.add_argument('--target-dataset', help='Specific dataset to test (e.g., nws77)')
    
    args = parser.parse_args()
    
    generator = BruteforceTemplateGenerator(
        credentials_path=args.credentials,
        ollama_model=args.ollama_model,
        max_concurrent=args.max_concurrent,
        target_dataset=args.target_dataset
    )
    
    generator.run_bruteforce(
        custom_template_file=args.custom_template,
        max_batches=args.max_batches,
        resume=args.resume
    )

if __name__ == "__main__":
    main()
