#!/usr/bin/env python3
"""
Enhanced Multi-Threaded Atom Tester with Operator Combinations
- 8-thread management system similar to consultant-templates-ollama
- Operator stacking combinations (0, 1, 2, 3 operators)
- Ollama for intelligent operator combination generation
- Progress saving and resume functionality
- Quality checks and submission validation
- Region-based testing (ASI, CHN, EUR, GLB, USA)
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
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import numpy as np
from datetime import datetime
import threading
import sys
import math
import subprocess
from collections import defaultdict, Counter
import statistics
import ollama
from itertools import combinations, product

# Configure logging with Unicode handling
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
        logging.FileHandler('enhanced_multi_threaded_atom_tester.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class OperatorCombination:
    """Represents a combination of operators to apply to data fields"""
    combination_id: str
    operators: List[str]  # List of operator names in order
    stack_level: int  # 0, 1, 2, or 3
    description: str
    category: str  # e.g., "momentum", "mean_reversion", "volatility"
    complexity: str  # "simple", "medium", "complex"
    generated_by: str = "ollama"

@dataclass
class AtomTestResult:
    """Result of an atom test with operator combinations"""
    atom_id: str
    expression: str
    data_field_id: str
    data_field_name: str
    dataset_id: str
    dataset_name: str
    region: str
    universe: str
    delay: int
    neutralization: str
    operator_combination: OperatorCombination
    status: str  # 'success', 'failed', 'error', 'too_good'
    sharpe_ratio: Optional[float] = None
    fitness: Optional[float] = None
    returns: Optional[float] = None
    max_drawdown: Optional[float] = None
    turnover: Optional[float] = None
    hit_ratio: Optional[float] = None
    pnl_data: Optional[Dict] = None
    submission_checks: Optional[Dict] = None
    color_status: Optional[str] = None  # RED, GREEN, YELLOW
    prod_correlation: Optional[float] = None
    error_message: Optional[str] = None
    test_timestamp: str = None
    execution_time: Optional[float] = None

@dataclass
class ProgressState:
    """Progress state for resuming operations"""
    current_region: str
    current_data_field_index: int
    completed_combinations: Set[str]
    completed_tests: int
    total_tests: int
    start_time: float
    last_save_time: float

class OllamaOperatorGenerator:
    """Generates operator combinations using Ollama for intelligent sequencing"""
    
    def __init__(self, model: str = "llama3.1", operators_file: str = "operatorRAW.json"):
        """Initialize the Ollama operator generator"""
        self.model = model
        self.operators_file = operators_file
        self.operators = []
        self.simple_operators = []
        self.complex_operators = []
        
        # Load operators
        self._load_operators()
        
        # Initialize Ollama
        self._setup_ollama()
    
    def _load_operators(self):
        """Load operators from operatorRAW.json"""
        try:
            with open(self.operators_file, 'r', encoding='utf-8') as f:
                self.operators = json.load(f)
            
            # Categorize operators by complexity
            for op in self.operators:
                if self._is_simple_operator(op):
                    self.simple_operators.append(op)
                else:
                    self.complex_operators.append(op)
            
            logger.info(f"✅ Loaded {len(self.operators)} operators ({len(self.simple_operators)} simple, {len(self.complex_operators)} complex)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load operators: {e}")
            raise
    
    def _is_simple_operator(self, operator: Dict) -> bool:
        """Determine if an operator is simple (suitable for stacking)"""
        name = operator.get('name', '').lower()
        description = operator.get('description', '').lower()
        
        # Simple operators that work well in combinations
        simple_indicators = [
            'rank', 'ts_rank', 'ts_min', 'ts_max', 'ts_mean', 'ts_std', 'ts_sum',
            'delta', 'delay', 'corr', 'cov', 'scale', 'clip', 'abs', 'sign',
            'log', 'sqrt', 'pow', 'exp', 'floor', 'ceil', 'round', 'add', 'subtract',
            'multiply', 'divide', 'max', 'min', 'mean', 'std', 'sum', 'count'
        ]
        
        # Complex operators that should be used sparingly
        complex_indicators = [
            'regression', 'neural', 'ml', 'machine', 'learning', 'lstm', 'gru',
            'transformer', 'attention', 'convolution', 'filter', 'smooth'
        ]
        
        # Check for simple indicators
        for indicator in simple_indicators:
            if indicator in name or indicator in description:
                return True
        
        # Check for complex indicators
        for indicator in complex_indicators:
            if indicator in name or indicator in description:
                return False
        
        # Default to simple for unknown operators
        return True
    
    def _setup_ollama(self):
        """Setup Ollama connection"""
        try:
            # Test Ollama connection
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "Hello, are you ready to help with operator combinations?"}]
            )
            logger.info("✅ Ollama connection established")
        except Exception as e:
            logger.error(f"❌ Failed to setup Ollama: {e}")
            raise
    
    def generate_operator_combinations(self, num_combinations: int = 10) -> List[OperatorCombination]:
        """Generate operator combinations using Ollama intelligence"""
        combinations = []
        
        # Generate combinations for each stack level (reduced for faster generation)
        for stack_level in [0, 1, 2, 3]:
            level_combinations = self._generate_level_combinations(stack_level, max(1, num_combinations // 4))
            combinations.extend(level_combinations)
        
        # Shuffle to randomize order
        random.shuffle(combinations)
        
        logger.info(f"✅ Generated {len(combinations)} operator combinations")
        return combinations
    
    def _generate_level_combinations(self, stack_level: int, num_combinations: int) -> List[OperatorCombination]:
        """Generate combinations for a specific stack level"""
        combinations = []
        
        if stack_level == 0:
            # No operators - just return data field as-is
            combinations.append(OperatorCombination(
                combination_id=f"stack_0_0",
                operators=[],
                stack_level=0,
                description="No operators applied - raw data field",
                category="raw",
                complexity="simple"
            ))
            return combinations
        
        # Use Ollama to generate intelligent combinations (limited to avoid infinite loops)
        max_ollama_attempts = min(num_combinations, 2)  # Limit Ollama calls to just 2 per level
        for i in range(max_ollama_attempts):
            try:
                combination = self._generate_single_combination(stack_level)
                if combination:
                    combination.combination_id = f"stack_{stack_level}_{i}"
                    combinations.append(combination)
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate combination {i} for level {stack_level}: {e}")
                # Fallback to random selection
                combination = self._generate_fallback_combination(stack_level)
                combination.combination_id = f"stack_{stack_level}_{i}_fallback"
                combinations.append(combination)
        
        # Fill remaining with fallback combinations
        while len(combinations) < num_combinations:
            combination = self._generate_fallback_combination(stack_level)
            combination.combination_id = f"stack_{stack_level}_{len(combinations)}_fallback"
            combinations.append(combination)
        
        return combinations
    
    def _generate_single_combination(self, stack_level: int) -> Optional[OperatorCombination]:
        """Generate a single combination using Ollama"""
        
        # Create context for Ollama
        available_operators = self.simple_operators[:20]  # Limit to top 20 simple operators
        operator_names = [op['name'] for op in available_operators]
        
        prompt = f"""
You are an expert quantitative researcher creating operator combinations for financial data analysis.

Available operators: {', '.join(operator_names[:10])}

Create a meaningful combination of {stack_level} operators that work well together for financial time series analysis.

Requirements:
- Use exactly {stack_level} operators
- Choose operators that complement each other
- Focus on operators that work well with financial data
- Consider momentum, mean reversion, volatility, and correlation patterns

Return your response in this exact format:
OPERATORS: [operator1, operator2, operator3]
CATEGORY: [momentum/mean_reversion/volatility/correlation/trend]
DESCRIPTION: [Brief description of what this combination does]
COMPLEXITY: [simple/medium/complex]

Example for 2 operators:
OPERATORS: [ts_rank, delta]
CATEGORY: momentum
DESCRIPTION: Ranks time series and calculates momentum
COMPLEXITY: simple
"""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={'timeout': 10}  # 10 second timeout
            )
            
            content = response['message']['content']
            return self._parse_ollama_response(content, stack_level)
            
        except Exception as e:
            logger.warning(f"⚠️ Ollama generation failed: {e}")
            return None
    
    def _parse_ollama_response(self, content: str, stack_level: int) -> Optional[OperatorCombination]:
        """Parse Ollama response into OperatorCombination"""
        try:
            lines = content.strip().split('\n')
            operators = []
            category = "general"
            description = "Generated combination"
            complexity = "simple"
            
            for line in lines:
                line = line.strip()
                if line.startswith('OPERATORS:'):
                    # Extract operators from [operator1, operator2, operator3] format
                    op_text = line.replace('OPERATORS:', '').strip()
                    if op_text.startswith('[') and op_text.endswith(']'):
                        op_text = op_text[1:-1]  # Remove brackets
                        operators = [op.strip() for op in op_text.split(',')]
                elif line.startswith('CATEGORY:'):
                    category = line.replace('CATEGORY:', '').strip()
                elif line.startswith('DESCRIPTION:'):
                    description = line.replace('DESCRIPTION:', '').strip()
                elif line.startswith('COMPLEXITY:'):
                    complexity = line.replace('COMPLEXITY:', '').strip()
            
            # Validate operators exist
            valid_operators = []
            for op in operators:
                if any(op == existing_op['name'] for existing_op in self.simple_operators):
                    valid_operators.append(op)
                else:
                    logger.warning(f"⚠️ Unknown operator: {op}")
            
            # If we don't have the right number of operators, try to fix it
            if len(valid_operators) != stack_level:
                if len(valid_operators) > stack_level:
                    # Take only the first N operators
                    valid_operators = valid_operators[:stack_level]
                elif len(valid_operators) < stack_level:
                    # Fill with random operators if needed
                    available_ops = [op['name'] for op in self.simple_operators if op['name'] not in valid_operators]
                    needed = stack_level - len(valid_operators)
                    if len(available_ops) >= needed:
                        valid_operators.extend(random.sample(available_ops, needed))
                    else:
                        logger.warning(f"⚠️ Not enough operators available, using {len(valid_operators)} instead of {stack_level}")
                        return None
            
            return OperatorCombination(
                combination_id="",  # Will be set by caller
                operators=valid_operators,
                stack_level=stack_level,
                description=description,
                category=category,
                complexity=complexity
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse Ollama response: {e}")
            return None
    
    def _generate_fallback_combination(self, stack_level: int) -> OperatorCombination:
        """Generate a fallback combination using random selection"""
        if stack_level == 0:
            return OperatorCombination(
                combination_id="",
                operators=[],
                stack_level=0,
                description="No operators applied - raw data field",
                category="raw",
                complexity="simple"
            )
        
        # Randomly select operators
        selected_operators = random.sample(
            [op['name'] for op in self.simple_operators], 
            min(stack_level, len(self.simple_operators))
        )
        
        return OperatorCombination(
            combination_id="",
            operators=selected_operators,
            stack_level=stack_level,
            description=f"Random combination of {stack_level} operators",
            category="random",
            complexity="simple"
        )

class EnhancedMultiThreadedAtomTester:
    """Enhanced multi-threaded atom tester with operator combinations"""
    
    def __init__(self, credential_file: str = "credential.txt"):
        """Initialize the enhanced atom tester"""
        self.credential_file = credential_file
        self.results: List[AtomTestResult] = []
        self.operator_combinations: List[OperatorCombination] = []
        self.data_fields = {}
        self.regions = ['ASI', 'CHN', 'EUR', 'GLB', 'USA']  # Start with ASI as requested
        self.universes = ['TOP3000', 'TOP2000', 'TOP1000']
        self.neutralizations = ['INDUSTRY', 'SUBINDUSTRY', 'SECTOR', 'COUNTRY', 'NONE']
        
        # Progress tracking
        self.progress_file = "atom_test_progress.json"
        self.results_file = "enhanced_atom_results.json"
        self.progress_state: Optional[ProgressState] = None
        
        # Load credentials and setup session
        self._load_credentials()
        
        # Setup session
        self.sess = requests.Session()
        
        # Setup authentication using session-based auth
        self._setup_auth()
        
        # Load data fields and operators
        self._load_data_fields()
        self._load_operators()
        
        # Initialize Ollama generator (skip for now to avoid issues)
        try:
            self.ollama_generator = OllamaOperatorGenerator()
            logger.info("✅ Ollama generator initialized")
        except Exception as e:
            logger.warning(f"⚠️ Ollama not available, using fallback combinations: {e}")
            self.ollama_generator = None
        
        # Generate operator combinations
        self._generate_operator_combinations()
        
        logger.info("✅ Enhanced Multi-Threaded Atom Tester initialized")
    
    def _load_credentials(self):
        """Load credentials from file"""
        try:
            with open(self.credential_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Try JSON format first
            try:
                creds = json.loads(content)
                if isinstance(creds, list) and len(creds) == 2:
                    self.username, self.password = creds
                else:
                    raise ValueError("Invalid JSON format")
            except (json.JSONDecodeError, ValueError):
                # Try two-line format
                lines = content.split('\n')
                if len(lines) >= 2:
                    self.username = lines[0].strip()
                    self.password = lines[1].strip()
                else:
                    raise ValueError("Invalid credential format")
            
            logger.info(f"✅ Credentials loaded for user: {self.username}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load credentials: {e}")
            raise
    
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
                raise Exception(f"Authentication failed with status {auth_response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Failed to authenticate: {e}")
            raise
    
    def _load_data_fields(self):
        """Load data fields from cache files"""
        cache_files = [f for f in os.listdir('.') if f.startswith('data_fields_cache_') and f.endswith('.json')]
        
        if not cache_files:
            raise FileNotFoundError("No data_fields_cache_*.json files found")
        
        total_fields = 0
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    fields = json.load(f)
                
                # Extract region from filename (handle format like ASI_1, USA_0, etc.)
                region_part = cache_file.replace('data_fields_cache_', '').replace('.json', '')
                # Extract just the region part (before the underscore)
                region = region_part.split('_')[0]
                
                self.data_fields[region] = fields
                total_fields += len(fields)
                
                logger.info(f"✅ Loaded {len(fields)} cached fields from {cache_file}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {cache_file}: {e}")
        
        logger.info(f"✅ Total data fields loaded: {total_fields}")
    
    def _load_operators(self):
        """Load operators from operatorRAW.json"""
        try:
            with open("operatorRAW.json", 'r', encoding='utf-8') as f:
                self.operators = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.operators)} operators")
            
        except Exception as e:
            logger.error(f"❌ Failed to load operators: {e}")
            raise
    
    def _generate_operator_combinations(self):
        """Generate operator combinations using Ollama"""
        if self.ollama_generator is None:
            logger.info("🔄 Using fallback operator combinations (Ollama not available)")
            self._generate_fallback_combinations()
            return
            
        try:
            self.operator_combinations = self.ollama_generator.generate_operator_combinations(10)  # Just 10 combinations
            logger.info(f"✅ Generated {len(self.operator_combinations)} operator combinations")
        except Exception as e:
            logger.error(f"❌ Failed to generate operator combinations: {e}")
            # Fallback to simple combinations
            self._generate_fallback_combinations()
    
    def _generate_fallback_combinations(self):
        """Generate fallback combinations without Ollama"""
        combinations = []
        
        # 0 operator combinations
        combinations.append(OperatorCombination(
            combination_id="stack_0_0",
            operators=[],
            stack_level=0,
            description="No operators applied - raw data field",
            category="raw",
            complexity="simple"
        ))
        
        # 1 operator combinations (top 10 simple operators)
        simple_ops = [op['name'] for op in self.operators if self._is_simple_operator(op)][:10]
        for i, op in enumerate(simple_ops):
            combinations.append(OperatorCombination(
                combination_id=f"stack_1_{i}",
                operators=[op],
                stack_level=1,
                description=f"Single operator: {op}",
                category="single",
                complexity="simple"
            ))
        
        # 2 operator combinations (top 5 pairs)
        from itertools import combinations as itertools_combinations
        for i, (op1, op2) in enumerate(itertools_combinations(simple_ops[:5], 2)):
            combinations.append(OperatorCombination(
                combination_id=f"stack_2_{i}",
                operators=[op1, op2],
                stack_level=2,
                description=f"Two operators: {op1} -> {op2}",
                category="double",
                complexity="medium"
            ))
        
        # 3 operator combinations (top 3 triplets)
        for i, (op1, op2, op3) in enumerate(itertools_combinations(simple_ops[:4], 3)):
            combinations.append(OperatorCombination(
                combination_id=f"stack_3_{i}",
                operators=[op1, op2, op3],
                stack_level=3,
                description=f"Three operators: {op1} -> {op2} -> {op3}",
                category="triple",
                complexity="complex"
            ))
        
        self.operator_combinations = combinations
        logger.info(f"✅ Generated {len(combinations)} fallback operator combinations")
    
    def _is_simple_operator(self, operator: Dict) -> bool:
        """Determine if an operator is simple (suitable for stacking)"""
        name = operator.get('name', '').lower()
        description = operator.get('description', '').lower()
        
        # Simple operators that work well in combinations
        simple_indicators = [
            'rank', 'ts_rank', 'ts_min', 'ts_max', 'ts_mean', 'ts_std', 'ts_sum',
            'delta', 'delay', 'corr', 'cov', 'scale', 'clip', 'abs', 'sign',
            'log', 'sqrt', 'pow', 'exp', 'floor', 'ceil', 'round', 'add', 'subtract',
            'multiply', 'divide', 'max', 'min', 'mean', 'std', 'sum', 'count'
        ]
        
        for indicator in simple_indicators:
            if indicator in name or indicator in description:
                return True
        
        return False
    
    def _build_expression(self, data_field_id: str, operator_combination: OperatorCombination) -> str:
        """Build expression by applying operator combination to data field"""
        if operator_combination.stack_level == 0:
            return data_field_id
        
        expression = data_field_id
        
        # Apply operators in sequence
        for operator in operator_combination.operators:
            # Find operator definition
            op_def = next((op for op in self.operators if op['name'] == operator), None)
            if not op_def:
                logger.warning(f"⚠️ Unknown operator: {operator}")
                continue
            
            # Build operator call
            if operator in ['add', 'subtract', 'multiply', 'divide']:
                # Binary operators need two operands
                expression = f"{operator}({expression}, {expression})"
            elif operator in ['ts_rank', 'ts_min', 'ts_max', 'ts_mean', 'ts_std', 'ts_sum']:
                # Time series operators
                expression = f"{operator}({expression}, 20)"  # 20-day window
            elif operator in ['delta', 'delay']:
                # Delta and delay operators
                expression = f"{operator}({expression}, 1)"
            else:
                # Unary operators
                expression = f"{operator}({expression})"
        
        return expression
    
    def _check_too_good_to_be_true(self, result: AtomTestResult) -> bool:
        """Check if results are too good to be true"""
        if not result.sharpe_ratio or not result.returns:
            return False
        
        # Check for unrealistic performance
        if result.sharpe_ratio > 5.0:  # Very high Sharpe
            return True
        
        if result.returns > 0.5:  # Very high returns
            return True
        
        if result.sharpe_ratio > 3.0 and result.returns > 0.3:  # High Sharpe + high returns
            return True
        
        return False
    
    def _check_submission_quality(self, alpha_id: str) -> Dict:
        """Check submission quality and return color status"""
        try:
            # Check alpha submission
            check_url = f"https://api.worldquantbrain.com/alphas/{alpha_id}/check"
            response = self.sess.get(check_url, timeout=30)
            
            if response.status_code != 200:
                return {"status": "error", "color": "RED", "message": "Failed to check submission"}
            
            checks = response.json().get('is', {}).get('checks', [])
            
            # Analyze checks
            has_fail = False
            has_warning = False
            prod_correlation = None
            
            for check in checks:
                if check.get('result') == 'FAIL':
                    has_fail = True
                elif check.get('result') == 'WARNING':
                    has_warning = True
                
                if check.get('name') == 'PROD_CORRELATION':
                    prod_correlation = check.get('value')
            
            # Determine color status
            if has_fail:
                color = "RED"
            elif has_warning:
                color = "YELLOW"
            else:
                color = "GREEN"
            
            return {
                "status": "success",
                "color": color,
                "prod_correlation": prod_correlation,
                "checks": checks
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to check submission quality: {e}")
            return {"status": "error", "color": "RED", "message": str(e)}
    
    def _test_single_atom(self, data_field: Dict, operator_combination: OperatorCombination, 
                         region: str, universe: str, neutralization: str) -> AtomTestResult:
        """Test a single atom with operator combination"""
        start_time = time.time()
        
        try:
            # Build expression
            expression = self._build_expression(data_field['id'], operator_combination)
            
            # Create simulation (using the correct format from original atom tester)
            simulation_data = {
                "type": "REGULAR",
                "settings": {
                    "instrumentType": "EQUITY",
                    "region": region,
                    "universe": universe,
                    "delay": 1,
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
            submit_url = "https://api.worldquantbrain.com/simulations"
            response = self.sess.post(submit_url, json=simulation_data, timeout=60)
            
            if response.status_code != 201:
                return AtomTestResult(
                    atom_id="",
                    expression=expression,
                    data_field_id=data_field['id'],
                    data_field_name=data_field.get('description', ''),
                    dataset_id=data_field.get('dataset', {}).get('id', ''),
                    dataset_name=data_field.get('dataset', {}).get('name', ''),
                    region=region,
                    universe=universe,
                    delay=1,
                    neutralization=neutralization,
                    operator_combination=operator_combination,
                    status="failed",
                    error_message=f"Submission failed: {response.status_code}",
                    test_timestamp=datetime.now().isoformat(),
                    execution_time=time.time() - start_time
                )
            
            simulation_id = response.json().get('id', '')
            
            # Monitor simulation
            simulation_result = self._monitor_simulation(simulation_id)
            
            if not simulation_result['success']:
                return AtomTestResult(
                    atom_id=simulation_id,
                    expression=expression,
                    data_field_id=data_field['id'],
                    data_field_name=data_field.get('description', ''),
                    dataset_id=data_field.get('dataset', {}).get('id', ''),
                    dataset_name=data_field.get('dataset', {}).get('name', ''),
                    region=region,
                    universe=universe,
                    delay=1,
                    neutralization=neutralization,
                    operator_combination=operator_combination,
                    status="failed",
                    error_message=simulation_result.get('error', 'Simulation failed'),
                    test_timestamp=datetime.now().isoformat(),
                    execution_time=time.time() - start_time
                )
            
            # Extract results
            sharpe_ratio = simulation_result.get('sharpe', 0.0)
            returns = simulation_result.get('returns', 0.0)
            fitness = simulation_result.get('fitness', 0.0)
            turnover = simulation_result.get('turnover', 0.0)
            max_drawdown = simulation_result.get('maxDrawdown', 0.0)
            hit_ratio = simulation_result.get('hitRatio', 0.0)
            pnl_data = simulation_result.get('pnl', {})
            
            # Check submission quality
            submission_checks = self._check_submission_quality(simulation_id)
            
            # Create result
            result = AtomTestResult(
                atom_id=simulation_id,
                expression=expression,
                data_field_id=data_field['id'],
                data_field_name=data_field.get('description', ''),
                dataset_id=data_field.get('dataset', {}).get('id', ''),
                dataset_name=data_field.get('dataset', {}).get('name', ''),
                region=region,
                universe=universe,
                delay=1,
                neutralization=neutralization,
                operator_combination=operator_combination,
                status="success",
                sharpe_ratio=sharpe_ratio,
                fitness=fitness,
                returns=returns,
                max_drawdown=max_drawdown,
                turnover=turnover,
                hit_ratio=hit_ratio,
                pnl_data=pnl_data,
                submission_checks=submission_checks,
                color_status=submission_checks.get('color', 'RED'),
                prod_correlation=submission_checks.get('prod_correlation'),
                test_timestamp=datetime.now().isoformat(),
                execution_time=time.time() - start_time
            )
            
            # Check if too good to be true
            if self._check_too_good_to_be_true(result):
                result.status = "too_good"
                result.color_status = "RED"
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error testing atom: {e}")
            return AtomTestResult(
                atom_id="",
                expression="",
                data_field_id=data_field['id'],
                data_field_name=data_field.get('description', ''),
                dataset_id=data_field.get('dataset', {}).get('id', ''),
                dataset_name=data_field.get('dataset', {}).get('name', ''),
                region=region,
                universe=universe,
                delay=1,
                neutralization=neutralization,
                operator_combination=operator_combination,
                status="error",
                error_message=str(e),
                test_timestamp=datetime.now().isoformat(),
                execution_time=time.time() - start_time
            )
    
    def _monitor_simulation(self, alpha_id: str, max_wait_time: int = 300) -> Dict:
        """Monitor simulation progress"""
        start_time = time.time()
        check_interval = 5  # Check every 5 seconds
        
        while time.time() - start_time < max_wait_time:
            try:
                # Check simulation status
                status_url = f"https://api.worldquantbrain.com/alphas/{alpha_id}/status"
                response = self.sess.get(status_url, timeout=10)
                
                if response.status_code != 200:
                    time.sleep(check_interval)
                    continue
                
                status_data = response.json()
                status = status_data.get('status', 0)
                
                if status == 1:  # Completed
                    # Get results
                    results_url = f"https://api.worldquantbrain.com/alphas/{alpha_id}/results"
                    results_response = self.sess.get(results_url, timeout=30)
                    
                    if results_response.status_code == 200:
                        results = results_response.json()
                        return {
                            'success': True,
                            'sharpe': results.get('sharpe', 0.0),
                            'returns': results.get('returns', 0.0),
                            'fitness': results.get('fitness', 0.0),
                            'turnover': results.get('turnover', 0.0),
                            'maxDrawdown': results.get('maxDrawdown', 0.0),
                            'hitRatio': results.get('hitRatio', 0.0),
                            'pnl': results.get('pnl', {})
                        }
                    else:
                        return {'success': False, 'error': 'Failed to get results'}
                
                elif status == -1:  # Failed
                    return {'success': False, 'error': 'Simulation failed'}
                
                # Still running, wait and check again
                time.sleep(check_interval)
                
            except Exception as e:
                logger.warning(f"⚠️ Error checking simulation progress: {e}")
                time.sleep(check_interval)
        
        return {'success': False, 'error': 'Simulation timeout'}
    
    def _save_progress(self):
        """Save current progress to file"""
        try:
            progress_data = {
                'current_region': self.progress_state.current_region if self.progress_state else 'ASI',
                'current_data_field_index': self.progress_state.current_data_field_index if self.progress_state else 0,
                'completed_combinations': list(self.progress_state.completed_combinations) if self.progress_state else [],
                'completed_tests': self.progress_state.completed_tests if self.progress_state else 0,
                'total_tests': self.progress_state.total_tests if self.progress_state else 0,
                'start_time': self.progress_state.start_time if self.progress_state else time.time(),
                'last_save_time': time.time()
            }
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)
            
            logger.info("💾 Progress saved")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save progress: {e}")
    
    def _load_progress(self) -> bool:
        """Load progress from file"""
        try:
            if not os.path.exists(self.progress_file):
                return False
            
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            self.progress_state = ProgressState(
                current_region=progress_data.get('current_region', 'ASI'),
                current_data_field_index=progress_data.get('current_data_field_index', 0),
                completed_combinations=set(progress_data.get('completed_combinations', [])),
                completed_tests=progress_data.get('completed_tests', 0),
                total_tests=progress_data.get('total_tests', 0),
                start_time=progress_data.get('start_time', time.time()),
                last_save_time=progress_data.get('last_save_time', time.time())
            )
            
            logger.info(f"📁 Loaded progress: {self.progress_state.completed_tests}/{self.progress_state.total_tests} tests completed")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load progress: {e}")
            return False
    
    def _save_results(self):
        """Save results to JSON file"""
        try:
            # Convert results to serializable format
            serializable_results = []
            for result in self.results:
                result_dict = asdict(result)
                # Convert operator combination to dict
                result_dict['operator_combination'] = asdict(result.operator_combination)
                serializable_results.append(result_dict)
            
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"💾 Results saved to {self.results_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save results: {e}")
    
    def run_multi_threaded_atom_tests(self, max_workers: int = 8, resume: bool = False):
        """Run multi-threaded atom tests with operator combinations"""
        logger.info("🚀 Starting Enhanced Multi-Threaded Atom Testing System...")
        logger.info(f"🔧 Using {max_workers} workers for parallel processing")
        
        # Load progress if resuming
        if resume and self._load_progress():
            logger.info("📁 Resuming from previous progress")
        else:
            # Initialize progress state
            self.progress_state = ProgressState(
                current_region='ASI',
                current_data_field_index=0,
                completed_combinations=set(),
                completed_tests=0,
                total_tests=0,
                start_time=time.time(),
                last_save_time=time.time()
            )
        
        # Calculate total tests
        total_tests = 0
        for region in self.regions:
            if region in self.data_fields:
                total_tests += len(self.data_fields[region]) * len(self.operator_combinations)
        
        self.progress_state.total_tests = total_tests
        logger.info(f"🎯 Total tests planned: {total_tests}")
        
        # Create test tasks
        test_tasks = []
        for region in self.regions:
            if region not in self.data_fields:
                continue
                
            for data_field in self.data_fields[region]:
                for operator_combination in self.operator_combinations:
                    # Skip if already completed
                    task_id = f"{region}_{data_field['id']}_{operator_combination.combination_id}"
                    if task_id in self.progress_state.completed_combinations:
                        continue
                    
                    test_tasks.append({
                        'region': region,
                        'data_field': data_field,
                        'operator_combination': operator_combination,
                        'task_id': task_id
                    })
        
        logger.info(f"📋 Created {len(test_tasks)} test tasks")
        
        # Run tests with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in test_tasks:
                future = executor.submit(
                    self._test_single_atom,
                    task['data_field'],
                    task['operator_combination'],
                    task['region'],
                    'TOP3000',  # Default universe
                    'INDUSTRY'  # Default neutralization
                )
                future_to_task[future] = task
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    
                    # Update progress
                    self.progress_state.completed_combinations.add(task['task_id'])
                    self.progress_state.completed_tests += 1
                    
                    # Log result
                    status_emoji = "✅" if result.status == "success" else "❌"
                    sharpe_str = f"{result.sharpe_ratio:.3f}" if result.sharpe_ratio is not None else "N/A"
                    color_str = result.color_status if result.color_status is not None else "N/A"
                    logger.info(f"{status_emoji} {result.region} | {result.data_field_name[:30]}... | {result.operator_combination.combination_id} | Sharpe: {sharpe_str} | Color: {color_str}")
                    
                    # Save progress every 10 tests
                    if self.progress_state.completed_tests % 10 == 0:
                        self._save_progress()
                        self._save_results()
                    
                except Exception as e:
                    logger.error(f"❌ Task failed: {e}")
        
        # Final save
        self._save_progress()
        self._save_results()
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print testing summary"""
        logger.info("🎉 Multi-threaded atom testing completed!")
        logger.info(f"📊 Total tests: {len(self.results)}")
        
        # Count by status
        status_counts = Counter(result.status for result in self.results)
        logger.info(f"📈 Status breakdown: {dict(status_counts)}")
        
        # Count by color
        color_counts = Counter(result.color_status for result in self.results)
        logger.info(f"🎨 Color breakdown: {dict(color_counts)}")
        
        # Best results
        successful_results = [r for r in self.results if r.status == "success" and r.sharpe_ratio is not None]
        if successful_results:
            best_sharpe = max(successful_results, key=lambda x: x.sharpe_ratio)
            logger.info(f"🏆 Best Sharpe: {best_sharpe.sharpe_ratio:.3f} ({best_sharpe.data_field_name})")
        
        logger.info(f"💾 Results saved to: {self.results_file}")
        logger.info(f"📁 Progress saved to: {self.progress_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced Multi-Threaded Atom Tester')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    
    args = parser.parse_args()
    
    try:
        tester = EnhancedMultiThreadedAtomTester()
        tester.run_multi_threaded_atom_tests(max_workers=args.workers, resume=args.resume)
    except KeyboardInterrupt:
        logger.info("⏹️ Testing interrupted by user")
    except Exception as e:
        logger.error(f"💥 Error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()