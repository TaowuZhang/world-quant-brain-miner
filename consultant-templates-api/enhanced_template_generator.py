#!/usr/bin/env python3
"""
Enhanced Template Generator with Multi-Simulation Testing
Integrates DeepSeek API template generation with multi-simulation testing capabilities
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_template_generator.log')
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

class EnhancedTemplateGenerator:
    def __init__(self, credentials_path: str, deepseek_api_key: str, max_concurrent: int = 5):
        """Initialize the enhanced template generator with multi-simulation capabilities"""
        self.sess = requests.Session()
        self.credentials_path = credentials_path
        self.deepseek_api_key = deepseek_api_key
        self.max_concurrent = max_concurrent
        self.setup_auth()
        
        # Region configurations
        self.region_configs = {
            "USA": RegionConfig("USA", "TOP3000", 0),
            "GLB": RegionConfig("GLB", "TOP3000", 1),
            "EUR": RegionConfig("EUR", "TOP2500", 1),
            "ASI": RegionConfig("ASI", "MINVOL1M", 1, max_trade=True),
            "CHN": RegionConfig("CHN", "TOP2000U", 1, max_trade=True)
        }
        
        # Load operators and data fields
        self.operators = self.load_operators()
        self.data_fields = {}
        
        # Results storage
        self.template_results = []
        
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
                    "content": "You are an expert in quantitative finance and WorldQuant Brain alpha expressions. Generate valid, creative alpha expression templates."
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
        """Generate templates for a specific region"""
        logger.info(f"Generating {num_templates} templates for region: {region}")
        
        # Get data fields for this region
        data_fields = self.get_data_fields_for_region(region)
        if not data_fields:
            logger.warning(f"No data fields found for region {region}")
            return []
        
        # Select a subset of operators and fields for template generation
        selected_operators = random.sample(self.operators, min(20, len(self.operators)))
        selected_fields = random.sample(data_fields, min(15, len(data_fields)))
        
        # Create prompt for DeepSeek
        operators_desc = []
        for op in selected_operators:
            operators_desc.append(f"- {op['name']}: {op['description']} (Definition: {op['definition']})")
        
        fields_desc = []
        for field in selected_fields:
            fields_desc.append(f"- {field['id']}: {field.get('description', 'No description')}")
        
        prompt = f"""Generate {num_templates} diverse and creative WorldQuant Brain alpha expression templates for the {region} region.

Available Operators:
{chr(10).join(operators_desc)}

Available Data Fields:
{chr(10).join(fields_desc)}

Requirements:
1. Each template should be a complete alpha expression
2. Use only the provided operators and data fields
3. Include proper syntax with parentheses and parameters
4. Make templates diverse - use different operator combinations
5. Include time series operations, arithmetic operations, and ranking functions
6. Each template should be on a separate line
7. Use realistic parameter values (e.g., 20, 60, 120 for time periods)
8. Include examples of complex nested operations

Format: Return only the alpha expressions, one per line, no explanations.

Example format:
ts_rank(ts_delta(close, 1), 20)
group_normalize(ts_zscore(volume, 60), industry)
winsorize(ts_regression(returns, volume, 120), std=3)

Generate {num_templates} templates:"""

        # Call DeepSeek API
        response = self.call_deepseek_api(prompt)
        if not response:
            logger.error(f"Failed to get response from DeepSeek for region {region}")
            return []
        
        # Parse the response
        templates = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Clean up the template
                template = re.sub(r'^\d+\.\s*', '', line)  # Remove numbering
                template = template.strip()
                if template:
                    templates.append({
                        'region': region,
                        'template': template,
                        'operators_used': self.extract_operators_from_template(template),
                        'fields_used': self.extract_fields_from_template(template, data_fields)
                    })
        
        logger.info(f"Generated {len(templates)} templates for region {region}")
        return templates
    
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
    
    def multi_simulate_templates(self, templates: List[Dict], region: str) -> List[TemplateResult]:
        """Multi-simulate a batch of templates using the powerhouse approach"""
        logger.info(f"Multi-simulating {len(templates)} templates for region {region}")
        
        # Create simulation settings for the region
        config = self.region_configs[region]
        settings = SimulationSettings(
            region=region,
            universe=config.universe,
            delay=config.delay,
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
    
    def generate_and_test_templates(self, regions: List[str] = None, templates_per_region: int = 10) -> Dict:
        """Generate templates and test them with multi-simulation"""
        if regions is None:
            regions = list(self.region_configs.keys())
        
        all_results = {
            'metadata': {
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_operators': len(self.operators),
                'regions': regions,
                'templates_per_region': templates_per_region
            },
            'templates': {},
            'simulation_results': {}
        }
        
        for region in regions:
            logger.info(f"Processing region: {region}")
            
            # Generate templates
            templates = self.generate_templates_for_region(region, templates_per_region)
            all_results['templates'][region] = templates
            
            if templates:
                # Multi-simulate the templates
                simulation_results = self.multi_simulate_templates(templates, region)
                all_results['simulation_results'][region] = [
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
                    for result in simulation_results
                ]
                
                # Store results for analysis
                self.template_results.extend(simulation_results)
            
            # Add delay between regions
            time.sleep(2)
        
        return all_results
    
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
    
    def save_results(self, results: Dict, filename: str = 'enhanced_generatedTemplate.json'):
        """Save results to JSON file"""
        try:
            # Add analysis to results
            results['analysis'] = self.analyze_results()
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced template generator with multi-simulation testing')
    parser.add_argument('--credentials', default='credential', help='Path to credentials file')
    parser.add_argument('--deepseek-key', required=True, help='DeepSeek API key')
    parser.add_argument('--output', default='enhanced_generatedTemplate.json', help='Output filename')
    parser.add_argument('--regions', nargs='+', help='Regions to process (default: all)')
    parser.add_argument('--templates-per-region', type=int, default=10, help='Number of templates per region')
    parser.add_argument('--max-concurrent', type=int, default=5, help='Maximum concurrent simulations')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = EnhancedTemplateGenerator(args.credentials, args.deepseek_key, args.max_concurrent)
        
        # Generate and test templates
        results = generator.generate_and_test_templates(args.regions, args.templates_per_region)
        
        # Save results
        generator.save_results(results, args.output)
        
        # Print summary
        total_templates = sum(len(templates) for templates in results['templates'].values())
        total_simulations = sum(len(sims) for sims in results['simulation_results'].values())
        successful_sims = sum(len([s for s in sims if s.get('success', False)]) for sims in results['simulation_results'].values())
        
        logger.info(f"Generation and testing complete!")
        logger.info(f"Total templates generated: {total_templates}")
        logger.info(f"Total simulations: {total_simulations}")
        logger.info(f"Successful simulations: {successful_sims}")
        logger.info(f"Success rate: {successful_sims/total_simulations*100:.1f}%" if total_simulations > 0 else "N/A")
        
        # Show best performing templates
        all_simulations = []
        for region_sims in results['simulation_results'].values():
            all_simulations.extend(region_sims)
        
        successful_simulations = [s for s in all_simulations if s.get('success', False)]
        if successful_simulations:
            best_template = max(successful_simulations, key=lambda x: x.get('sharpe', 0))
            logger.info(f"Best performing template (Sharpe: {best_template.get('sharpe', 0):.3f}):")
            logger.info(f"  {best_template.get('template', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Enhanced template generation failed: {e}")
        raise

if __name__ == '__main__':
    main()
