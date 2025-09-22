#!/usr/bin/env python3
"""
Template Generator for WorldQuant Brain Alpha Expressions
Uses Ollama API to generate comprehensive templates combining operators and data fields
"""

import argparse
import requests
import json
import os
import random
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from requests.auth import HTTPBasicAuth
import re
import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('template_generator.log')
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

class TemplateGenerator:
    def __init__(self, credentials_path: str, ollama_model: str = "llama3.1"):
        """Initialize the template generator"""
        self.sess = requests.Session()
        self.credentials_path = credentials_path
        self.ollama_model = ollama_model
        self.setup_auth()
        
        # Region configurations
        self.region_configs = {
            "USA": RegionConfig("USA", "TOP3000", 1),
            "GLB": RegionConfig("GLB", "TOP3000", 1),
            "EUR": RegionConfig("EUR", "TOP2500", 1),
            "ASI": RegionConfig("ASI", "MINVOL1M", 1, max_trade=True),
            "CHN": RegionConfig("CHN", "TOP2000U", 1, max_trade=True)
        }
        
        # Load operators and data fields
        self.operators = self.load_operators()
        self.data_fields = {}
        
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
    
    def call_ollama_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call Ollama API to generate templates in JSON format"""
        system_prompt = """You are an expert in quantitative finance and WorldQuant Brain alpha expressions. 
        Generate valid, creative alpha expression templates with proper syntax.
        ALWAYS respond with valid JSON format. Return only the JSON response, no additional text or reasoning.
        Example format: {"templates": ["template1", "template2", "template3"]}"""
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Ollama API call attempt {attempt + 1}/{max_retries}")
                
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
                    options={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 2000
                    }
                )
                
                content = response['message']['content']
                logger.info("Ollama API call successful")
                
                # Try to parse as JSON to ensure it's valid
                try:
                    json.loads(content)
                    return content
                except json.JSONDecodeError:
                    logger.warning("Ollama response is not valid JSON, attempting to extract JSON")
                    # Try to extract JSON from the response
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json_match.group(0)
                    else:
                        logger.error("No valid JSON found in Ollama response")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return None
                    
            except Exception as e:
                logger.error(f"Ollama API call failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        
        return None
    
    def generate_templates_for_region(self, region: str) -> List[Dict]:
        """Generate templates for a specific region"""
        logger.info(f"Generating templates for region: {region}")
        
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
        
        prompt = f"""Generate 10 diverse and creative WorldQuant Brain alpha expression templates for the {region} region.

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

IMPORTANT: Return your response in JSON format only. Use this exact format:
{{"templates": ["template1", "template2", "template3"]}}

Generate 10 templates in JSON format:"""

        # Call Ollama API
        response = self.call_ollama_api(prompt)
        if not response:
            logger.error(f"Failed to get response from Ollama for region {region}")
            return []
        
        # Parse the JSON response
        templates = []
        try:
            # Parse JSON response
            response_data = json.loads(response)
            
            # Extract templates from JSON
            if 'templates' in response_data:
                template_list = response_data['templates']
            elif isinstance(response_data, list):
                template_list = response_data
            else:
                logger.error("Invalid JSON format: expected 'templates' key or array")
                return []
            
            for template in template_list:
                if isinstance(template, str) and template.strip():
                    template = template.strip()
                    templates.append({
                        'region': region,
                        'template': template,
                        'operators_used': self.extract_operators_from_template(template),
                        'fields_used': self.extract_fields_from_template(template, data_fields)
                    })
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Fallback to line-by-line parsing if JSON parsing fails
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
    
    def generate_all_templates(self) -> Dict:
        """Generate templates for all regions"""
        all_templates = {
            'metadata': {
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_operators': len(self.operators),
                'regions': list(self.region_configs.keys())
            },
            'templates': {}
        }
        
        for region in self.region_configs.keys():
            logger.info(f"Processing region: {region}")
            templates = self.generate_templates_for_region(region)
            all_templates['templates'][region] = templates
            
            # Add delay between regions to avoid rate limiting
            time.sleep(2)
        
        return all_templates
    
    def save_templates(self, templates: Dict, filename: str = 'generatedTemplate.json'):
        """Save templates to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(templates, f, indent=2)
            logger.info(f"Templates saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate alpha expression templates using Ollama API')
    parser.add_argument('--credentials', default='credential', help='Path to credentials file')
    parser.add_argument('--ollama-model', default='llama3.1', help='Ollama model to use (default: llama3.1)')
    parser.add_argument('--output', default='generatedTemplate.json', help='Output filename')
    parser.add_argument('--region', help='Generate templates for specific region only')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = TemplateGenerator(args.credentials, args.ollama_model)
        
        if args.region:
            # Generate for specific region
            if args.region not in generator.region_configs:
                logger.error(f"Invalid region: {args.region}")
                return
            
            templates = generator.generate_templates_for_region(args.region)
            result = {
                'metadata': {
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'region': args.region,
                    'total_operators': len(generator.operators)
                },
                'templates': {args.region: templates}
            }
        else:
            # Generate for all regions
            result = generator.generate_all_templates()
        
        # Save results
        generator.save_templates(result, args.output)
        
        # Print summary
        total_templates = sum(len(templates) for templates in result['templates'].values())
        logger.info(f"Generation complete! Total templates: {total_templates}")
        
        for region, templates in result['templates'].items():
            logger.info(f"Region {region}: {len(templates)} templates")
        
    except Exception as e:
        logger.error(f"Template generation failed: {e}")
        raise

if __name__ == '__main__':
    main()
