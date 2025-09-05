import argparse
import requests
import json
import os
from time import sleep
from requests.auth import HTTPBasicAuth
from typing import List, Dict
import time
import re
import logging
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# Configure logger
logger = logging.getLogger(__name__)

class RetryQueue:
    def __init__(self, generator, max_retries=3, retry_delay=60):
        self.queue = Queue()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.generator = generator  # Store reference to generator
        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()
    
    def add(self, alpha: str, retry_count: int = 0):
        self.queue.put((alpha, retry_count))
    
    def _process_queue(self):
        while True:
            if not self.queue.empty():
                alpha, retry_count = self.queue.get()
                if retry_count >= self.max_retries:
                    logging.error(f"Max retries exceeded for alpha: {alpha}")
                    continue
                    
                try:
                    result = self.generator._test_alpha_impl(alpha)  # Use _test_alpha_impl to avoid recursion
                    if result.get("status") == "error" and "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
                        logging.info(f"Simulation limit exceeded, requeueing alpha: {alpha}")
                        time.sleep(self.retry_delay)
                        self.add(alpha, retry_count + 1)
                    else:
                        self.generator.results.append({
                            "alpha": alpha,
                            "result": result
                        })
                except Exception as e:
                    logging.error(f"Error processing alpha: {str(e)}")
                    
            time.sleep(1)  # Prevent busy waiting

class AlphaGenerator:
    def __init__(self, credentials_path: str, ollama_url: str = "http://localhost:11434", max_concurrent: int = 2):
        self.sess = requests.Session()
        self.credentials_path = credentials_path  # Store path for reauth
        self.setup_auth(credentials_path)
        self.ollama_url = ollama_url
        self.results = []
        self.pending_results = {}
        self.retry_queue = RetryQueue(self)
        # Reduce concurrent workers to prevent VRAM issues
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)  # For concurrent simulations
        self.vram_cleanup_interval = 10  # Cleanup every 10 operations
        self.operation_count = 0
        
        # Model downgrade tracking
        self.initial_model = getattr(self, 'model_name', 'deepseek-r1:8b')
        self.error_count = 0
        self.max_errors_before_downgrade = 3
        self.model_fleet = [
            'deepseek-r1:8b',   # Primary model
            'deepseek-r1:7b',   # First fallback
            'deepseek-r1:1.5b', # Second fallback
            'llama3:3b',        # Third fallback
            'phi3:mini'         # Emergency fallback
        ]
        self.current_model_index = 0
        
    def setup_auth(self, credentials_path: str) -> None:
        """Set up authentication with WorldQuant Brain."""
        logging.info(f"Loading credentials from {credentials_path}")
        with open(credentials_path) as f:
            credentials = json.load(f)
        
        username, password = credentials
        self.sess.auth = HTTPBasicAuth(username, password)
        
        logging.info("Authenticating with WorldQuant Brain...")
        response = self.sess.post('https://api.worldquantbrain.com/authentication')
        logging.info(f"Authentication response status: {response.status_code}")
        logging.debug(f"Authentication response: {response.text[:500]}...")
        
        if response.status_code != 201:
            raise Exception(f"Authentication failed: {response.text}")
    
    def cleanup_vram(self):
        """Perform VRAM cleanup by forcing garbage collection and waiting."""
        try:
            import gc
            gc.collect()
            logging.info("Performed VRAM cleanup")
            # Add a small delay to allow GPU memory to be freed
            time.sleep(2)
        except Exception as e:
            logging.warning(f"VRAM cleanup failed: {e}")
        
    def get_data_fields(self, region: str = "USA", universe: str = "TOP3000") -> List[Dict]:
        """Fetch available data fields from WorldQuant Brain across multiple datasets with random sampling."""
        all_fields = []
        
        # Region-specific delay availability
        # ASI and CHN regions only support delay=1 for EQUITY
        if region in ["ASI", "CHN"]:
            delays = [1]  # Only delay 1 for ASI/CHN
        else:
            delays = [0, 1]  # Both delays for other regions
        
        for delay in delays:
            base_params = {
                'delay': delay,
                'instrumentType': 'EQUITY',
                'limit': 20,
                'region': region,
                'universe': universe
            }
            
            try:
                # First, get available datasets for this region
                datasets_params = {
                    'category': 'fundamental',
                    'delay': delay,
                    'instrumentType': 'EQUITY',
                    'region': region,
                    'universe': universe,
                    'limit': 50
                }
                
                print(f"Getting available datasets for region {region} with delay {delay}")
                datasets_response = self.sess.get('https://api.worldquantbrain.com/data-sets', params=datasets_params)
                
                if datasets_response.status_code == 200:
                    datasets_data = datasets_response.json()
                    available_datasets = datasets_data.get('results', [])
                    print(f"Found {len(available_datasets)} available datasets for region {region} with delay {delay}")
                    
                    # Extract dataset IDs
                    dataset_ids = [ds.get('id') for ds in available_datasets if ds.get('id')]
                    print(f"Available dataset IDs: {dataset_ids}")
                    
                    # If no datasets found, fall back to default datasets
                    if not dataset_ids:
                        print(f"No datasets found for region {region} with delay {delay}, using default datasets")
                        dataset_ids = ['fundamental6', 'fundamental2', 'analyst4', 'model16', 'model51', 'news12']
                else:
                    print(f"Failed to get datasets for region {region} with delay {delay}: {datasets_response.text[:500]}")
                    # Fall back to default datasets
                    dataset_ids = ['fundamental6', 'fundamental2', 'analyst4', 'model16', 'model51', 'news12']
                
                print(f"Requesting data fields from available datasets for delay {delay}...")
                for dataset in dataset_ids:
                    # First get the count
                    params = base_params.copy()
                    params['dataset.id'] = dataset
                    params['limit'] = 1  # Just to get count efficiently
                    
                    print(f"Getting field count for dataset: {dataset}")
                    count_response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params)
                    
                    if count_response.status_code == 200:
                        count_data = count_response.json()
                        total_fields = count_data.get('count', 0)
                        print(f"Total fields in {dataset}: {total_fields}")
                        
                        if total_fields > 0:
                            # Generate random offset
                            max_offset = max(0, total_fields - base_params['limit'])
                            random_offset = random.randint(0, max_offset)
                            
                            # Fetch random subset
                            params['offset'] = random_offset
                            params['limit'] = min(20, total_fields)  # Don't exceed total fields
                            
                            print(f"Fetching fields for {dataset} with offset {random_offset}")
                            response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params)
                            
                            if response.status_code == 200:
                                data = response.json()
                                fields = data.get('results', [])
                                print(f"Found {len(fields)} fields in {dataset}")
                                all_fields.extend(fields)
                            else:
                                print(f"Failed to fetch fields for {dataset}: {response.text[:500]}")
                    else:
                        print(f"Failed to get count for {dataset}: {count_response.text[:500]}")
                        
            except Exception as e:
                logger.error(f"Failed to fetch data fields for delay {delay}: {e}")
                continue
        
        # Remove duplicates if any
        unique_fields = {field['id']: field for field in all_fields}.values()
        print(f"Total unique fields found: {len(unique_fields)}")
        return list(unique_fields)

    def get_operators(self) -> List[Dict]:
        """Fetch available operators from WorldQuant Brain."""
        print("Requesting operators...")
        response = self.sess.get('https://api.worldquantbrain.com/operators')
        print(f"Operators response status: {response.status_code}")
        print(f"Operators response: {response.text[:500]}...")  # Print first 500 chars
        
        if response.status_code != 200:
            raise Exception(f"Failed to get operators: {response.text}")
        
        data = response.json()
        # The operators endpoint might return a direct array instead of an object with 'items' or 'results'
        if isinstance(data, list):
            return data
        elif 'results' in data:
            return data['results']
        else:
            raise Exception(f"Unexpected operators response format. Response: {data}")

    def clean_alpha_ideas(self, ideas: List[str]) -> List[str]:
        """Clean and validate alpha ideas, keeping only valid expressions."""
        cleaned_ideas = []
        
        for idea in ideas:
            # Skip if idea is just a number or single word
            if re.match(r'^\d+\.?$|^[a-zA-Z]+$', idea):
                continue
            
            # Skip if idea is a description (contains common English words)
            common_words = ['it', 'the', 'is', 'are', 'captures', 'provides', 'measures']
            if any(word in idea.lower() for word in common_words):
                continue
            
            # Verify idea contains valid operators/functions
            valid_functions = ['ts_mean', 'divide', 'subtract', 'add', 'multiply', 'zscore', 
                              'ts_rank', 'ts_std_dev', 'rank', 'log', 'sqrt']
            if not any(func in idea for func in valid_functions):
                continue
            
            cleaned_ideas.append(idea)  # Just append the expression string
        
        return cleaned_ideas

    def _fix_expression_syntax(self, expression: str) -> str:
        """Fix common syntax issues in alpha expressions."""
        import re
        
        # Fix array brackets [20] -> 20
        expression = re.sub(r'\[(\d+)\]', r'\1', expression)
        
        # Fix multiple parameters in brackets [25, 60] -> 25 (take first parameter)
        expression = re.sub(r'\[(\d+),\s*\d+\]', r'\1', expression)
        
        # Fix any remaining array syntax
        expression = re.sub(r'\[([^\]]+)\]', r'\1', expression)
        
        # Fix common operator syntax issues
        expression = expression.replace('ts_std_dev', 'ts_stddev')
        expression = expression.replace('ts_stddev', 'ts_std_dev')
        
        # Remove any trailing commas
        expression = re.sub(r',\s*\)', ')', expression)
        
        # Fix named parameters like lookback_days=90 -> 90
        expression = re.sub(r'lookback_days=(\d+)', r'\1', expression)
        expression = re.sub(r'lambda_max=(\d+)', r'\1', expression)
        expression = re.sub(r'window=(\d+)', r'\1', expression)
        expression = re.sub(r'period=(\d+)', r'\1', expression)
        
        # Fix missing commas between parameters
        # Pattern: field1 field2 number) -> field1, field2, number)
        expression = re.sub(r'(\w+)\s+(\w+)\s+(\d+)\)', r'\1, \2, \3)', expression)
        
        return expression.strip()

    def generate_alpha_ideas_with_ollama(self, data_fields: List[Dict], operators: List[Dict]) -> List[str]:
        """Generate alpha ideas using Ollama with FinGPT model."""
        print("Organizing operators by category...")
        operator_by_category = {}
        for op in operators:
            category = op['category']
            if category not in operator_by_category:
                operator_by_category[category] = []
            operator_by_category[category].append({
                'name': op['name'],
                'type': op.get('type', 'SCALAR'),
                'definition': op['definition'],
                'description': op['description']
            })

        try:
            # Clear tested expressions if we hit token limit in previous attempt
            if hasattr(self, '_hit_token_limit'):
                logger.info("Clearing tested expressions due to previous token limit")
                self.results = []
                delattr(self, '_hit_token_limit')

            # Randomly sample ~60% of operators from each category
            sampled_operators = {}
            for category, ops in operator_by_category.items():
                sample_size = max(1, int(len(ops) * 0.5))  # At least 1 operator per category
                sampled_operators[category] = random.sample(ops, sample_size)

            print("Preparing prompt for FinGPT...")
            # Format operators with their types, definitions, and descriptions
            def format_operators(ops):
                formatted = []
                for op in ops:
                    formatted.append(f"{op['name']} ({op['type']})\n"
                                   f"  Definition: {op['definition']}\n"
                                   f"  Description: {op['description']}")
                return formatted

            # Format fields and operators with descriptions
            fields_formatted = [f"{field['id']} ({field.get('description', 'No description')})" for field in data_fields]
            operators_formatted = []
            for op in sampled_operators.get('Time Series', []):
                operators_formatted.append(f"{op['name']} ({op.get('description', 'No description')})")
            
            prompt = f"""Generate 100 WorldQuant Brain alpha expressions.

Fields: {fields_formatted}
Operators: {operators_formatted}

CRITICAL SYNTAX RULES:
1. Use only provided fields and operators
2. Return ONLY expressions, one per line
3. No text, no explanations
4. CORRECT syntax: operator(field_id, number) - use numbers like 20, 60, NOT [20] or [60]
5. WRONG: ts_std_dev(field, [20]) 
6. CORRECT: ts_std_dev(field, 20)
7. Use the descriptions to understand how each component works

Example: ts_rank(act_12m_eps_value, 60)

Generate 100 expressions:"""

            # Prepare Ollama API request
            model_name = getattr(self, 'model_name', self.model_fleet[self.current_model_index])
            ollama_data = {
                'model': model_name,
                'prompt': prompt,
                'stream': False,
                'think': False,
                'temperature': 0.3,
                'top_p': 0.9,
                'num_predict': 1000  # Use num_predict instead of max_tokens for Ollama
            }

            print("Sending request to Ollama API...")
            try:
                response = requests.post(
                    f'{self.ollama_url}/api/generate',
                    json=ollama_data,
                    timeout=1800  # 30 minutes timeout
                )

                print(f"Ollama API response status: {response.status_code}")
                print(f"Ollama API response: {response.text[:500]}...")  # Print first 500 chars

                if response.status_code == 500:
                    logging.error(f"Ollama API returned 500 error: {response.text}")
                    # Trigger model downgrade for 500 errors
                    self._handle_ollama_error("500_error")
                    return []
                elif response.status_code != 200:
                    raise Exception(f"Ollama API request failed: {response.text}")
                    
            except requests.exceptions.Timeout:
                logging.error("Ollama API request timed out (1800s)")
                # Trigger model downgrade for timeouts
                self._handle_ollama_error("timeout")
                return []
            except requests.exceptions.ConnectionError as e:
                if "Read timed out" in str(e):
                    logging.error("Ollama API read timeout")
                    # Trigger model downgrade for read timeouts
                    self._handle_ollama_error("read_timeout")
                    return []
                else:
                    raise e

            response_data = response.json()
            print(f"Ollama API response JSON keys: {list(response_data.keys())}")

            if 'response' not in response_data:
                raise Exception(f"Unexpected Ollama API response format: {response_data}")

            print("Processing Ollama API response...")
            content = response_data['response']
            
            # Extract pure alpha expressions by:
            # 1. Remove thinking tags and explanations
            # 2. Remove markdown backticks
            # 3. Remove numbering and comments
            # 4. Keep only valid expressions
            alpha_ideas = []
            
            # Remove thinking tags first
            if '<think>' in content:
                # Extract only the part after the last </think> tag
                parts = content.split('</think>')
                if len(parts) > 1:
                    content = parts[-1].strip()
            
            for line in content.split('\n'):
                line = line.strip()
                
                # Skip empty lines, comments, and thinking parts
                if not line or line.startswith('#') or line.startswith('*') or line.startswith('Comment:'):
                    continue
                
                # Skip thinking and explanation lines
                if any(line.startswith(prefix) for prefix in [
                    'Hmm,', 'Okay,', 'I need', 'Let me', 'First,', 'Then,', 'Finally,',
                    'The expression', 'Here is', 'This is', 'Let me think', 'I think',
                    'Based on', 'Using the', 'Combining', 'Creating', 'Generating',
                    'python', 'def ', 'import ', 'Expression:', 'Answer:', 'Result:', 'Output:'
                ]):
                    continue
                
                # Remove markdown backticks
                line = line.replace('`', '')
                
                # Remove numbering (e.g., "1. ", "2. ")
                if '. ' in line and line[0].isdigit():
                    line = line.split('. ', 1)[1]
                
                # Only keep lines that look like actual expressions (contain parentheses and operators)
                if line and '(' in line and ')' in line and any(op in line for op in ['ts_', 'rank', 'divide', 'multiply', 'add', 'subtract', 'winsorize']):
                    # Fix common syntax issues
                    line = self._fix_expression_syntax(line)
                    alpha_ideas.append(line)
            
            print(f"Generated {len(alpha_ideas)} alpha ideas")
            for i, alpha in enumerate(alpha_ideas, 1):
                print(f"Alpha {i}: {alpha}")
            
            # Clean and validate ideas
            cleaned_ideas = self.clean_alpha_ideas(alpha_ideas)
            logging.info(f"Found {len(cleaned_ideas)} valid alpha expressions")
            
            return cleaned_ideas

        except Exception as e:
            if "token limit" in str(e).lower():
                self._hit_token_limit = True  # Mark that we hit token limit
            logging.error(f"Error generating alpha ideas: {str(e)}")
            return []
    
    def _handle_ollama_error(self, error_type: str):
        """Handle Ollama errors by downgrading model if needed."""
        self.error_count += 1
        logging.warning(f"Ollama error ({error_type}) - Count: {self.error_count}/{self.max_errors_before_downgrade}")
        
        if self.error_count >= self.max_errors_before_downgrade:
            self._downgrade_model()
            self.error_count = 0  # Reset error count after downgrade
    
    def _downgrade_model(self):
        """Downgrade to the next smaller model in the fleet."""
        if self.current_model_index >= len(self.model_fleet) - 1:
            logging.error("Already using the smallest model in the fleet!")
            # Reset to initial model if we've exhausted all options
            self.current_model_index = 0
            self.model_name = self.initial_model
            logging.info(f"Reset to initial model: {self.initial_model}")
            return
        
        old_model = self.model_fleet[self.current_model_index]
        self.current_model_index += 1
        new_model = self.model_fleet[self.current_model_index]
        
        logging.warning(f"Downgrading model: {old_model} -> {new_model}")
        self.model_name = new_model
        
        # Update the model in the orchestrator if it exists
        try:
            # Try to update the orchestrator's model fleet manager
            if hasattr(self, 'orchestrator') and hasattr(self.orchestrator, 'model_fleet_manager'):
                self.orchestrator.model_fleet_manager.current_model_index = self.current_model_index
                self.orchestrator.model_fleet_manager.save_state()
                logging.info(f"Updated orchestrator model fleet to use: {new_model}")
        except Exception as e:
            logging.warning(f"Could not update orchestrator model fleet: {e}")
        
        logging.info(f"Successfully downgraded to {new_model}")

    def test_alpha_batch(self, alphas: List[str]) -> None:
        """Submit a batch of alphas for testing with monitoring, respecting concurrent limits."""
        logging.info(f"Starting batch test of {len(alphas)} alphas")
        for alpha in alphas:
            logging.info(f"Alpha expression: {alpha}")
        
        # Check if multi-simulate is enabled
        if hasattr(self, 'multi_simulate_enabled') and self.multi_simulate_enabled:
            # Prepare alphas for multi-simulation
            alpha_list = [(alpha, 0) for alpha in alphas]  # Add decay parameter
            batch_size_sim = getattr(self, 'batch_size_sim', 10)
            concurrent_batches = getattr(self, 'concurrent_batches', 10)
            pools = self.load_task_pool(alpha_list, batch_size_sim, concurrent_batches)
            logging.info(f"Created {len(pools)} pools for multi-simulation")
            
            # Run multi-simulations
            logging.info("Starting multi-simulations...")
            self.multi_simulate(pools, "INDUSTRY", "USA", "TOP3000", 0)
            
            logging.info(f"Multi-simulation complete for {len(alphas)} alphas")
            return len(alphas)  # Return number of alphas processed
        else:
            # Fall back to original single simulation method
            logging.info("Using single simulation method (multi-simulate disabled)")
            return self._test_alpha_batch_original(alphas)

    def _test_alpha_batch_original(self, alphas: List[str]) -> int:
        """Original batch processing method for single simulations."""
        logging.info(f"Processing batch of {len(alphas)} alphas using single simulations")
        
        # Submit alphas in smaller chunks to respect concurrent limits
        max_concurrent = self.executor._max_workers
        submitted = 0
        queued = 0
        
        for i in range(0, len(alphas), max_concurrent):
            chunk = alphas[i:i + max_concurrent]
            logging.info(f"Submitting chunk {i//max_concurrent + 1}/{(len(alphas)-1)//max_concurrent + 1} ({len(chunk)} alphas)")
            
            # Submit chunk
            futures = []
            for j, alpha in enumerate(chunk, 1):
                logging.info(f"Submitting alpha {i+j}/{len(alphas)}")
                future = self.executor.submit(self._test_alpha_impl, alpha)
                futures.append((alpha, future))
            
            # Process results for this chunk
            for alpha, future in futures:
                try:
                    result = future.result()
                    if result.get("status") == "error":
                        if "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
                            self.retry_queue.add(alpha)
                            queued += 1
                            logging.info(f"Queued for retry: {alpha}")
                        else:
                            logging.error(f"Simulation error for {alpha}: {result.get('message')}")
                        continue
                        
                    sim_id = result.get("result", {}).get("id")
                    progress_url = result.get("result", {}).get("progress_url")
                    if sim_id and progress_url:
                        self.pending_results[sim_id] = {
                            "alpha": alpha,
                            "progress_url": progress_url,
                            "status": "pending",
                            "attempts": 0
                        }
                        submitted += 1
                        logging.info(f"Successfully submitted {alpha} (ID: {sim_id})")
                        
                except Exception as e:
                    logging.error(f"Error submitting alpha {alpha}: {str(e)}")
            
            # Wait between chunks to avoid overwhelming the API
            if i + max_concurrent < len(alphas):
                logging.info(f"Waiting 10 seconds before next chunk...")
                sleep(10)
        
        logging.info(f"Batch submission complete: {submitted} submitted, {queued} queued for retry")
        
        # Monitor progress until all complete or need retry
        total_successful = 0
        max_monitoring_time = 3600  # 1 hour maximum monitoring time
        start_time = time.time()
        
        while self.pending_results:
            # Check for timeout
            if time.time() - start_time > max_monitoring_time:
                logging.warning(f"Monitoring timeout reached ({max_monitoring_time}s), stopping monitoring")
                logging.warning(f"Remaining pending simulations: {list(self.pending_results.keys())}")
                break
                
            logging.info(f"Monitoring {len(self.pending_results)} pending simulations...")
            completed = self.check_pending_results()
            total_successful += completed
            sleep(5)  # Wait between checks
        
        logging.info(f"Batch complete: {total_successful} successful simulations")
        return total_successful

    def check_pending_results(self) -> int:
        """Check status of all pending simulations with proper retry handling."""
        completed = []
        retry_queue = []
        successful = 0
        
        for sim_id, info in self.pending_results.items():
            if info["status"] == "pending":
                # Check if simulation has been pending too long (30 minutes)
                if "start_time" not in info:
                    info["start_time"] = time.time()
                elif time.time() - info["start_time"] > 1800:  # 30 minutes
                    logging.warning(f"Simulation {sim_id} has been pending for too long, marking as failed")
                    completed.append(sim_id)
                    continue
                try:
                    sim_progress_resp = self.sess.get(info["progress_url"])
                    logging.info(f"Checking simulation {sim_id} for alpha: {info['alpha'][:50]}...")
                    
                    # Handle rate limits
                    if sim_progress_resp.status_code == 429:
                        logging.info("Rate limit hit, will retry later")
                        continue
                        
                    # Handle simulation limits
                    if "SIMULATION_LIMIT_EXCEEDED" in sim_progress_resp.text:
                        logging.info(f"Simulation limit exceeded for alpha: {info['alpha']}")
                        retry_queue.append((info['alpha'], sim_id))
                        continue
                        
                    # Handle retry-after
                    retry_after = sim_progress_resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = int(float(retry_after))  # Handle decimal values like "2.5"
                            logging.info(f"Need to wait {wait_time}s before next check")
                            time.sleep(wait_time)
                        except (ValueError, TypeError):
                            logging.warning(f"Invalid Retry-After header: {retry_after}, using default 5s")
                            time.sleep(5)
                        continue
                    
                    sim_result = sim_progress_resp.json()
                    status = sim_result.get("status")
                    logging.info(f"Simulation {sim_id} status: {status}")
                    
                    # Log additional details for debugging
                    if status == "PENDING":
                        logging.debug(f"Simulation {sim_id} still pending...")
                    elif status == "RUNNING":
                        logging.debug(f"Simulation {sim_id} is running...")
                    elif status not in ["COMPLETE", "ERROR"]:
                        logging.warning(f"Simulation {sim_id} has unknown status: {status}")
                    
                    if status == "COMPLETE":
                        alpha_id = sim_result.get("alpha")
                        if alpha_id:
                            alpha_resp = self.sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}')
                            if alpha_resp.status_code == 200:
                                alpha_data = alpha_resp.json()
                                fitness = alpha_data.get("is", {}).get("fitness")
                                logging.info(f"Alpha {alpha_id} completed with fitness: {fitness}")
                                
                                self.results.append({
                                    "alpha": info["alpha"],
                                    "result": sim_result,
                                    "alpha_data": alpha_data
                                })
                                
                                # Check if fitness is not None and greater than threshold
                                if fitness is not None and fitness > 0.5:
                                    logging.info(f"Found promising alpha! Fitness: {fitness}")
                                    self.log_hopeful_alpha(info["alpha"], alpha_data)
                                    successful += 1
                                elif fitness is None:
                                    logging.warning(f"Alpha {alpha_id} has no fitness data, skipping hopeful alpha logging")
                    elif status == "ERROR":
                        logging.error(f"Simulation failed for alpha: {info['alpha']}")
                    completed.append(sim_id)
                    
                except Exception as e:
                    logging.error(f"Error checking result for {sim_id}: {str(e)}")
        
        # Remove completed simulations
        for sim_id in completed:
            del self.pending_results[sim_id]
        
        # Requeue failed simulations
        for alpha, sim_id in retry_queue:
            del self.pending_results[sim_id]
            self.retry_queue.add(alpha)
        
        return successful

    def test_alpha(self, alpha: str) -> Dict:
        result = self._test_alpha_impl(alpha)
        if result.get("status") == "error" and "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
            self.retry_queue.add(alpha)
            return {"status": "queued", "message": "Added to retry queue"}
        return result

    def _test_alpha_impl(self, alpha_expression: str) -> Dict:
        """Implementation of alpha testing with proper URL handling."""
        def submit_simulation():
            simulation_data = {
                'type': 'REGULAR',
                'settings': {
                    'instrumentType': 'EQUITY',
                    'region': 'USA',
                    'universe': 'TOP3000',
                    'delay': 1,
                    'decay': 0,
                    'neutralization': 'INDUSTRY',
                    'truncation': 0.08,
                    'pasteurization': 'ON',
                    'unitHandling': 'VERIFY',
                    'nanHandling': 'OFF',
                    'language': 'FASTEXPR',
                    'visualization': False,
                },
                'regular': alpha_expression
            }
            return self.sess.post('https://api.worldquantbrain.com/simulations', json=simulation_data)

        try:
            sim_resp = submit_simulation()
            
            # Handle authentication error
            if sim_resp.status_code == 401 or (
                sim_resp.status_code == 400 and 
                "authentication credentials" in sim_resp.text.lower()
            ):
                logger.warning("Authentication expired, refreshing session...")
                self.setup_auth(self.credentials_path)  # Refresh authentication
                sim_resp = submit_simulation()  # Retry with new auth
            
            if sim_resp.status_code != 201:
                return {"status": "error", "message": sim_resp.text}

            sim_progress_url = sim_resp.headers.get('location')
            if not sim_progress_url:
                return {"status": "error", "message": "No progress URL received"}

            return {
                "status": "success", 
                "result": {
                    "id": f"{time.time()}_{random.random()}",
                    "progress_url": sim_progress_url
                }
            }
            
        except Exception as e:
            logger.error(f"Error testing alpha {alpha_expression}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def multi_simulate(self, alpha_pools: list, neut: str = "INDUSTRY", region: str = "USA", universe: str = "TOP3000", start: int = 0):
        """Run multiple alpha simulations in parallel using multi-simulate functionality."""
        logging.info(f"Starting multi-simulate for {len(alpha_pools)} pools")
        
        for x, pool in enumerate(alpha_pools):
            if x < start:
                continue
                
            progress_urls = []
            logging.info(f"Processing pool {x+1}/{len(alpha_pools)}")
            
            for y, task in enumerate(pool):
                sim_data_list = self.generate_sim_data(task, region, universe, neut)
                logging.info(f"Generated simulation data for task {y+1}/{len(pool)}")
                
                try:
                    # If only one simulation, send it directly without wrapping array
                    if len(sim_data_list) == 1:
                        simulation_response = self.sess.post('https://api.worldquantbrain.com/simulations', 
                                                         json=sim_data_list[0])
                    else:
                        simulation_response = self.sess.post('https://api.worldquantbrain.com/simulations', 
                                                         json=sim_data_list)
                    
                    if simulation_response.status_code == 401:
                        logging.info("Session expired, re-authenticating...")
                        self.setup_auth(self.credentials_path)
                        if len(sim_data_list) == 1:
                            simulation_response = self.sess.post('https://api.worldquantbrain.com/simulations', 
                                                            json=sim_data_list[0])
                        else:
                            simulation_response = self.sess.post('https://api.worldquantbrain.com/simulations', 
                                                            json=sim_data_list)
                    
                    if simulation_response.status_code != 201:
                        logging.error(f"Simulation API error: {simulation_response.text}")
                        continue
                        
                    simulation_progress_url = simulation_response.headers.get('Location')
                    if not simulation_progress_url:
                        logging.error("No Location header in response")
                        continue
                        
                    progress_urls.append(simulation_progress_url)
                    logging.info(f"Posted simulation for task {y+1}, got progress URL: {simulation_progress_url}")
                    
                except Exception as e:
                    logging.error(f"Error posting simulation: {str(e)}")
                    sleep(600)
                    self.setup_auth(self.credentials_path)
                    continue

            self._monitor_progress(progress_urls)
            logging.info(f"Pool {x+1} simulations completed")

    def _monitor_progress(self, progress_urls: list):
        """Monitor simulation progress."""
        for j, progress in enumerate(progress_urls):
            try:
                while True:
                    simulation_progress = self.sess.get(progress)
                    retry_after = simulation_progress.headers.get("Retry-After", 0)
                    if not retry_after:
                        break
                    sleep(float(retry_after))

                status = simulation_progress.json().get("status")
                logging.info(f"Task {j+1} status: {status}")
                if status != "COMPLETE":
                    logging.warning(f"Task not complete: {progress}")

            except Exception as e:
                logging.error(f"Error monitoring progress: {str(e)}")

    def generate_sim_data(self, alpha_list, region, uni, neut):
        """Generate simulation data for multi-simulate."""
        sim_data_list = []
        for alpha, decay in alpha_list:
            simulation_data = {
                'type': 'REGULAR',
                'settings': {
                    'instrumentType': 'EQUITY',
                    'region': region,
                    'universe': uni,
                    'delay': 1,
                    'decay': decay,
                    'neutralization': neut,
                    'truncation': 0.08,
                    'pasteurization': 'ON',
                    'unitHandling': 'VERIFY',
                    'nanHandling': 'OFF',
                    'language': 'FASTEXPR',
                    'visualization': False,
                },
                'regular': alpha}

            sim_data_list.append(simulation_data)
        return sim_data_list

    def load_task_pool(self, alpha_list: list, batch_size: int = 10, concurrent_batches: int = 10) -> list:
        """Split alpha list into pools of batches for concurrent processing."""
        pools = []
        current_pool = []
        current_batch = []
        
        for alpha in alpha_list:
            current_batch.append(alpha)
            
            if len(current_batch) >= batch_size:
                current_pool.append(current_batch)
                current_batch = []
                
                if len(current_pool) >= concurrent_batches:
                    pools.append(current_pool)
                    current_pool = []
        
        # Add any remaining batches
        if current_batch:
            current_pool.append(current_batch)
        if current_pool:
            pools.append(current_pool)
        
        logging.info(f"Created {len(pools)} pools with {batch_size} alphas per batch")
        return pools

    def log_hopeful_alpha(self, expression: str, alpha_data: Dict) -> None:
        """Log promising alphas to a JSON file."""
        log_file = 'hopeful_alphas.json'
        
        # Load existing data
        existing_data = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {log_file}, starting fresh")
        
        # Add new alpha with timestamp
        entry = {
            "expression": expression,  # Store just the expression string
            "timestamp": int(time.time()),
            "alpha_id": alpha_data.get("id", "unknown"),
            "fitness": alpha_data.get("is", {}).get("fitness"),
            "sharpe": alpha_data.get("is", {}).get("sharpe"),
            "turnover": alpha_data.get("is", {}).get("turnover"),
            "returns": alpha_data.get("is", {}).get("returns"),
            "grade": alpha_data.get("grade", "UNKNOWN"),
            "checks": alpha_data.get("is", {}).get("checks", [])
        }
        
        existing_data.append(entry)
        
        # Save updated data
        with open(log_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print(f"Logged promising alpha to {log_file}")

    def get_results(self) -> List[Dict]:
        """Get all processed results including retried alphas."""
        return self.results

    def fetch_submitted_alphas(self):
        """Fetch submitted alphas from the WorldQuant API with retry logic"""
        url = "https://api.worldquantbrain.com/users/self/alphas"
        params = {
            "limit": 100,
            "offset": 0,
            "status!=": "UNSUBMITTED%1FIS-FAIL",
            "order": "-dateCreated",
            "hidden": "false"
        }
        
        max_retries = 3
        retry_delay = 60  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.sess.get(url, params=params)
                if response.status_code == 429:  # Too Many Requests
                    wait_time = int(response.headers.get('Retry-After', retry_delay))
                    logger.info(f"Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                return response.json()["results"]
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to fetch submitted alphas after {max_retries} attempts: {e}")
                    return []
        
        return []

def extract_expressions(alphas):
    """Extract expressions from submitted alphas"""
    expressions = []
    for alpha in alphas:
        if alpha.get("regular") and alpha["regular"].get("code"):
            expressions.append({
                "expression": alpha["regular"]["code"],
                "performance": {
                    "sharpe": alpha["is"].get("sharpe", 0),
                    "fitness": alpha["is"].get("fitness", 0)
                }
            })
    return expressions

def is_similar_to_existing(new_expression, existing_expressions, similarity_threshold=0.7):
    """Check if new expression is too similar to existing ones"""
    for existing in existing_expressions:
        # Basic similarity checks
        if new_expression == existing["expression"]:
            return True
            
        # Check for structural similarity
        if structural_similarity(new_expression, existing["expression"]) > similarity_threshold:
            return True
    
    return False

def calculate_similarity(expr1: str, expr2: str) -> float:
    """Calculate similarity between two expressions using token-based comparison."""
    # Normalize expressions
    expr1_tokens = set(tokenize_expression(normalize_expression(expr1)))
    expr2_tokens = set(tokenize_expression(normalize_expression(expr2)))
    
    if not expr1_tokens or not expr2_tokens:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(expr1_tokens.intersection(expr2_tokens))
    union = len(expr1_tokens.union(expr2_tokens))
    
    return intersection / union

def structural_similarity(expr1, expr2):
    """Calculate structural similarity between two expressions"""
    return calculate_similarity(expr1, expr2)  # Use our new similarity function

def normalize_expression(expr):
    """Normalize expression for comparison"""
    # Remove whitespace and convert to lowercase
    expr = re.sub(r'\s+', '', expr.lower())
    return expr

def tokenize_expression(expr):
    """Split expression into meaningful tokens"""
    # Split on operators and parentheses while keeping them
    tokens = re.findall(r'[\w._]+|[(),*/+-]', expr)
    return tokens

def generate_alpha():
    """Generate new alpha expression"""
    generator = AlphaGenerator("./credential.txt", "http://localhost:11434")
    data_fields = generator.get_data_fields()
    operators = generator.get_operators()
    
    # Fetch existing alphas first
    submitted_alphas = generator.fetch_submitted_alphas()
    existing_expressions = extract_expressions(submitted_alphas)
    
    max_attempts = 50
    attempts = 0
    
    while attempts < max_attempts:
        alpha_ideas = generator.generate_alpha_ideas_with_ollama(data_fields, operators)
        for idea in alpha_ideas:
            if not is_similar_to_existing(idea, existing_expressions):
                logger.info(f"Generated unique expression: {idea}")
                return idea
                
        attempts += 1
        logger.debug(f"Attempt {attempts}: All expressions were too similar")
    
    logger.warning("Failed to generate unique expression after maximum attempts")
    return None

def main():
    parser = argparse.ArgumentParser(description='Generate and test alpha factors using WorldQuant Brain API with Ollama/FinGPT')
    parser.add_argument('--credentials', type=str, default='./credential.txt',
                      help='Path to credentials file (default: ./credential.txt)')
    parser.add_argument('--output-dir', type=str, default='./results',
                      help='Directory to save results (default: ./results)')
    parser.add_argument('--batch-size', type=int, default=3,
                      help='Number of alpha factors to generate per batch (default: 3)')
    parser.add_argument('--sleep-time', type=int, default=10,
                      help='Sleep time between batches in seconds (default: 10)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level (default: INFO)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                      help='Ollama API URL (default: http://localhost:11434)')
    parser.add_argument('--ollama-model', type=str, default='deepseek-r1:8b',
                                             help='Ollama model to use (default: deepseek-r1:8b for RTX A4000)')
    parser.add_argument('--max-concurrent', type=int, default=2,
                      help='Maximum concurrent simulations (default: 2)')
    parser.add_argument('--multi-simulate', type=str, default='false',
                      help='Enable multi-simulate functionality (default: false)')
    parser.add_argument('--batch-size-sim', type=int, default=10,
                      help='Batch size for multi-simulate (default: 10)')
    parser.add_argument('--concurrent-batches', type=int, default=10,
                      help='Number of concurrent batches for multi-simulate (default: 10)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler('alpha_generator_ollama.log')  # Also log to file
        ]
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize alpha generator with Ollama
        generator = AlphaGenerator(args.credentials, args.ollama_url, args.max_concurrent)
        generator.model_name = args.ollama_model  # Set the model name
        generator.initial_model = args.ollama_model  # Set the initial model for reset
        
        # Set multi-simulate parameters
        generator.multi_simulate_enabled = args.multi_simulate.lower() == 'true'
        generator.batch_size_sim = args.batch_size_sim
        generator.concurrent_batches = args.concurrent_batches
        
        if generator.multi_simulate_enabled:
            logging.info(f"Multi-simulate enabled: batch_size={args.batch_size_sim}, concurrent_batches={args.concurrent_batches}")
        else:
            logging.info("Multi-simulate disabled, using single simulation method")
        
        # Get data fields and operators once
        print("Fetching data fields and operators...")
        data_fields = generator.get_data_fields()
        operators = generator.get_operators()
        
        batch_number = 1
        total_successful = 0
        
        print(f"Starting continuous alpha mining with batch size {args.batch_size}")
        print(f"Results will be saved to {args.output_dir}")
        print(f"Using Ollama at {args.ollama_url}")
        
        while True:
            try:
                logging.info(f"\nProcessing batch #{batch_number}")
                logging.info("-" * 50)
                
                # Generate and submit batch using Ollama
                alpha_ideas = generator.generate_alpha_ideas_with_ollama(data_fields, operators)
                batch_successful = generator.test_alpha_batch(alpha_ideas)
                total_successful += batch_successful
                
                # Perform VRAM cleanup every few batches
                generator.operation_count += 1
                if generator.operation_count % generator.vram_cleanup_interval == 0:
                    generator.cleanup_vram()
                
                # Save batch results
                results = generator.get_results()
                timestamp = int(time.time())
                output_file = os.path.join(args.output_dir, f'batch_{batch_number}_{timestamp}.json')
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logging.info(f"Batch {batch_number} results saved to {output_file}")
                logging.info(f"Batch successful: {batch_successful}")
                logging.info(f"Total successful alphas: {total_successful}")
                
                batch_number += 1
                
                # Continue immediately to next batch
                print(f"Continuing to next batch immediately...")
                
            except Exception as e:
                logging.error(f"Error in batch {batch_number}: {str(e)}")
                logging.info("Sleeping for 5 minutes before retrying...")
                sleep(300)
                continue
        
    except KeyboardInterrupt:
        logging.info("\nStopping alpha mining...")
        logging.info(f"Total batches processed: {batch_number - 1}")
        logging.info(f"Total successful alphas: {total_successful}")
        return 0
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
