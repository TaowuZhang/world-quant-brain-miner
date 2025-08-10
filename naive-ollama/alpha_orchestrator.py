import argparse
import requests
import json
import os
import time
import logging
import schedule
from datetime import datetime, timedelta
from typing import List, Dict
from requests.auth import HTTPBasicAuth
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('alpha_orchestrator.log')
    ]
)
logger = logging.getLogger(__name__)

class AlphaOrchestrator:
    def __init__(self, credentials_path: str, ollama_url: str = "http://localhost:11434"):
        self.sess = requests.Session()
        self.credentials_path = credentials_path
        self.ollama_url = ollama_url
        self.setup_auth(credentials_path)
        self.last_submission_date = None
        self.submission_log_file = "submission_log.json"
        self.load_submission_history()
        
    def setup_auth(self, credentials_path: str) -> None:
        """Set up authentication with WorldQuant Brain."""
        logger.info(f"Loading credentials from {credentials_path}")
        with open(credentials_path) as f:
            credentials = json.load(f)
        
        username, password = credentials
        self.sess.auth = HTTPBasicAuth(username, password)
        
        logger.info("Authenticating with WorldQuant Brain...")
        response = self.sess.post('https://api.worldquantbrain.com/authentication')
        logger.info(f"Authentication response status: {response.status_code}")
        
        if response.status_code != 201:
            raise Exception(f"Authentication failed: {response.text}")
        logger.info("Authentication successful")

    def load_submission_history(self):
        """Load submission history to track daily submissions."""
        if os.path.exists(self.submission_log_file):
            try:
                with open(self.submission_log_file, 'r') as f:
                    data = json.load(f)
                    self.last_submission_date = data.get('last_submission_date')
                    logger.info(f"Loaded submission history. Last submission: {self.last_submission_date}")
            except Exception as e:
                logger.warning(f"Could not load submission history: {e}")
                self.last_submission_date = None
        else:
            self.last_submission_date = None

    def save_submission_history(self):
        """Save submission history."""
        data = {
            'last_submission_date': self.last_submission_date,
            'updated_at': datetime.now().isoformat()
        }
        with open(self.submission_log_file, 'w') as f:
            json.dump(data, f, indent=2)

    def can_submit_today(self) -> bool:
        """Check if we can submit alphas today (only once per day)."""
        today = datetime.now().date().isoformat()
        
        if self.last_submission_date == today:
            logger.info(f"Already submitted today ({today}). Skipping submission.")
            return False
        
        logger.info(f"Can submit today. Last submission was: {self.last_submission_date}")
        return True

    def run_alpha_expression_miner(self, promising_alpha_file: str = "hopeful_alphas.json"):
        """Run alpha expression miner on promising alphas."""
        logger.info("Starting alpha expression miner on promising alphas...")
        
        if not os.path.exists(promising_alpha_file):
            logger.warning(f"Promising alphas file {promising_alpha_file} not found. Skipping mining.")
            return
        
        try:
            with open(promising_alpha_file, 'r') as f:
                promising_alphas = json.load(f)
            
            if not promising_alphas:
                logger.info("No promising alphas found. Skipping mining.")
                return
            
            logger.info(f"Found {len(promising_alphas)} promising alphas to mine")
            
            # Run alpha expression miner for each promising alpha
            for i, alpha_data in enumerate(promising_alphas, 1):
                expression = alpha_data.get('expression', '')
                if not expression:
                    continue
                
                logger.info(f"Mining alpha {i}/{len(promising_alphas)}: {expression[:100]}...")
                
                # Run the alpha expression miner as a subprocess
                try:
                    result = subprocess.run([
                        sys.executable, 'alpha_expression_miner.py',
                        '--expression', expression,
                        '--auto-mode',  # Run in automated mode
                        '--output-file', f'mining_results_{i}.json'
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        logger.info(f"Successfully mined alpha {i}")
                    else:
                        logger.error(f"Failed to mine alpha {i}: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"Mining alpha {i} timed out")
                except Exception as e:
                    logger.error(f"Error mining alpha {i}: {e}")
                
                # Small delay between mining operations
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"Error running alpha expression miner: {e}")

    def run_alpha_submitter(self, batch_size: int = 5):
        """Run alpha submitter with daily rate limiting."""
        logger.info("Starting alpha submitter...")
        
        if not self.can_submit_today():
            return
        
        try:
            # Run the alpha submitter as a subprocess
            result = subprocess.run([
                sys.executable, 'successful_alpha_submitter.py',
                '--batch-size', str(batch_size),
                '--auto-mode'  # Run in automated mode
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("Successfully completed alpha submission")
                # Update submission date
                self.last_submission_date = datetime.now().date().isoformat()
                self.save_submission_history()
            else:
                logger.error(f"Alpha submission failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("Alpha submission timed out")
        except Exception as e:
            logger.error(f"Error running alpha submitter: {e}")

    def run_alpha_generator(self, batch_size: int = 5, sleep_time: int = 30):
        """Run the main alpha generator with Ollama."""
        logger.info("Starting alpha generator with Ollama...")
        
        try:
            # Run the alpha generator as a subprocess
            result = subprocess.run([
                sys.executable, 'alpha_generator_ollama.py',
                '--batch-size', str(batch_size),
                '--sleep-time', str(sleep_time),
                '--ollama-url', self.ollama_url,
                '--ollama-model', 'llama3.2:3b'
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                logger.info("Alpha generator completed successfully")
            else:
                logger.error(f"Alpha generator failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("Alpha generator timed out")
        except Exception as e:
            logger.error(f"Error running alpha generator: {e}")

    def daily_workflow(self):
        """Run the complete daily workflow."""
        logger.info("Starting daily alpha workflow...")
        
        # 1. Run alpha generator for a few hours
        logger.info("Phase 1: Running alpha generator...")
        self.run_alpha_generator(batch_size=3, sleep_time=60)
        
        # 2. Run alpha expression miner on promising alphas
        logger.info("Phase 2: Running alpha expression miner...")
        self.run_alpha_expression_miner()
        
        # 3. Run alpha submitter (once per day)
        logger.info("Phase 3: Running alpha submitter...")
        self.run_alpha_submitter(batch_size=3)
        
        logger.info("Daily workflow completed")

    def continuous_mining(self, mining_interval_hours: int = 6):
        """Run continuous mining with periodic expression mining and daily submission."""
        logger.info(f"Starting continuous mining with {mining_interval_hours}h intervals...")
        
        # Schedule daily submission at 2 PM
        schedule.every().day.at("14:00").do(self.run_alpha_submitter)
        
        # Schedule expression mining every 6 hours
        schedule.every(mining_interval_hours).hours.do(self.run_alpha_expression_miner)
        
        while True:
            try:
                # Run pending scheduled tasks
                schedule.run_pending()
                
                # Run alpha generator continuously
                logger.info("Running alpha generator cycle...")
                self.run_alpha_generator(batch_size=3, sleep_time=30)
                
                # Small delay before next cycle
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Stopping continuous mining...")
                break
            except Exception as e:
                logger.error(f"Error in continuous mining: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

def main():
    parser = argparse.ArgumentParser(description='Alpha Orchestrator - Manage alpha generation, mining, and submission')
    parser.add_argument('--credentials', type=str, default='./credential.txt',
                      help='Path to credentials file (default: ./credential.txt)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                      help='Ollama API URL (default: http://localhost:11434)')
    parser.add_argument('--mode', type=str, choices=['daily', 'continuous', 'miner', 'submitter', 'generator'],
                      default='continuous', help='Operation mode (default: continuous)')
    parser.add_argument('--mining-interval', type=int, default=6,
                      help='Mining interval in hours for continuous mode (default: 6)')
    parser.add_argument('--batch-size', type=int, default=3,
                      help='Batch size for operations (default: 3)')
    
    args = parser.parse_args()
    
    try:
        orchestrator = AlphaOrchestrator(args.credentials, args.ollama_url)
        
        if args.mode == 'daily':
            orchestrator.daily_workflow()
        elif args.mode == 'continuous':
            orchestrator.continuous_mining(args.mining_interval)
        elif args.mode == 'miner':
            orchestrator.run_alpha_expression_miner()
        elif args.mode == 'submitter':
            orchestrator.run_alpha_submitter(args.batch_size)
        elif args.mode == 'generator':
            orchestrator.run_alpha_generator(args.batch_size)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
