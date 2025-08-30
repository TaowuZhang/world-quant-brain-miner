import argparse
import requests
import json
import os
import time
import logging
import subprocess
import sys
import threading
from typing import List, Dict
from dataclasses import dataclass
import schedule
from datetime import datetime, timedelta

# Import the adaptive miner
from adaptive_alpha_miner import AdaptiveAlphaMiner, AlphaResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integrated_alpha_miner.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MiningConfig:
    """Configuration for integrated mining."""
    adaptive_batch_size: int = 5
    adaptive_iterations: int = 3
    lateral_count: int = 3
    generator_batch_size: int = 10
    generator_sleep_time: int = 30
    mining_interval_hours: int = 6
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:8b"

class IntegratedAlphaMiner:
    """Integrated alpha miner that combines adaptive mining with alpha generation."""
    
    def __init__(self, credentials_path: str, config: MiningConfig):
        self.credentials_path = credentials_path
        self.config = config
        
        # Initialize adaptive miner
        self.adaptive_miner = AdaptiveAlphaMiner(credentials_path, config.ollama_url, config.ollama_model)
        
        # Control flags
        self.running = True
        self.adaptive_mining_active = False
        self.generator_active = False
        
        # Performance tracking
        self.total_adaptive_alphas = 0
        self.total_generator_alphas = 0
        self.best_adaptive_score = 0
        self.best_generator_score = 0
        
        # Load state
        self.load_state()
    
    def load_state(self):
        """Load miner state."""
        try:
            if os.path.exists('integrated_miner_state.json'):
                with open('integrated_miner_state.json', 'r') as f:
                    state = json.load(f)
                    self.total_adaptive_alphas = state.get('total_adaptive_alphas', 0)
                    self.total_generator_alphas = state.get('total_generator_alphas', 0)
                    self.best_adaptive_score = state.get('best_adaptive_score', 0)
                    self.best_generator_score = state.get('best_generator_score', 0)
                    logger.info(f"Loaded state - Adaptive: {self.total_adaptive_alphas}, Generator: {self.total_generator_alphas}")
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
    
    def save_state(self):
        """Save miner state."""
        try:
            state = {
                'total_adaptive_alphas': self.total_adaptive_alphas,
                'total_generator_alphas': self.total_generator_alphas,
                'best_adaptive_score': self.best_adaptive_score,
                'best_generator_score': self.best_generator_score,
                'timestamp': time.time()
            }
            with open('integrated_miner_state.json', 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save state: {e}")
    
    def run_adaptive_mining_session(self):
        """Run a single adaptive mining session."""
        if self.adaptive_mining_active:
            logger.info("Adaptive mining already active, skipping session")
            return
        
        logger.info("Starting adaptive mining session...")
        self.adaptive_mining_active = True
        
        try:
            for i in range(self.config.adaptive_iterations):
                if not self.running:
                    break
                
                logger.info(f"Adaptive mining iteration {i+1}/{self.config.adaptive_iterations}")
                
                # Run adaptive mining batch
                results = self.adaptive_miner.mine_adaptive_batch(self.config.adaptive_batch_size)
                self.total_adaptive_alphas += len(results)
                
                # Update best score
                if self.adaptive_miner.best_alpha:
                    current_score = self.adaptive_miner.best_alpha.sharpe * self.adaptive_miner.best_alpha.fitness
                    if current_score > self.best_adaptive_score:
                        self.best_adaptive_score = current_score
                        logger.info(f"New best adaptive alpha score: {self.best_adaptive_score:.3f}")
                
                # Perform lateral movement on best result
                if results:
                    best_result = max(results, key=lambda r: r.sharpe * r.fitness if r.success else 0)
                    if best_result.success:
                        logger.info(f"Performing lateral movement on best result (Sharpe: {best_result.sharpe:.3f})")
                        lateral_results = self.adaptive_miner.lateral_movement(best_result, self.config.lateral_count)
                        self.total_adaptive_alphas += len(lateral_results)
                
                # Save states
                self.adaptive_miner.save_state()
                self.save_state()
                
                # Sleep between iterations
                if i < self.config.adaptive_iterations - 1 and self.running:
                    logger.info("Sleeping for 120 seconds...")
                    time.sleep(120)
        
        except Exception as e:
            logger.error(f"Error in adaptive mining session: {e}")
        finally:
            self.adaptive_mining_active = False
            logger.info("Adaptive mining session completed")
    
    def run_alpha_generator_session(self):
        """Run a single alpha generator session."""
        if self.generator_active:
            logger.info("Alpha generator already active, skipping session")
            return
        
        logger.info("Starting alpha generator session...")
        self.generator_active = True
        
        try:
            # Run alpha generator with multi-simulate
            cmd = [
                sys.executable, 'alpha_generator_ollama.py',
                '--batch-size', str(self.config.generator_batch_size),
                '--sleep-time', str(self.config.generator_sleep_time),
                '--ollama-url', self.config.ollama_url,
                '--ollama-model', self.config.ollama_model,
                '--multi-simulate', 'true',
                '--batch-size-sim', '10',
                '--concurrent-batches', '10'
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run the generator
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info("Alpha generator session completed successfully")
                self.total_generator_alphas += self.config.generator_batch_size
            else:
                logger.error(f"Alpha generator failed: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            logger.warning("Alpha generator session timed out")
        except Exception as e:
            logger.error(f"Error in alpha generator session: {e}")
        finally:
            self.generator_active = False
            logger.info("Alpha generator session completed")
    
    def start_continuous_mining(self):
        """Start continuous mining with both adaptive and generator sessions."""
        logger.info(f"Starting continuous integrated mining with {self.config.mining_interval_hours}h intervals")
        
        # Schedule mining sessions
        schedule.every(self.config.mining_interval_hours).hours.do(self._run_mining_cycle)
        
        # Schedule daily submission at 2 PM
        schedule.every().day.at("14:00").do(self._run_submission_session)
        
        # Run initial mining cycle
        self._run_mining_cycle()
        
        # Main loop
        while self.running:
            try:
                # Run pending scheduled tasks
                schedule.run_pending()
                
                # Small delay before next cycle
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping...")
                break
            except Exception as e:
                logger.error(f"Error in continuous mining: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _run_mining_cycle(self):
        """Run a complete mining cycle with both adaptive and generator sessions."""
        logger.info("Starting mining cycle...")
        
        # Run adaptive mining in a separate thread
        adaptive_thread = threading.Thread(
            target=self.run_adaptive_mining_session,
            daemon=True
        )
        adaptive_thread.start()
        
        # Wait a bit for adaptive mining to start
        time.sleep(30)
        
        # Run alpha generator in a separate thread
        generator_thread = threading.Thread(
            target=self.run_alpha_generator_session,
            daemon=True
        )
        generator_thread.start()
        
        # Wait for both to complete
        adaptive_thread.join(timeout=1800)  # 30 minutes timeout
        generator_thread.join(timeout=1800)  # 30 minutes timeout
        
        logger.info("Mining cycle completed")
    
    def _run_submission_session(self):
        """Run a submission session."""
        logger.info("Starting submission session...")
        
        try:
            # Submit best adaptive alpha
            if self.adaptive_miner.best_alpha:
                success = self.adaptive_miner.submit_best_alpha()
                if success:
                    logger.info("Successfully submitted best adaptive alpha!")
                else:
                    logger.warning("Failed to submit adaptive alpha")
            
            # Note: Alpha generator alphas are typically submitted through the generator itself
            logger.info("Submission session completed")
            
        except Exception as e:
            logger.error(f"Error in submission session: {e}")
    
    def get_status(self) -> Dict:
        """Get current status of the integrated miner."""
        return {
            'adaptive_mining_active': self.adaptive_mining_active,
            'generator_active': self.generator_active,
            'total_adaptive_alphas': self.total_adaptive_alphas,
            'total_generator_alphas': self.total_generator_alphas,
            'best_adaptive_score': self.best_adaptive_score,
            'best_generator_score': self.best_generator_score,
            'best_adaptive_alpha': {
                'sharpe': self.adaptive_miner.best_alpha.sharpe if self.adaptive_miner.best_alpha else 0,
                'fitness': self.adaptive_miner.best_alpha.fitness if self.adaptive_miner.best_alpha else 0,
                'expression': self.adaptive_miner.best_alpha.expression[:100] + "..." if self.adaptive_miner.best_alpha else "None"
            },
            'bandit_arms': len(self.adaptive_miner.bandit.arms),
            'genetic_generation': self.adaptive_miner.genetic_algo.generation
        }
    
    def run_single_cycle(self):
        """Run a single mining cycle."""
        logger.info("Running single mining cycle...")
        self._run_mining_cycle()
    
    def run_adaptive_only(self):
        """Run only adaptive mining."""
        logger.info("Running adaptive mining only...")
        self.run_adaptive_mining_session()
    
    def run_generator_only(self):
        """Run only alpha generator."""
        logger.info("Running alpha generator only...")
        self.run_alpha_generator_session()
    
    def submit_best_alpha(self):
        """Submit the best alpha found."""
        logger.info("Submitting best alpha...")
        success = self.adaptive_miner.submit_best_alpha()
        if success:
            logger.info("Alpha submitted successfully!")
        else:
            logger.error("Failed to submit alpha")
        return success
    
    def reset_system(self):
        """Reset the integrated system."""
        logger.info("Resetting integrated system...")
        
        # Reset adaptive miner
        self.adaptive_miner.best_alpha = None
        self.adaptive_miner.best_score = 0
        self.adaptive_miner.results_history = []
        self.adaptive_miner.genetic_algo.population = []
        self.adaptive_miner.genetic_algo.generation = 0
        self.adaptive_miner.bandit.arms = {}
        
        # Reset integrated miner
        self.total_adaptive_alphas = 0
        self.total_generator_alphas = 0
        self.best_adaptive_score = 0
        self.best_generator_score = 0
        
        # Save reset state
        self.adaptive_miner.save_state()
        self.save_state()
        
        logger.info("Integrated system reset complete")
    
    def stop(self):
        """Stop the integrated miner."""
        logger.info("Stopping integrated miner...")
        self.running = False
        
        # Save final state
        self.adaptive_miner.save_state()
        self.save_state()

def main():
    parser = argparse.ArgumentParser(description='Integrated Alpha Miner - Combines Adaptive Mining with Alpha Generation')
    parser.add_argument('--credentials', type=str, default='./credential.txt',
                      help='Path to credentials file (default: ./credential.txt)')
    parser.add_argument('--mode', type=str, 
                      choices=['continuous', 'single', 'adaptive-only', 'generator-only', 'submit', 'status', 'reset'],
                      default='continuous', help='Operation mode (default: continuous)')
    parser.add_argument('--adaptive-batch-size', type=int, default=5,
                      help='Adaptive mining batch size (default: 5)')
    parser.add_argument('--adaptive-iterations', type=int, default=3,
                      help='Adaptive mining iterations (default: 3)')
    parser.add_argument('--lateral-count', type=int, default=3,
                      help='Lateral movement count (default: 3)')
    parser.add_argument('--generator-batch-size', type=int, default=10,
                      help='Alpha generator batch size (default: 10)')
    parser.add_argument('--generator-sleep-time', type=int, default=30,
                      help='Alpha generator sleep time (default: 30)')
    parser.add_argument('--mining-interval', type=int, default=6,
                      help='Mining interval in hours for continuous mode (default: 6)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                      help='Ollama API URL (default: http://localhost:11434)')
    parser.add_argument('--ollama-model', type=str, default='deepseek-r1:8b',
                      help='Ollama model to use (default: deepseek-r1:8b)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = MiningConfig(
        adaptive_batch_size=args.adaptive_batch_size,
        adaptive_iterations=args.adaptive_iterations,
        lateral_count=args.lateral_count,
        generator_batch_size=args.generator_batch_size,
        generator_sleep_time=args.generator_sleep_time,
        mining_interval_hours=args.mining_interval,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model
    )
    
    try:
        miner = IntegratedAlphaMiner(args.credentials, config)
        
        if args.mode == 'continuous':
            logger.info(f"Starting continuous integrated mining with {args.mining_interval}h intervals")
            miner.start_continuous_mining()
        
        elif args.mode == 'single':
            logger.info("Running single mining cycle")
            miner.run_single_cycle()
        
        elif args.mode == 'adaptive-only':
            logger.info("Running adaptive mining only")
            miner.run_adaptive_only()
        
        elif args.mode == 'generator-only':
            logger.info("Running alpha generator only")
            miner.run_generator_only()
        
        elif args.mode == 'submit':
            logger.info("Submitting best alpha")
            miner.submit_best_alpha()
        
        elif args.mode == 'status':
            status = miner.get_status()
            print(json.dumps(status, indent=2))
        
        elif args.mode == 'reset':
            logger.info("Resetting integrated system")
            miner.reset_system()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        miner.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    finally:
        miner.stop()
    
    return 0

if __name__ == "__main__":
    exit(main())
