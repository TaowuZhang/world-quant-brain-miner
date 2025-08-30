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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from dataclasses import dataclass

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

@dataclass
class ModelInfo:
    """Information about a model in the fleet."""
    name: str
    size_mb: int
    priority: int  # Lower number = higher priority (used first)
    description: str

class ModelFleetManager:
    """Manages a fleet of models with automatic downgrading on VRAM issues."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.current_model_index = 0
        self.vram_error_count = 0
        self.max_vram_errors = 3  # Number of VRAM errors before downgrading
        
        # Model fleet ordered by priority (largest to smallest)
        # Optimized for RTX A4000 (16GB VRAM) with DeepSeek-R1 reasoning models
        self.model_fleet = [
            ModelInfo("deepseek-r1:8b", 5200, 1, "DeepSeek-R1 8B - Reasoning model (RTX A4000 optimized)"),
            ModelInfo("deepseek-r1:7b", 4700, 2, "DeepSeek-R1 7B - Reasoning model"),
            ModelInfo("deepseek-r1:1.5b", 1100, 3, "DeepSeek-R1 1.5B - Reasoning model"),
            ModelInfo("llama3:3b", 2048, 4, "Llama 3.2 3B - Fallback model"),
            ModelInfo("phi3:mini", 2200, 5, "Phi3 mini - Emergency fallback"),
        ]
        
        # State file to persist current model selection
        self.state_file = "model_fleet_state.json"
        self.load_state()
        
    def load_state(self):
        """Load the current model state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.current_model_index = state.get('current_model_index', 0)
                    self.vram_error_count = state.get('vram_error_count', 0)
                    logger.info(f"Loaded state: model_index={self.current_model_index}, vram_errors={self.vram_error_count}")
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
            self.current_model_index = 0
            self.vram_error_count = 0
    
    def save_state(self):
        """Save the current model state to file."""
        try:
            state = {
                'current_model_index': self.current_model_index,
                'vram_error_count': self.vram_error_count,
                'current_model': self.get_current_model().name,
                'timestamp': time.time()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved state: {state}")
        except Exception as e:
            logger.error(f"Could not save state: {e}")
    
    def get_current_model(self) -> ModelInfo:
        """Get the current model in use."""
        if self.current_model_index >= len(self.model_fleet):
            self.current_model_index = len(self.model_fleet) - 1
        return self.model_fleet[self.current_model_index]
    
    def get_available_models(self) -> List[str]:
        """Get list of available models via Ollama API."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
            else:
                logger.warning(f"Failed to get available models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def downgrade_model(self):
        """Downgrade to the next smaller model."""
        if self.current_model_index < len(self.model_fleet) - 1:
            self.current_model_index += 1
            self.vram_error_count = 0
            self.save_state()
            current_model = self.get_current_model()
            logger.warning(f"🔄 Downgraded to model: {current_model.name} ({current_model.description})")
            return True
        else:
            logger.error("❌ Cannot downgrade further - already at smallest model")
            return False
    
    def reset_to_largest_model(self):
        """Reset to the largest model."""
        self.current_model_index = 0
        self.vram_error_count = 0
        self.save_state()
        current_model = self.get_current_model()
        logger.info(f"🔄 Reset to largest model: {current_model.name} ({current_model.description})")
        return True
            
    def trigger_application_reset(self):
        """Trigger a complete application reset."""
        logger.warning("🔄 Triggering application reset - returning to largest model")
        return self.reset_to_largest_model()
    
    def get_fleet_status(self) -> Dict:
        """Get the current status of the model fleet."""
        current_model = self.get_current_model()
        available_models = self.get_available_models()
        
        return {
            'current_model': {
                'name': current_model.name,
                'size_mb': current_model.size_mb,
                'priority': current_model.priority,
                'description': current_model.description
            },
            'vram_error_count': self.vram_error_count,
            'max_vram_errors': self.max_vram_errors,
            'available_models': available_models,
            'current_model_available': current_model.name in available_models,
            'fleet_size': len(self.model_fleet),
            'can_downgrade': self.current_model_index < len(self.model_fleet) - 1
        }
    
class AlphaOrchestrator:
    """Enhanced alpha orchestrator that manages the integrated alpha miner."""
    
    def __init__(self, credentials_path: str, ollama_url: str = "http://localhost:11434"):
        self.credentials_path = credentials_path
        self.ollama_url = ollama_url
        
        # Model fleet management
        self.model_fleet_manager = ModelFleetManager(ollama_url)
        
        # Process management
        self.integrated_miner_process = None
        self.vram_monitor_process = None
        self.restart_thread = None
        self.running = True
        
        # Configuration
        self.max_concurrent_simulations = 3
        self.restart_interval = 30 * 60  # 30 minutes
        self.last_restart_time = time.time()
        self.last_submission_date = ""
        
        # Load submission history
        self.load_submission_history()

    def load_submission_history(self):
        """Load submission history from file."""
        try:
            if os.path.exists('submission_history.json'):
                with open('submission_history.json', 'r') as f:
                    history = json.load(f)
                    self.last_submission_date = history.get('last_submission_date', '')
        except Exception as e:
            logger.warning(f"Could not load submission history: {e}")

    def save_submission_history(self):
        """Save submission history to file."""
        try:
            history = {
            'last_submission_date': self.last_submission_date,
                'timestamp': time.time()
            }
            with open('submission_history.json', 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save submission history: {e}")
    
    def run_integrated_mining(self, mode: str = 'single', **kwargs):
        """Run integrated mining system with specified parameters."""
        current_model = self.model_fleet_manager.get_current_model()
        logger.info(f"Running integrated mining in {mode} mode with model: {current_model.name}")
        
        try:
            # Build command with all available parameters
            cmd = [
                sys.executable, 'integrated_alpha_miner.py',
                '--mode', mode,
                '--credentials', self.credentials_path,
                '--ollama-url', self.ollama_url,
                '--ollama-model', current_model.name
            ]
            
            # Add optional parameters if provided
            if 'adaptive_batch_size' in kwargs:
                cmd.extend(['--adaptive-batch-size', str(kwargs['adaptive_batch_size'])])
            if 'adaptive_iterations' in kwargs:
                cmd.extend(['--adaptive-iterations', str(kwargs['adaptive_iterations'])])
            if 'lateral_count' in kwargs:
                cmd.extend(['--lateral-count', str(kwargs['lateral_count'])])
            if 'generator_batch_size' in kwargs:
                cmd.extend(['--generator-batch-size', str(kwargs['generator_batch_size'])])
            if 'generator_sleep_time' in kwargs:
                cmd.extend(['--generator-sleep-time', str(kwargs['generator_sleep_time'])])
            if 'mining_interval' in kwargs:
                cmd.extend(['--mining-interval', str(kwargs['mining_interval'])])
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
            
            if result.returncode == 0:
                logger.info("Integrated mining completed successfully")
                return True
            else:
                logger.error(f"Integrated mining failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Integrated mining timed out")
            return False
        except Exception as e:
            logger.error(f"Error running integrated mining: {e}")
            return False
    
    def start_integrated_mining_continuous(self, **kwargs):
        """Start integrated mining in continuous mode."""
        current_model = self.model_fleet_manager.get_current_model()
        logger.info(f"Starting continuous integrated mining with model: {current_model.name}")
        
        try:
            # Build command for continuous mode
            cmd = [
                sys.executable, 'integrated_alpha_miner.py',
                '--mode', 'continuous',
                '--credentials', self.credentials_path,
                '--ollama-url', self.ollama_url,
                '--ollama-model', current_model.name
            ]
            
            # Add optional parameters if provided
            if 'adaptive_batch_size' in kwargs:
                cmd.extend(['--adaptive-batch-size', str(kwargs['adaptive_batch_size'])])
            if 'adaptive_iterations' in kwargs:
                cmd.extend(['--adaptive-iterations', str(kwargs['adaptive_iterations'])])
            if 'lateral_count' in kwargs:
                cmd.extend(['--lateral-count', str(kwargs['lateral_count'])])
            if 'generator_batch_size' in kwargs:
                cmd.extend(['--generator-batch-size', str(kwargs['generator_batch_size'])])
            if 'generator_sleep_time' in kwargs:
                cmd.extend(['--generator-sleep-time', str(kwargs['generator_sleep_time'])])
            if 'mining_interval' in kwargs:
                cmd.extend(['--mining-interval', str(kwargs['mining_interval'])])
            
            logger.info(f"Starting command: {' '.join(cmd)}")
            
            # Start the integrated miner as a subprocess
            self.integrated_miner_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            logger.info(f"Integrated miner started with PID: {self.integrated_miner_process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting integrated miner: {e}")
            return False
    
    def run_alpha_submitter(self, batch_size: int = 3):
        """Run alpha submitter."""
        if not self.can_submit_today():
            return False
        
        logger.info("Running alpha submitter...")
        
        try:
            # Run the alpha submitter as a subprocess
            result = subprocess.run([
                sys.executable, 'alpha_submitter.py',
                '--batch-size', str(batch_size),
                '--credentials', self.credentials_path
            ], capture_output=True, text=True, timeout=1800)  # 30 minutes timeout
            
            if result.returncode == 0:
                logger.info("Alpha submitter completed successfully")
                # Update submission date
                self.last_submission_date = datetime.now().date().isoformat()
                self.save_submission_history()
                return True
            else:
                logger.error(f"Alpha submitter failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Alpha submitter timed out")
            return False
        except Exception as e:
            logger.error(f"Error running alpha submitter: {e}")
            return False
    
    def start_vram_monitoring(self):
        """Start VRAM monitoring in a separate process."""
        try:
            self.vram_monitor_process = subprocess.Popen([
                sys.executable, 'vram_monitor.py',
                '--ollama-url', self.ollama_url,
                '--threshold', '0.9',  # 90% VRAM usage threshold
                '--check-interval', '30'  # Check every 30 seconds
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            logger.info(f"VRAM monitoring started with PID: {self.vram_monitor_process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting VRAM monitoring: {e}")
            return False

    def stop_processes(self):
        """Stop all running processes."""
        logger.info("Stopping all processes...")
        
        if self.integrated_miner_process and self.integrated_miner_process.poll() is None:
            self.integrated_miner_process.terminate()
            self.integrated_miner_process.wait(timeout=30)
            logger.info("Integrated miner stopped")
        
        if self.vram_monitor_process and self.vram_monitor_process.poll() is None:
            self.vram_monitor_process.terminate()
            self.vram_monitor_process.wait(timeout=30)
            logger.info("VRAM monitoring stopped")
        
        self.running = False
    
    def restart_all_processes(self):
        """Restart all processes."""
        logger.info("Restarting all processes...")
        
        try:
            # Stop current processes
            self.stop_processes()
            
            # Wait a moment
            time.sleep(5)
            
            # Start processes again with default parameters
            self.start_integrated_mining_continuous(
                adaptive_batch_size=5,
                adaptive_iterations=3,
                lateral_count=3,
                generator_batch_size=10,
                generator_sleep_time=30,
                mining_interval=6
            )
            self.start_vram_monitoring()
            
            # Update restart time
            self.last_restart_time = time.time()
            
            logger.info("All processes restarted successfully")
            
        except Exception as e:
            logger.error(f"Error restarting processes: {e}")
    
    def start_restart_monitoring(self):
        """Start restart monitoring in a separate thread."""
        if not self.restart_thread or not self.restart_thread.is_alive():
            self.restart_thread = threading.Thread(target=self._restart_monitor_loop, daemon=True)
            self.restart_thread.start()
            logger.info("🔄 Restart monitoring started (30-minute intervals)")
    
    def _restart_monitor_loop(self):
        """Monitor and restart processes every 30 minutes."""
        while self.running:
            try:
                current_time = time.time()
                time_since_last_restart = current_time - self.last_restart_time
                
                if time_since_last_restart >= self.restart_interval:
                    logger.info(f"⏰ 30 minutes elapsed since last restart, initiating restart...")
                    self.restart_all_processes()
                else:
                    remaining_time = self.restart_interval - time_since_last_restart
                    logger.debug(f"⏰ Next restart in {remaining_time/60:.1f} minutes")
                
                # Check every minute
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in restart monitoring: {e}")
                time.sleep(60)

    def get_model_fleet_status(self) -> Dict:
        """Get the current status of the model fleet."""
        return self.model_fleet_manager.get_fleet_status()

    def reset_model_fleet(self):
        """Reset the model fleet to the largest model."""
        return self.model_fleet_manager.reset_to_largest_model()

    def force_model_downgrade(self):
        """Force downgrade to the next smaller model."""
        return self.model_fleet_manager.downgrade_model()
    
    def force_application_reset(self):
        """Force a complete application reset."""
        logger.warning("Forcing application reset")
        return self.model_fleet_manager.trigger_application_reset()

    def can_submit_today(self) -> bool:
        """Check if we can submit alphas today (only once per day)."""
        today = datetime.now().date().isoformat()
        
        if self.last_submission_date == today:
            logger.info(f"Already submitted today ({today}). Skipping submission.")
            return False
        
        logger.info(f"Can submit today. Last submission was: {self.last_submission_date}")
        return True

    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        return {
            'model_fleet': self.get_model_fleet_status(),
            'processes': {
                'integrated_miner_running': self.integrated_miner_process and self.integrated_miner_process.poll() is None,
                'vram_monitor_running': self.vram_monitor_process and self.vram_monitor_process.poll() is None,
                'restart_monitor_running': self.restart_thread and self.restart_thread.is_alive()
            },
            'configuration': {
                'max_concurrent_simulations': self.max_concurrent_simulations,
                'restart_interval_minutes': self.restart_interval // 60,
                'last_submission_date': self.last_submission_date
            },
            'available_systems': {
                'integrated_mining': os.path.exists('integrated_alpha_miner.py'),
                'alpha_submitter': os.path.exists('alpha_submitter.py'),
                'vram_monitor': os.path.exists('vram_monitor.py')
            }
        }

    def daily_workflow(self):
        """Run daily workflow with integrated mining."""
        logger.info("Starting daily workflow...")
        
        # 1. Run integrated mining in single mode
        logger.info("Phase 1: Running integrated mining...")
        self.run_integrated_mining(
            mode='single',
            adaptive_batch_size=5,
            adaptive_iterations=3,
            lateral_count=3,
            generator_batch_size=10,
            generator_sleep_time=30
        )
        
        # 2. Run alpha submitter (once per day)
        logger.info("Phase 2: Running alpha submitter...")
        self.run_alpha_submitter(batch_size=3)
        
        logger.info("Daily workflow completed")

    def continuous_mining(self, mining_interval_hours: int = 6, **kwargs):
        """Run continuous mining with integrated alpha miner."""
        logger.info(f"Starting continuous integrated mining with {mining_interval_hours}h intervals...")
        
        try:
            # Start VRAM monitoring
            logger.info("Starting VRAM monitoring...")
            self.start_vram_monitoring()
            
            # Start restart monitoring
            logger.info("Starting restart monitoring...")
            self.start_restart_monitoring()
            
            # Start integrated miner in continuous mode
            self.start_integrated_mining_continuous(
                adaptive_batch_size=kwargs.get('adaptive_batch_size', 5),
                adaptive_iterations=kwargs.get('adaptive_iterations', 3),
                lateral_count=kwargs.get('lateral_count', 3),
                generator_batch_size=kwargs.get('generator_batch_size', 10),
                generator_sleep_time=kwargs.get('generator_sleep_time', 30),
                mining_interval=kwargs.get('mining_interval', mining_interval_hours)
            )
            
            # Schedule daily submission at 2 PM
            schedule.every().day.at("14:00").do(self.run_alpha_submitter)
            
            logger.info("Integrated mining system is running")
            logger.info(f"Max concurrent simulations: {self.max_concurrent_simulations}")
            
            while self.running:
                try:
                    # Run pending scheduled tasks
                    schedule.run_pending()
                    
                    # Check if integrated miner process is still running
                    if self.integrated_miner_process and self.integrated_miner_process.poll() is not None:
                        logger.warning("Integrated miner process stopped, restarting...")
                        self.start_integrated_mining_continuous(
                            adaptive_batch_size=kwargs.get('adaptive_batch_size', 5),
                            adaptive_iterations=kwargs.get('adaptive_iterations', 3),
                            lateral_count=kwargs.get('lateral_count', 3),
                            generator_batch_size=kwargs.get('generator_batch_size', 10),
                            generator_sleep_time=kwargs.get('generator_sleep_time', 30),
                            mining_interval=kwargs.get('mining_interval', mining_interval_hours)
                        )
                    
                    # Small delay before next cycle
                    time.sleep(60)
                    
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal, stopping...")
                    break
                except Exception as e:
                    logger.error(f"Error in continuous mining: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
                    
        finally:
            self.stop_processes()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Alpha Orchestrator - Manage integrated alpha mining system')
    parser.add_argument('--credentials', type=str, default='./credential.txt',
                      help='Path to credentials file (default: ./credential.txt)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                      help='Ollama API URL (default: http://localhost:11434)')
    parser.add_argument('--mode', type=str, 
                      choices=['daily', 'continuous', 'single', 'submitter', 'fleet-status', 'fleet-reset', 'fleet-downgrade', 'fleet-reset-app', 'restart', 'status'],
                      default='continuous', help='Operation mode (default: continuous)')
    parser.add_argument('--mining-interval', type=int, default=6,
                      help='Mining interval in hours for continuous mode (default: 6)')
    parser.add_argument('--batch-size', type=int, default=3,
                      help='Batch size for operations (default: 3)')
    parser.add_argument('--max-concurrent', type=int, default=3,
                      help='Maximum concurrent simulations (default: 3)')
    parser.add_argument('--restart-interval', type=int, default=30,
                      help='Restart interval in minutes (default: 30)')
    parser.add_argument('--ollama-model', type=str, default='deepseek-r1:8b',
                      help='Ollama model to use (default: deepseek-r1:8b)')
    
    # Integrated miner specific parameters
    parser.add_argument('--adaptive-batch-size', type=int, default=5,
                      help='Adaptive mining batch size (default: 5)')
    parser.add_argument('--adaptive-iterations', type=int, default=3,
                      help='Adaptive mining iterations (default: 3)')
    parser.add_argument('--lateral-count', type=int, default=3,
                      help='Lateral movement count (default: 3)')
    parser.add_argument('--generator-batch-size', type=int, default=10,
                      help='Generator batch size (default: 10)')
    parser.add_argument('--generator-sleep-time', type=int, default=30,
                      help='Generator sleep time in seconds (default: 30)')
    
    args = parser.parse_args()
    
    try:
        orchestrator = AlphaOrchestrator(args.credentials, args.ollama_url)
        orchestrator.max_concurrent_simulations = args.max_concurrent
        orchestrator.restart_interval = args.restart_interval * 60  # Convert minutes to seconds
        
        # Update the model fleet to use the specified model
        if args.ollama_model:
            # Find the model in the fleet and set it as current
            for i, model_info in enumerate(orchestrator.model_fleet_manager.model_fleet):
                if model_info.name == args.ollama_model:
                    orchestrator.model_fleet_manager.current_model_index = i
                    orchestrator.model_fleet_manager.save_state()
                    logger.info(f"Set model fleet to use: {args.ollama_model}")
                    break
        
        # Prepare integrated miner parameters
        integrated_params = {
            'adaptive_batch_size': args.adaptive_batch_size,
            'adaptive_iterations': args.adaptive_iterations,
            'lateral_count': args.lateral_count,
            'generator_batch_size': args.generator_batch_size,
            'generator_sleep_time': args.generator_sleep_time,
            'mining_interval': args.mining_interval
        }
        
        if args.mode == 'daily':
            orchestrator.daily_workflow()
        elif args.mode == 'continuous':
            orchestrator.continuous_mining(args.mining_interval, **integrated_params)
        elif args.mode == 'single':
            orchestrator.run_integrated_mining('single', **integrated_params)
        elif args.mode == 'submitter':
            orchestrator.run_alpha_submitter(args.batch_size)
        elif args.mode == 'fleet-status':
            status = orchestrator.get_model_fleet_status()
            print(json.dumps(status, indent=2))
        elif args.mode == 'fleet-reset':
            orchestrator.reset_model_fleet()
            print("Model fleet reset to largest model")
        elif args.mode == 'fleet-downgrade':
            orchestrator.force_model_downgrade()
            print("Model fleet downgraded to next smaller model")
        elif args.mode == 'fleet-reset-app':
            orchestrator.force_application_reset()
            print("Application reset completed - returned to largest model")
        elif args.mode == 'restart':
            orchestrator.restart_all_processes()
            print("Manual restart completed")
        elif args.mode == 'status':
            status = orchestrator.get_system_status()
            print(json.dumps(status, indent=2))
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
