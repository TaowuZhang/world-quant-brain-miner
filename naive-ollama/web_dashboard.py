from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import os
import time
import subprocess
import threading
from datetime import datetime, timedelta
import requests
import logging
from typing import Dict, List, Optional
import psutil

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaDashboard:
    def __init__(self):
        self.status_file = "dashboard_status.json"
        self.log_file = "alpha_orchestrator.log"
        self.submission_log_file = "submission_log.json"
        self.results_dir = "results"
        self.logs_dir = "logs"
        
        # Alpha process tracking
        self.alpha_processes = {}
        self.process_lock = threading.Lock()
        
        # Model configuration
        self.model_config = {
            "ollama": {
                "url": "http://localhost:11434",
                "models": ["llama3.2:3b", "qwen2.5:7b", "deepseek-coder:6.7b"]
            },
            "openrouter": {
                "url": "https://openrouter.ai/api/v1",
                "models": [
                    "moonshotai/kimi-dev-72b:free", 
                    "google/gemma-3-27b-it:free",
                    "openai/gpt-oss-20b:free",
                    "meta-llama/llama-3.3-8b-instruct:free",
                    "google/gemma-3n-e4b-it:free",
                    "qwen/qwen3-4b:free",
                    "mistralai/mistral-7b-instruct:free"
                ]
            }
        }
        self.current_model_provider = "openrouter"  # Prioritize OpenRouter
        self.current_model = "moonshotai/kimi-dev-72b:free"
        
        # Model performance tracking
        self.model_performance = {}
        
    def get_system_status(self) -> Dict:
        """Get overall system status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "gpu": self.get_gpu_status(),
            "ollama": self.get_ollama_status(),
            "orchestrator": self.get_orchestrator_status(),
            "worldquant": self.get_worldquant_status(),
            "recent_activity": self.get_recent_activity(),
            "statistics": self.get_statistics(),
            "alpha_processes": self.get_alpha_processes_status(),
            "model_config": {
                "current_provider": self.current_model_provider,
                "current_model": self.current_model,
                "available_providers": list(self.model_config.keys())
            }
        }
        return status
    
    def get_gpu_status(self) -> Dict:
        """Get GPU status and utilization."""
        try:
            # Try to get GPU info from nvidia-smi (for NVIDIA GPUs)
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    parts = lines[0].split(', ')
                    if len(parts) >= 5:
                        return {
                            "status": "active",
                            "name": parts[0],
                            "memory_used_mb": int(parts[1]),
                            "memory_total_mb": int(parts[2]),
                            "utilization_percent": int(parts[3]),
                            "temperature_c": int(parts[4]),
                            "memory_percent": round((int(parts[1]) / int(parts[2])) * 100, 1)
                        }
        except Exception as e:
            logger.debug(f"NVIDIA GPU not available: {e}")
        
        # For macOS, try to detect Apple Silicon GPU or integrated graphics
        try:
            # Check system_profiler for GPU info on macOS
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout:
                gpu_info = result.stdout
                if 'Apple' in gpu_info or 'Metal' in gpu_info:
                    # Extract GPU name from system_profiler output
                    lines = gpu_info.split('\n')
                    gpu_name = "Apple Silicon GPU"
                    for line in lines:
                        if 'Chipset Model:' in line:
                            gpu_name = line.split(':')[1].strip()
                            break
                    return {
                        "status": "active",
                        "name": gpu_name,
                        "type": "integrated",
                        "platform": "macOS"
                    }
        except Exception as e:
            logger.debug(f"Could not get macOS GPU info: {e}")
        
        # Fallback: assume CPU-only mode
        return {
            "status": "cpu_only", 
            "name": "CPU Mode",
            "message": "Running in CPU mode (no dedicated GPU detected)"
        }
    
    def get_ollama_status(self) -> Dict:
        """Get Ollama service status."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return {
                    "status": "running",
                    "models": [model.get("name", "") for model in models],
                    "model_count": len(models)
                }
        except Exception as e:
            logger.warning(f"Could not get Ollama status: {e}")
        
        return {"status": "not_responding", "error": "Ollama service not available"}
    
    def get_orchestrator_status(self) -> Dict:
        """Get orchestrator status from local processes and logs."""
        status = {
            "status": "unknown",
            "last_activity": None,
            "current_mode": "continuous",
            "next_mining": None,
            "next_submission": None
        }
        
        try:
            # Check if alpha_orchestrator process is running
            result = subprocess.run([
                "ps", "aux"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                processes = result.stdout
                if "alpha_orchestrator.py" in processes:
                    status["status"] = "active"
                    # Extract process info
                    for line in processes.split('\n'):
                        if "alpha_orchestrator.py" in line and "grep" not in line:
                            parts = line.split()
                            if len(parts) > 10:
                                # Extract mode from command line
                                cmd_line = ' '.join(parts[10:])
                                if "--mode generator" in cmd_line:
                                    status["current_mode"] = "generator"
                                elif "--mode miner" in cmd_line:
                                    status["current_mode"] = "miner"
                                elif "--mode submitter" in cmd_line:
                                    status["current_mode"] = "submitter"
                                status["last_activity"] = f"Running: {cmd_line}"
                            break
                else:
                    status["status"] = "stopped"
                    status["last_activity"] = "No orchestrator process found"
            
            # Check local log files for recent activity
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if not status["last_activity"] or status["status"] == "stopped":
                            status["last_activity"] = last_line
                        
                        # Check for recent activity in logs
                        for line in reversed(lines[-20:]):
                            line_lower = line.lower()
                            if any(keyword in line_lower for keyword in ["alpha generator", "generating alpha", "running alpha", "alpha idea"]):
                                if status["status"] == "unknown":
                                    status["status"] = "active"
                                break
                            elif any(keyword in line_lower for keyword in ["error", "failed", "exception"]):
                                if status["status"] == "unknown":
                                    status["status"] = "error"
                                break
                
                # Check submission schedule
                if os.path.exists(self.submission_log_file):
                    with open(self.submission_log_file, 'r') as f:
                        data = json.load(f)
                        last_submission = data.get("last_submission_date")
                        if last_submission:
                            last_date = datetime.fromisoformat(last_submission)
                            next_submission = last_date + timedelta(days=1)
                            next_submission = next_submission.replace(hour=14, minute=0, second=0, microsecond=0)
                            status["next_submission"] = next_submission.isoformat()
                
                # Calculate next mining time (every 6 hours)
                now = datetime.now()
                hours_since_midnight = now.hour + now.minute / 60
                next_mining_hour = ((int(hours_since_midnight // 6) + 1) * 6) % 24
                next_mining = now.replace(hour=int(next_mining_hour), minute=0, second=0, microsecond=0)
                if next_mining <= now:
                    next_mining += timedelta(days=1)
                status["next_mining"] = next_mining.isoformat()
                
        except Exception as e:
            logger.warning(f"Could not get orchestrator status: {e}")
        
        return status
    
    def get_worldquant_status(self) -> Dict:
        """Check WorldQuant Brain API status."""
        try:
            if os.path.exists("credential.txt"):
                with open("credential.txt", 'r') as f:
                    credentials = json.load(f)
                
                session = requests.Session()
                session.auth = (credentials[0], credentials[1])
                response = session.post('https://api.worldquantbrain.com/authentication', timeout=10)
                
                if response.status_code == 201:
                    return {"status": "connected", "message": "Authentication successful"}
                else:
                    return {"status": "auth_failed", "message": f"Status: {response.status_code}"}
        except Exception as e:
            logger.warning(f"Could not check WorldQuant status: {e}")
        
        return {"status": "unknown", "message": "Could not verify connection"}
    
    def get_recent_activity(self) -> List[Dict]:
        """Get recent activity from local log files."""
        activities = []
        
        try:
            # Read from local log files
            log_files = [self.log_file, "alpha_generator_ollama.log", "monitor.log"]
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        # Get last 10 lines from each log file
                        for line in lines[-10:]:
                            line = line.strip()
                            if line and not line.startswith('---'):
                                # Parse timestamp and message
                                try:
                                    # Try to parse standard log format: YYYY-MM-DD HH:MM:SS - MESSAGE
                                    if ' - ' in line:
                                        timestamp_str, message = line.split(' - ', 1)
                                        try:
                                            # Try different timestamp formats
                                            if 'T' in timestamp_str:
                                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                            else:
                                                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                                        except:
                                            timestamp = datetime.now()
                                        
                                        activities.append({
                                            "timestamp": timestamp.isoformat(),
                                            "message": message,
                                            "source": log_file,
                                            "type": "info" if "INFO" in message else "error" if "ERROR" in message else "warning" if "WARNING" in message else "debug"
                                        })
                                    else:
                                        # If no timestamp separator, use current time
                                        activities.append({
                                            "timestamp": datetime.now().isoformat(),
                                            "message": line,
                                            "source": log_file,
                                            "type": "unknown"
                                        })
                                except Exception as e:
                                    # Fallback for unparseable lines
                                    activities.append({
                                        "timestamp": datetime.now().isoformat(),
                                        "message": line,
                                        "source": log_file,
                                        "type": "unknown"
                                    })
                                
        except Exception as e:
            # Fallback to basic log reading
            try:
                if os.path.exists("alpha_orchestrator.log"):
                    with open("alpha_orchestrator.log", 'r') as f:
                        lines = f.readlines()
                        for line in lines[-5:]:
                            line = line.strip()
                            if line:
                                activities.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "message": line,
                                    "source": "alpha_orchestrator.log",
                                    "type": "info"
                                })
            except:
                activities.append({
                    "timestamp": datetime.now().isoformat(),
                    "message": "Unable to read logs",
                    "source": "system",
                    "type": "error"
                })
        
        # Sort by timestamp (newest first)
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        return activities[:30]  # Return last 30 activities
    
    def parse_log_line(self, line):
        """Parse a log line to extract timestamp, level, and message"""
        import re
        
        # Try to match common log formats
        patterns = [
            # Format: 2024-01-01 12:00:00 - INFO - Message
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*-\s*(\w+)\s*-\s*(.*)',
            # Format: [2024-01-01 12:00:00] INFO: Message
            r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*(\w+):\s*(.*)',
            # Format: INFO:2024-01-01 12:00:00:Message
            r'(\w+):(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):(.*)',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                if len(match.groups()) == 3:
                    timestamp_str, level, message = match.groups()
                    try:
                        # Try to parse timestamp
                        from datetime import datetime
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').isoformat()
                        return {
                            'timestamp': timestamp,
                            'level': level.upper(),
                            'message': message.strip()
                        }
                    except ValueError:
                        pass
        
        # If no pattern matches, return as INFO level with current timestamp
        from datetime import datetime
        return {
            'timestamp': datetime.now().isoformat(),
            'level': 'INFO',
            'message': line.strip()
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about generated alphas and results."""
        stats = {
            "total_alphas_generated": 0,
            "successful_alphas": 0,  # Alphas with expected content/promising results
            "completed_alphas": 0,   # Alphas that finished backtesting
            "failed_alphas": 0,
            "last_24h_generated": 0,
            "last_24h_successful": 0,
            "model_performance": self.get_model_performance_stats()
        }
        
        try:
            # Analyze alpha generator logs for statistics
            alpha_log_file = "alpha_generator_ollama.log"
            if os.path.exists(alpha_log_file):
                with open(alpha_log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Count successful (with expected content/promising results)
                    if any(keyword in line.lower() for keyword in [
                        "alpha generated successfully",
                        "promising alpha found",
                        "alpha meets criteria",
                        "expected performance",
                        "positive sharpe",
                        "good correlation",
                        "alpha validation passed",
                        "alpha idea generated",
                        "alpha expression created"
                    ]):
                        stats["successful_alphas"] += 1
                        
                    # Count completed (finished backtesting)
                    if any(keyword in line.lower() for keyword in [
                        "backtesting completed",
                        "backtest finished",
                        "alpha testing complete",
                        "simulation completed",
                        "performance analysis done",
                        "alpha submitted",
                        "submission successful"
                    ]):
                        stats["completed_alphas"] += 1
                        
                    # Count failed
                    if any(keyword in line.lower() for keyword in [
                        "alpha generation failed",
                        "generation error",
                        "failed to generate",
                        "alpha rejected",
                        "validation failed",
                        "error in alpha"
                    ]):
                        stats["failed_alphas"] += 1
        except Exception as e:
            logger.warning(f"Could not analyze alpha logs for statistics: {e}")
        
        # Calculate total alphas generated
        stats["total_alphas_generated"] = stats["successful_alphas"] + stats["failed_alphas"]
        
        return stats
        
    def get_alpha_processes_status(self) -> Dict:
        """Get status of all Alpha processes."""
        with self.process_lock:
            status = {}
            for process_id, process_info in self.alpha_processes.items():
                try:
                    process = process_info['process']
                    if process.poll() is None:  # Process is still running
                        status[process_id] = {
                            "status": "running",
                            "start_time": process_info['start_time'],
                            "type": process_info['type'],
                            "pid": process.pid,
                            "duration": (datetime.now() - datetime.fromisoformat(process_info['start_time'])).total_seconds()
                        }
                    else:
                        # Process has finished
                        status[process_id] = {
                            "status": "completed" if process.returncode == 0 else "failed",
                            "start_time": process_info['start_time'],
                            "type": process_info['type'],
                            "return_code": process.returncode,
                            "duration": (datetime.now() - datetime.fromisoformat(process_info['start_time'])).total_seconds()
                        }
                except Exception as e:
                    status[process_id] = {
                        "status": "error",
                        "error": str(e),
                        "type": process_info.get('type', 'unknown')
                    }
            return status
    
    def add_alpha_process(self, process_type: str, process: subprocess.Popen) -> str:
        """Add a new Alpha process to tracking."""
        process_id = f"{process_type}_{int(time.time())}"
        with self.process_lock:
            self.alpha_processes[process_id] = {
                "process": process,
                "type": process_type,
                "start_time": datetime.now().isoformat(),
                "pid": process.pid
            }
        return process_id
    
    def cleanup_finished_processes(self):
        """Clean up finished processes from tracking."""
        with self.process_lock:
            finished_processes = []
            for process_id, process_info in self.alpha_processes.items():
                try:
                    if process_info['process'].poll() is not None:
                        # Process has finished, mark for cleanup after 5 minutes
                        start_time = datetime.fromisoformat(process_info['start_time'])
                        if datetime.now() - start_time > timedelta(minutes=5):
                            finished_processes.append(process_id)
                except:
                    finished_processes.append(process_id)
            
            for process_id in finished_processes:
                del self.alpha_processes[process_id]
    
    def get_model_performance_stats(self) -> Dict:
        """Get model performance statistics including generation speed and backtest pass rate."""
        performance_stats = {}
        
        for provider in self.model_config:
            for model in self.model_config[provider]["models"]:
                model_key = f"{provider}:{model}"
                
                # Initialize default stats
                performance_stats[model_key] = {
                    "total_generations": 0,
                    "successful_generations": 0,
                    "failed_generations": 0,
                    "avg_generation_time": 0.0,
                    "backtest_pass_rate": 0.0,
                    "last_used": None,
                    "success_rate": 0.0
                }
                
                # Get actual performance data from tracking
                if model_key in self.model_performance:
                    perf_data = self.model_performance[model_key]
                    performance_stats[model_key].update({
                        "total_generations": perf_data.get("total_generations", 0),
                        "successful_generations": perf_data.get("successful_generations", 0),
                        "failed_generations": perf_data.get("failed_generations", 0),
                        "avg_generation_time": perf_data.get("avg_generation_time", 0.0),
                        "backtest_pass_rate": perf_data.get("backtest_pass_rate", 0.0),
                        "last_used": perf_data.get("last_used"),
                        "success_rate": perf_data.get("success_rate", 0.0)
                    })
        
        return performance_stats
    
    def update_model_performance(self, model_key: str, generation_time: float, success: bool, backtest_passed: bool = False):
        """Update model performance tracking."""
        if model_key not in self.model_performance:
            self.model_performance[model_key] = {
                "total_generations": 0,
                "successful_generations": 0,
                "failed_generations": 0,
                "total_generation_time": 0.0,
                "avg_generation_time": 0.0,
                "backtest_passes": 0,
                "backtest_attempts": 0,
                "backtest_pass_rate": 0.0,
                "last_used": None,
                "success_rate": 0.0
            }
        
        perf = self.model_performance[model_key]
        perf["total_generations"] += 1
        perf["total_generation_time"] += generation_time
        perf["avg_generation_time"] = perf["total_generation_time"] / perf["total_generations"]
        perf["last_used"] = datetime.now().isoformat()
        
        if success:
            perf["successful_generations"] += 1
        else:
            perf["failed_generations"] += 1
            
        perf["success_rate"] = perf["successful_generations"] / perf["total_generations"] * 100
        
        # Track backtest results
        if backtest_passed is not None:
            perf["backtest_attempts"] += 1
            if backtest_passed:
                perf["backtest_passes"] += 1
            perf["backtest_pass_rate"] = perf["backtest_passes"] / perf["backtest_attempts"] * 100
    
    def get_logs(self, log_type: str = "all", lines: int = 100, level_filter: str = None) -> List[Dict]:
        """Get logs from local files with structured data and level filtering."""
        logs = []
        
        try:
            # Get logs from local files
            log_files = ["alpha_orchestrator.log", "alpha_generator_ollama.log", "monitor.log"]
            
            if log_type == "alpha":
                log_files = ["alpha_generator_ollama.log"]
            elif log_type == "orchestrator":
                log_files = ["alpha_orchestrator.log"]
            elif log_type == "monitor":
                log_files = ["monitor.log"]
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    try:
                        with open(log_file, 'r') as f:
                            file_logs = f.readlines()
                            # Parse each log line
                            for line in file_logs[-(lines//max(len(log_files), 1)):]:
                                if line.strip():
                                    parsed_log = self.parse_log_line(line)
                                    parsed_log['source'] = log_file
                                    
                                    # Apply level filter if specified
                                    if level_filter is None or parsed_log['level'] == level_filter.upper():
                                        logs.append(parsed_log)
                    except Exception as e:
                        logs.append({
                            'timestamp': datetime.now().isoformat(),
                            'level': 'ERROR',
                            'message': f"Error reading file: {str(e)}",
                            'source': log_file
                        })
                else:
                    logs.append({
                        'timestamp': datetime.now().isoformat(),
                        'level': 'WARNING',
                        'message': "File not found",
                        'source': log_file
                    })
                            
        except Exception as e:
            logs = [{
                'timestamp': datetime.now().isoformat(),
                'level': 'ERROR',
                'message': f"Error reading logs: {str(e)}",
                'source': 'system'
            }]
            
        return logs[-lines:]
    
    def get_alpha_generator_logs(self, lines: int = 50, level_filter: str = None) -> List[Dict]:
        """Get Alpha Generator specific logs from local file with structured data."""
        logs = []
        try:
            # Read from local Alpha Generator log file
            alpha_log_file = "alpha_generator_ollama.log"
            if os.path.exists(alpha_log_file):
                with open(alpha_log_file, 'r') as f:
                    file_logs = f.readlines()
                    for line in file_logs[-lines:]:
                        if line.strip():
                            parsed_log = self.parse_log_line(line)
                            parsed_log['source'] = 'alpha_generator'
                            
                            # Apply level filter if specified
                            if level_filter is None or parsed_log['level'] == level_filter.upper():
                                logs.append(parsed_log)
            else:
                logs = [{
                    'timestamp': datetime.now().isoformat(),
                    'level': 'WARNING',
                    'message': "Alpha Generator log file not found",
                    'source': 'alpha_generator'
                }]
                        
        except Exception as e:
            logs = [{
                'timestamp': datetime.now().isoformat(),
                'level': 'ERROR',
                'message': f"Error reading Alpha Generator logs: {str(e)}",
                'source': 'alpha_generator'
            }]
                
        return logs
    
    def trigger_mining(self) -> Dict:
        """Trigger mining with process tracking."""
        try:
            # Clean up old processes first
            self.cleanup_finished_processes()
            
            # Start the mining process
            process = subprocess.Popen([
                "python3", "alpha_orchestrator.py", 
                "--mode", "miner",
                "--credentials", "./credential.txt"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Add to process tracking
            process_id = self.add_alpha_process("mining", process)
            
            return {
                "success": True,
                "message": "Mining started successfully",
                "process_id": process_id,
                "pid": process.pid
            }
        except Exception as e:
            logger.error(f"Error triggering mining: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def trigger_submission(self) -> Dict:
        """Trigger submission with process tracking."""
        try:
            # Clean up old processes first
            self.cleanup_finished_processes()
            
            # Start the submission process
            process = subprocess.Popen([
                "python3", "alpha_orchestrator.py", 
                "--mode", "submitter",
                "--credentials", "./credential.txt",
                "--batch-size", "3"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Add to process tracking
            process_id = self.add_alpha_process("submission", process)
            
            return {
                "success": True,
                "message": "Submission started successfully",
                "process_id": process_id,
                "pid": process.pid
            }
        except Exception as e:
            logger.error(f"Error triggering submission: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def trigger_alpha_generation(self) -> Dict:
        """Trigger alpha generation with process tracking."""
        try:
            # Clean up old processes first
            self.cleanup_finished_processes()
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # Start the alpha generation process with current model
            cmd = [
                "python3", "alpha_orchestrator.py", 
                "--mode", "generator",
                "--credentials", "./credential.txt",
                "--batch-size", "1"
            ]
            
            # Add model configuration if using OpenRouter
            if self.current_model_provider == "openrouter":
                cmd.extend(["--model-provider", "openrouter"])
                cmd.extend(["--model", self.current_model])
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Add to process tracking
            process_id = self.add_alpha_process("alpha_generation", process)
            
            # Track model usage
            model_key = f"{self.current_model_provider}:{self.current_model}"
            generation_time = time.time() - start_time
            
            # We'll update success/failure status later when process completes
            # For now, just record the attempt
            logger.info(f"Alpha generation started with model: {model_key}")
            
            return {
                "success": True,
                "message": f"Alpha generation started with {self.current_model}",
                "process_id": process_id,
                "pid": process.pid,
                "model": self.current_model,
                "provider": self.current_model_provider
            }
        except Exception as e:
            logger.error(f"Error triggering alpha generation: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Global dashboard instance
dashboard = AlphaDashboard()

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    return jsonify(dashboard.get_system_status())

@app.route('/api/logs')
def get_logs():
    """API endpoint for logs with level filtering."""
    lines = request.args.get('lines', 100, type=int)
    level_filter = request.args.get('level')
    return jsonify(dashboard.get_logs('all', lines, level_filter))

@app.route('/api/alpha_logs')
def get_alpha_logs():
    """API endpoint for alpha generator specific logs with level filtering."""
    lines = request.args.get('lines', 100, type=int)
    level_filter = request.args.get('level')
    return jsonify(dashboard.get_alpha_generator_logs(lines, level_filter))

@app.route('/api/trigger_mining', methods=['POST'])
def api_trigger_mining():
    """API endpoint to trigger manual mining."""
    result = dashboard.trigger_mining()
    return jsonify(result)

@app.route('/api/trigger_submission', methods=['POST'])
def api_trigger_submission():
    """API endpoint to trigger manual submission."""
    result = dashboard.trigger_submission()
    return jsonify(result)

@app.route('/api/trigger_alpha_generation', methods=['POST'])
def api_trigger_alpha_generation():
    """API endpoint to trigger manual alpha generation."""
    result = dashboard.trigger_alpha_generation()
    return jsonify(result)

# Add new API endpoints for Alpha process tracking and model management
@app.route('/api/alpha_processes')
def api_alpha_processes():
    """Get Alpha processes status."""
    return jsonify(dashboard.get_alpha_processes_status())

@app.route('/api/models')
def api_models():
    """Get available models configuration."""
    return jsonify({
        "current_provider": dashboard.current_model_provider,
        "current_model": dashboard.current_model,
        "providers": dashboard.model_config
    })

@app.route('/api/models/switch', methods=['POST'])
def api_switch_model():
    """Switch model provider and model."""
    data = request.get_json()
    provider = data.get('provider')
    model = data.get('model')
    
    if provider not in dashboard.model_config:
        return jsonify({"success": False, "error": "Invalid provider"})
    
    if model not in dashboard.model_config[provider]['models']:
        return jsonify({"success": False, "error": "Invalid model for provider"})
    
    dashboard.current_model_provider = provider
    dashboard.current_model = model
    
    return jsonify({
        "success": True,
        "message": f"Switched to {provider}:{model}"
    })

@app.route('/api/models/add', methods=['POST'])
def api_add_model():
    """Add a new model to a provider."""
    data = request.get_json()
    provider = data.get('provider')
    model = data.get('model')
    
    if provider not in dashboard.model_config:
        return jsonify({"success": False, "error": "Invalid provider"})
    
    if model in dashboard.model_config[provider]['models']:
        return jsonify({"success": False, "error": "Model already exists"})
    
    dashboard.model_config[provider]['models'].append(model)
    
    return jsonify({
        "success": True,
        "message": f"Added model {model} to {provider}"
    })

@app.route('/api/process/<process_id>/stop', methods=['POST'])
def api_stop_process(process_id):
    """Stop a running Alpha process."""
    try:
        with dashboard.process_lock:
            if process_id in dashboard.alpha_processes:
                process_info = dashboard.alpha_processes[process_id]
                process = process_info['process']
                
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    # Wait a bit for graceful termination
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()  # Force kill if it doesn't terminate gracefully
                    
                    return jsonify({
                        "success": True,
                        "message": f"Process {process_id} stopped successfully"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "Process is not running"
                    })
            else:
                return jsonify({
                    "success": False,
                    "error": "Process not found"
                })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/statistics')
def api_statistics():
    """API endpoint for statistics."""
    return jsonify(dashboard.get_statistics())

@app.route('/api/refresh')
def api_refresh():
    """API endpoint to refresh status."""
    return jsonify(dashboard.get_system_status())

if __name__ == '__main__':
    # Create templates directory if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Alpha Generator Dashboard')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the dashboard on')
    args = parser.parse_args()
    
    os.makedirs('templates', exist_ok=True)

    print("Starting Alpha Generator Dashboard...")
    print(f"Dashboard will be available at: http://localhost:{args.port}")
    print("Ollama WebUI: http://localhost:3000")
    print("Ollama API: http://localhost:11434")

    app.run(host='0.0.0.0', port=args.port, debug=True)
