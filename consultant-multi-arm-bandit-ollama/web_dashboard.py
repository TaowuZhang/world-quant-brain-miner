#!/usr/bin/env python3
"""
Web Dashboard for Integrated Alpha Mining System
Provides a simple web interface to monitor mining progress and results.
"""

import os
import json
import time
import argparse
import logging
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import psutil
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# HTML template for the dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Integrated Alpha Mining System - Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .content {
            padding: 30px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }
        .card {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border-left: 4px solid #667eea;
        }
        .card h3 {
            margin: 0 0 20px 0;
            color: #2c3e50;
            font-size: 1.3em;
            display: flex;
            align-items: center;
        }
        .card h3::before {
            content: "📊";
            margin-right: 10px;
            font-size: 1.2em;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #e9ecef;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-label {
            font-weight: 500;
            color: #495057;
        }
        .metric-value {
            font-weight: 600;
            color: #2c3e50;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online {
            background-color: #28a745;
        }
        .status-offline {
            background-color: #dc3545;
        }
        .status-warning {
            background-color: #ffc107;
        }
        .logs-section {
            background: #2c3e50;
            color: #ecf0f1;
            border-radius: 12px;
            padding: 25px;
            margin-top: 30px;
        }
        .logs-section h3 {
            margin: 0 0 20px 0;
            color: #ecf0f1;
            font-size: 1.3em;
        }
        .log-entry {
            background: #34495e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }
        .refresh-info {
            text-align: center;
            margin-top: 20px;
            color: #6c757d;
            font-size: 0.9em;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
        .gpu-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .gpu-card {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
        }
        .gpu-name {
            font-weight: 600;
            color: #1976d2;
            margin-bottom: 8px;
        }
        .gpu-metrics {
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Integrated Alpha Mining System</h1>
            <p>Real-time monitoring and status tracking</p>
        </div>
        
        <div class="content">
            <div class="grid">
                <div class="card">
                    <h3>System Status</h3>
                    <div class="metric">
                        <span class="metric-label">Alpha Orchestrator</span>
                        <span class="metric-value">
                            <span class="status-indicator status-{{ 'online' if system_status.orchestrator_online else 'offline' }}"></span>
                            {{ 'Online' if system_status.orchestrator_online else 'Offline' }}
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Ollama Service</span>
                        <span class="metric-value">
                            <span class="status-indicator status-{{ 'online' if system_status.ollama_online else 'offline' }}"></span>
                            {{ 'Online' if system_status.ollama_online else 'Offline' }}
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Web Dashboard</span>
                        <span class="metric-value">
                            <span class="status-indicator status-online"></span>
                            Online
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Last Update</span>
                        <span class="metric-value">{{ system_status.last_update }}</span>
                    </div>
                </div>

                <div class="card">
                    <h3>Mining Performance</h3>
                    <div class="metric">
                        <span class="metric-label">Total Adaptive Alphas</span>
                        <span class="metric-value">{{ mining_stats.total_adaptive_alphas }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Generator Alphas</span>
                        <span class="metric-value">{{ mining_stats.total_generator_alphas }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Best Sharpe Ratio</span>
                        <span class="metric-value">{{ "%.3f"|format(mining_stats.best_sharpe) if mining_stats.best_sharpe else 'N/A' }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Best Fitness</span>
                        <span class="metric-value">{{ "%.3f"|format(mining_stats.best_fitness) if mining_stats.best_fitness else 'N/A' }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Mining Cycle</span>
                        <span class="metric-value">{{ mining_stats.current_cycle }}/{{ mining_stats.total_cycles }}</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {{ (mining_stats.current_cycle / mining_stats.total_cycles * 100) if mining_stats.total_cycles > 0 else 0 }}%"></div>
                    </div>
                </div>

                <div class="card">
                    <h3>AI Model Status</h3>
                    <div class="metric">
                        <span class="metric-label">Current Model</span>
                        <span class="metric-value">{{ ollama_info.current_model }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Model Status</span>
                        <span class="metric-value">
                            <span class="status-indicator status-{{ 'online' if ollama_info.model_loaded else 'offline' }}"></span>
                            {{ 'Loaded' if ollama_info.model_loaded else 'Not Loaded' }}
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Available Models</span>
                        <span class="metric-value">{{ ollama_info.available_models|length }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Last Model Update</span>
                        <span class="metric-value">{{ ollama_info.last_update }}</span>
                    </div>
                </div>

                <div class="card">
                    <h3>System Resources</h3>
                    <div class="metric">
                        <span class="metric-label">CPU Usage</span>
                        <span class="metric-value">{{ "%.1f"|format(system_metrics.cpu_percent) }}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Memory Usage</span>
                        <span class="metric-value">{{ "%.1f"|format(system_metrics.memory_percent) }}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Disk Usage</span>
                        <span class="metric-value">{{ "%.1f"|format(system_metrics.disk_percent) }}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">GPU Count</span>
                        <span class="metric-value">{{ system_metrics.gpu_count }}</span>
                    </div>
                    {% if system_metrics.gpu_info %}
                    <div class="gpu-info">
                        {% for gpu in system_metrics.gpu_info %}
                        <div class="gpu-card">
                            <div class="gpu-name">{{ gpu.name }}</div>
                            <div class="gpu-metrics">
                                <div>VRAM: {{ gpu.memory_used }}MB / {{ gpu.memory_total }}MB</div>
                                <div>Utilization: {{ gpu.utilization }}%</div>
                                <div>Temperature: {{ gpu.temperature }}°C</div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    {% endif %}
                </div>
            </div>

            <div class="logs-section">
                <h3>📋 Recent Activity Logs</h3>
                <div class="log-entry">{{ recent_logs }}</div>
            </div>

            <div class="refresh-info">
                Auto-refreshing every 30 seconds | Last updated: {{ system_status.last_update }}
            </div>
        </div>
    </div>

    <script>
        // Auto-refresh the page every 30 seconds
        setTimeout(function() {
            location.reload();
        }, 30000);
    </script>
</body>
</html>
"""

def get_system_metrics():
    """Get system resource metrics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get GPU information
        gpu_info = []
        gpu_count = 0
        try:
            gpus = GPUtil.getGPUs()
            gpu_count = len(gpus)
            for gpu in gpus:
                gpu_info.append({
                    'name': gpu.name,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'utilization': gpu.load * 100,
                    'temperature': gpu.temperature
                                    })
        except Exception as e:
            gpu_count = 0
            gpu_info = []
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': (disk.used / disk.total) * 100,
            'gpu_count': gpu_count,
            'gpu_info': gpu_info
        }
    except Exception as e:
        return {
            'cpu_percent': 0,
            'memory_percent': 0,
            'disk_percent': 0,
            'gpu_count': 0,
            'gpu_info': []
        }

def get_ollama_status():
    """Check if Ollama service is running."""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
                    except:
        return False

def get_orchestrator_status():
    """Check if alpha orchestrator is running."""
    try:
        return os.path.exists('orchestrator_state.json')
                        except:
        return False

def get_mining_stats():
    """Get mining statistics from orchestrator state."""
    try:
        if os.path.exists('orchestrator_state.json'):
            with open('orchestrator_state.json', 'r') as f:
                state = json.load(f)
                return {
                    'total_adaptive_alphas': state.get('total_adaptive_alphas', 0),
                    'total_generator_alphas': state.get('total_generator_alphas', 0),
                    'best_sharpe': state.get('best_sharpe', 0),
                    'best_fitness': state.get('best_fitness', 0),
                    'current_cycle': state.get('current_cycle', 0),
                    'total_cycles': state.get('total_cycles', 0)
                }
        except Exception as e:
        pass
    
    return {
        'total_adaptive_alphas': 0,
        'total_generator_alphas': 0,
        'best_sharpe': 0,
        'best_fitness': 0,
        'current_cycle': 0,
        'total_cycles': 0
    }

def get_ollama_info():
    """Get Ollama model information."""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            current_model = 'deepseek-r1:8b'  # Default model
            return {
                'current_model': current_model,
                'model_loaded': len(models) > 0,
                'available_models': models,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
        pass
            
            return {
        'current_model': 'Unknown',
        'model_loaded': False,
        'available_models': [],
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def get_recent_logs():
    """Get recent logs from various log files."""
    log_files = [
        'logs/alpha_orchestrator.log',
        'logs/integrated_alpha_miner.log',
        'logs/adaptive_alpha_miner.log',
        'logs/alpha_generator_ollama.log'
    ]
    
    recent_logs = []
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Get last 10 lines
                    recent_lines = lines[-10:] if len(lines) > 10 else lines
                    recent_logs.extend([f"[{log_file}] {line.strip()}" for line in recent_lines])
            except Exception as e:
                recent_logs.append(f"[{log_file}] Error reading log: {str(e)}")
    
    # Sort by timestamp if available
    recent_logs.sort(reverse=True)
    
    # Return last 20 log entries
    return '\n'.join(recent_logs[-20:]) if recent_logs else "No recent logs available"

@app.route('/')
def dashboard():
    """Main dashboard page."""
    system_metrics = get_system_metrics()
    system_status = {
        'orchestrator_online': get_orchestrator_status(),
        'ollama_online': get_ollama_status(),
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    mining_stats = get_mining_stats()
    ollama_info = get_ollama_info()
    recent_logs = get_recent_logs()
    
    return render_template_string(DASHBOARD_TEMPLATE,
                                system_metrics=system_metrics,
                                system_status=system_status,
                                mining_stats=mining_stats,
                                ollama_info=ollama_info,
                                recent_logs=recent_logs)

@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'system_status': {
            'orchestrator_online': get_orchestrator_status(),
            'ollama_online': get_ollama_status()
        },
        'mining_stats': get_mining_stats(),
        'system_metrics': get_system_metrics(),
        'ollama_info': get_ollama_info()
    })

def main():
    """Main function to run the web dashboard."""
    parser = argparse.ArgumentParser(description='Web Dashboard for Integrated Alpha Mining System')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the dashboard on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Integrated Alpha Mining System Dashboard")
    print(f"📊 Dashboard will be available at: http://{args.host}:{args.port}")
    print(f"🔧 Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
