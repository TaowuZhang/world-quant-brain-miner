# Integrated Model Fleet Management System

## Overview

The Model Fleet Management system has been successfully integrated into the `alpha_orchestrator.py` to provide automatic VRAM management within the Docker environment. The system now monitors Ollama's Docker logs directly for VRAM errors and automatically downgrades to smaller models when needed.

## Key Features

### 1. **Integrated VRAM Monitoring**
- **Direct Docker Log Monitoring**: The system now monitors `naive-ollma-gpu` container logs directly using `docker logs --tail 50 naive-ollma-gpu`
- **Real-time Error Detection**: Detects VRAM errors from Ollama's actual log output, not Python application logs
- **Automatic Model Switching**: Downgrades to smaller models after 3 consecutive VRAM errors

### 2. **Model Fleet Hierarchy**
The system maintains a fleet of models from largest to smallest:
1. **llama3.2:3b** (2.0 GB) - Large model - 3B parameters
2. **phi3:mini** (2.2 GB) - Medium model - Phi3 mini  
3. **tinyllama:1.1b** (637 MB) - Small model - 1.1B parameters
4. **qwen2.5:0.5b** (397 MB) - Tiny model - 0.5B parameters

### 3. **Docker Integration**
- **Container-aware**: Works within the Docker environment
- **Ollama API Integration**: Uses Ollama's REST API to check available models and download new ones
- **Automatic Configuration Updates**: Updates `alpha_generator_ollama.py` with new model settings
- **Process Management**: Restarts alpha generator processes with new models

## How It Works

### VRAM Error Detection
The system monitors Docker logs for these VRAM error patterns:
- `"gpu VRAM usage didn't recover within timeout"`
- `"VRAM usage didn't recover"`
- `"gpu memory exhausted"`
- `"CUDA out of memory"`
- `"GPU memory allocation failed"`
- `"msg=\"gpu VRAM usage didn't recover within timeout\""`
- `"level=WARN source=sched.go"`

### Automatic Downgrading Process
1. **Error Detection**: Monitors Docker logs every 30 seconds
2. **Error Counting**: Increments VRAM error count on each detection
3. **Threshold Check**: After 3 errors, triggers model downgrade
4. **Model Download**: Ensures new model is available via Ollama API
5. **Configuration Update**: Updates `alpha_generator_ollama.py` with new model
6. **Process Restart**: Restarts alpha generator with new model
7. **State Persistence**: Saves current model and error count to `model_fleet_state.json`

## Usage

### Command Line Interface
The orchestrator now supports model fleet management commands:

```bash
# Check current fleet status
python alpha_orchestrator.py --mode fleet-status

# Reset to largest model
python alpha_orchestrator.py --mode fleet-reset

# Force downgrade to next smaller model
python alpha_orchestrator.py --mode fleet-downgrade

# Start continuous mining with VRAM monitoring
python alpha_orchestrator.py --mode continuous
```

### Docker Integration
The Docker container automatically uses the integrated system:
```yaml
command: ["python", "alpha_orchestrator.py", "--mode", "continuous", "--batch-size", "2", "--max-concurrent", "2"]
```

## Technical Implementation

### Key Classes

#### `ModelFleetManager`
- **State Management**: Loads/saves model fleet state to JSON
- **Model Discovery**: Uses Ollama API to check available models
- **Error Detection**: Parses Docker logs for VRAM errors
- **Configuration Updates**: Modifies Python files with new model settings

#### `AlphaOrchestrator` (Enhanced)
- **VRAM Monitoring Thread**: Runs background monitoring loop
- **Process Management**: Handles alpha generator restarts
- **Fleet Integration**: Uses current model from fleet manager
- **Error Recovery**: Automatically recovers from VRAM issues

### State Persistence
The system maintains state in `model_fleet_state.json`:
```json
{
  "current_model_index": 0,
  "vram_error_count": 0,
  "current_model": "llama3.2:3b",
  "timestamp": 1234567890.123
}
```

## Benefits

### 1. **Automatic Recovery**
- No manual intervention needed for VRAM issues
- System automatically finds optimal model for current hardware
- Continuous operation without downtime

### 2. **Performance Optimization**
- Starts with largest model for best quality
- Gracefully degrades when hardware limits are reached
- Maintains operation even with limited VRAM

### 3. **Docker Native**
- Works seamlessly within containerized environment
- Monitors actual Ollama logs, not application logs
- Integrates with existing Docker Compose setup

### 4. **State Persistence**
- Remembers model selection across restarts
- Maintains error counts for intelligent downgrading
- Can be reset to largest model when needed

## Monitoring and Logs

### Log Files
- `alpha_orchestrator.log` - Main orchestrator logs including VRAM monitoring
- `model_fleet_state.json` - Current model and error state
- Docker logs - Direct Ollama VRAM error messages

### Status Information
The fleet status command shows:
- Current model in use
- VRAM error count and threshold
- Available models in Ollama
- Fleet size and downgrade capability

## Future Enhancements

1. **Performance Metrics**: Track model performance vs VRAM usage
2. **Smart Scheduling**: Use larger models during off-peak hours
3. **Multi-GPU Support**: Distribute models across multiple GPUs
4. **Custom Model Support**: Add support for custom fine-tuned models
5. **Dynamic Thresholds**: Adjust error thresholds based on model size

## Conclusion

The integrated Model Fleet Management system provides robust, automatic VRAM management for the WorldQuant Alpha Mining system. It ensures continuous operation by intelligently managing model resources within the Docker environment, automatically detecting and recovering from VRAM issues without manual intervention.
