# Model Fleet Management System

This document explains the Model Fleet Management system that automatically handles VRAM issues by downgrading to smaller models.

## Overview

The Model Fleet Management system provides automatic model switching when VRAM issues occur. It maintains a fleet of models from largest to smallest and automatically downgrades when VRAM errors are detected.

## Model Fleet Hierarchy

The system uses the following model hierarchy (largest to smallest):

1. **llama3.2:3b** (2.0 GB) - Large model - 3B parameters
2. **phi3:mini** (2.2 GB) - Medium model - Phi3 mini
3. **tinyllama:1.1b** (637 MB) - Small model - 1.1B parameters
4. **qwen2.5:0.5b** (397 MB) - Tiny model - 0.5B parameters

## How It Works

### 1. **VRAM Error Detection**
The system monitors Docker logs for VRAM-related errors:
- "gpu VRAM usage didn't recover within timeout"
- "VRAM usage didn't recover"
- "gpu memory exhausted"
- "CUDA out of memory"
- "GPU memory allocation failed"

### 2. **Automatic Downgrading**
- After 3 consecutive VRAM errors, the system automatically downgrades to the next smaller model
- The application is restarted with the new model
- VRAM error count is reset after successful downgrade

### 3. **State Persistence**
- Current model selection and error counts are saved to `model_fleet_state.json`
- System remembers the last used model across restarts
- Can be reset to largest model if needed

## Usage

### Starting the System

**Linux/Mac:**
```bash
./start_with_model_fleet.sh
```

**Windows:**
```cmd
start_with_model_fleet.bat
```

### Manual Commands

**Check Fleet Status:**
```bash
python model_fleet_manager.py --status
```

**Start VRAM Monitoring:**
```bash
python model_fleet_manager.py --monitor
```

**Reset to Largest Model:**
```bash
python model_fleet_manager.py --reset
```

**Force Downgrade:**
```bash
python model_fleet_manager.py --downgrade
```

## Configuration

### Model Fleet Configuration

Edit `model_fleet_manager.py` to modify the model fleet:

```python
self.model_fleet = [
    ModelInfo("llama3.2:3b", 2048, 1, "Large model - 3B parameters"),
    ModelInfo("phi3:mini", 2200, 2, "Medium model - Phi3 mini"),
    ModelInfo("tinyllama:1.1b", 637, 3, "Small model - 1.1B parameters"),
    ModelInfo("qwen2.5:0.5b", 397, 4, "Tiny model - 0.5B parameters"),
]
```

### VRAM Error Threshold

Modify the error threshold in `model_fleet_manager.py`:

```python
self.max_vram_errors = 3  # Number of VRAM errors before downgrading
```

## Monitoring and Logs

### Log Files
- `model_fleet.log` - Model fleet manager logs
- `vram_monitor.log` - VRAM monitoring logs
- `alpha_generator_ollama.log` - Alpha generator logs

### Status Information

The status command shows:
- Current model in use
- VRAM error count
- Available models
- Fleet size
- Whether downgrade is possible

Example output:
```json
{
  "current_model": {
    "name": "llama3.2:3b",
    "size_mb": 2048,
    "description": "Large model - 3B parameters",
    "index": 0
  },
  "vram_error_count": 0,
  "max_vram_errors": 3,
  "available_models": [
    "tinyllama:1.1b",
    "qwen2.5:0.5b",
    "phi3:mini",
    "llama3.2:3b"
  ],
  "fleet_size": 4,
  "can_downgrade": true
}
```

## Troubleshooting

### Model Not Available
If a model is not available, the system will automatically download it. If download fails:
1. Check internet connection
2. Verify Docker container is running
3. Check available disk space

### VRAM Issues Persist
If VRAM issues persist even with the smallest model:
1. Reduce batch size in `docker-compose.gpu.yml`
2. Lower GPU layers setting
3. Reduce memory limits
4. Add more VRAM cleanup intervals

### System Won't Start
If the system won't start:
1. Check Docker is running
2. Verify NVIDIA drivers are installed
3. Check container logs: `docker-compose -f docker-compose.gpu.yml logs`

## Integration with Alpha Mining

The model fleet manager integrates seamlessly with the alpha mining system:

1. **Automatic Configuration Updates**: Updates `alpha_generator_ollama.py` with new model
2. **Container Restart**: Restarts the Docker container with new model
3. **State Persistence**: Maintains model selection across restarts
4. **Error Recovery**: Automatically recovers from VRAM issues

## Performance Considerations

### Model Size vs Performance
- **Larger models** (3B+): Better quality, higher VRAM usage
- **Medium models** (1-3B): Balanced performance
- **Smaller models** (<1B): Lower VRAM usage, reduced quality

### Recommended Settings
- Start with largest model for best results
- Let system automatically downgrade if needed
- Monitor logs for performance patterns
- Consider manual reset to larger model during low-usage periods

## Future Enhancements

Potential improvements:
1. **Dynamic Model Loading**: Load models on-demand
2. **Performance Metrics**: Track model performance vs VRAM usage
3. **Smart Scheduling**: Use larger models during off-peak hours
4. **Custom Model Support**: Add support for custom fine-tuned models
5. **Multi-GPU Support**: Distribute models across multiple GPUs
