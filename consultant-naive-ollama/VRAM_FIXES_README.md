# VRAM Fixes and Alpha Management Improvements

This document describes the improvements made to fix VRAM issues and implement automatic alpha removal from `hopeful_alphas.json` after successful mining.

## Changes Made

### 1. VRAM Management Improvements

#### Docker Configuration Updates (`docker-compose.gpu.yml`)
- **Reduced GPU layers**: Changed from 50 to 35 layers to reduce VRAM usage
- **Added memory limits**: Set GPU memory utilization to 80%
- **Reduced batch size**: Changed from 3 to 2 concurrent operations
- **Increased sleep time**: Changed from 60 to 90 seconds between batches
- **Reduced memory limit**: Changed from 16GB to 12GB system memory

#### Alpha Generator Updates (`alpha_generator_ollama.py`)
- **Reduced default concurrency**: Changed from 3 to 2 concurrent workers
- **Added VRAM cleanup**: Automatic garbage collection every 10 operations
- **Added operation tracking**: Monitor operation count for cleanup scheduling

### 2. Alpha Removal Functionality

#### Alpha Expression Miner Updates (`alpha_expression_miner.py`)
- **Added `remove_alpha_from_hopeful()` method**: Automatically removes successfully mined alphas
- **Integrated removal logic**: Alphas are removed only after successful mining
- **Failed mining protection**: Failed alphas remain in `hopeful_alphas.json` for retry

#### Orchestrator Updates (`alpha_orchestrator.py`)
- **Updated comments**: Clarified that mining automatically removes successful alphas
- **Better error handling**: Failed alphas are preserved for retry

### 3. VRAM Monitoring System

#### New VRAM Monitor (`vram_monitor.py`)
- **Real-time monitoring**: Checks GPU memory usage every 30-60 seconds
- **Automatic cleanup**: Attempts VRAM cleanup when usage exceeds threshold
- **Service restart**: Restarts Ollama service if cleanup fails
- **Configurable thresholds**: Adjustable VRAM usage limits

#### Startup Scripts
- **`start_with_vram_monitor.sh`**: Linux/macOS startup with VRAM monitoring
- **`start_with_vram_monitor.bat`**: Windows startup with VRAM monitoring

## Usage

### Starting the System with VRAM Monitoring

#### Windows
```batch
cd naive-ollama
start_with_vram_monitor.bat
```

#### Linux/macOS
```bash
cd naive-ollama
chmod +x start_with_vram_monitor.sh
./start_with_vram_monitor.sh
```

### Manual VRAM Monitoring
```bash
python vram_monitor.py --threshold 0.85 --interval 30
```

### Manual Alpha Mining
```bash
python alpha_expression_miner.py --expression "rank(divide(revenue, assets))" --auto-mode
```

## Configuration

### VRAM Monitor Settings
- **Threshold**: Default 0.85 (85% VRAM usage triggers cleanup)
- **Interval**: Default 30 seconds between checks
- **Max restarts**: 3 automatic restarts before manual intervention

### Docker Settings
- **GPU layers**: 35 (reduced from 50)
- **Memory utilization**: 80%
- **Batch size**: 2 (reduced from 3)
- **Sleep time**: 90 seconds (increased from 60)

## How Alpha Removal Works

1. **Alpha Generation**: New alphas are added to `hopeful_alphas.json`
2. **Mining Process**: Alpha expression miner processes alphas from the file
3. **Success Check**: If mining produces results, the alpha is removed
4. **Failure Handling**: If mining fails, the alpha remains for retry
5. **Automatic Cleanup**: No manual intervention required

## Monitoring and Logs

### Log Files
- `alpha_miner.log`: Alpha mining operations
- `alpha_generator_ollama.log`: Alpha generation and testing
- `vram_monitor.log`: VRAM monitoring and cleanup operations

### Dashboard Access
- **Main Dashboard**: http://localhost:5000
- **Ollama WebUI**: http://localhost:3000

## Troubleshooting

### VRAM Issues
1. **High VRAM usage**: VRAM monitor will automatically attempt cleanup
2. **Service restarts**: If cleanup fails, services will restart automatically
3. **Manual intervention**: If max restarts exceeded, manual restart required

### Alpha Mining Issues
1. **Failed mining**: Check `alpha_miner.log` for specific errors
2. **No alphas found**: Ensure `hopeful_alphas.json` contains valid alphas
3. **Authentication issues**: Verify `credential.txt` is properly configured

### System Performance
1. **Reduce concurrency**: Lower `--max-concurrent` parameter
2. **Increase sleep time**: Increase `--sleep-time` between batches
3. **Monitor resources**: Use `nvidia-smi` to check GPU usage

## Benefits

1. **Automatic VRAM Management**: Prevents system crashes due to memory issues
2. **Self-Cleaning Alpha List**: No manual cleanup of processed alphas required
3. **Better Resource Utilization**: Optimized for stable long-term operation
4. **Improved Reliability**: Automatic recovery from memory issues
5. **Reduced Manual Intervention**: System manages itself more effectively

## Future Improvements

1. **Dynamic VRAM adjustment**: Automatically adjust based on available memory
2. **Predictive cleanup**: Clean VRAM before it reaches threshold
3. **Better error recovery**: More sophisticated retry mechanisms
4. **Performance metrics**: Track and optimize system performance
