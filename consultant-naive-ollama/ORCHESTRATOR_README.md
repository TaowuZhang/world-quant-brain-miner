# Alpha Orchestrator - Concurrent Execution

## Overview

The Alpha Orchestrator has been enhanced to run `alpha_generator_ollama` and `alpha_expression_miner` concurrently, with proper coordination and the ability to limit concurrent simulations to 3 (configurable).

## Key Improvements

### 1. Concurrent Execution
- **Before**: Sequential execution - generator runs first, then miner
- **After**: Concurrent execution - both run simultaneously
- Generator continuously creates alphas and logs promising ones to `hopeful_alphas.json`
- Miner continuously monitors `hopeful_alphas.json` and processes new alphas

### 2. Max Concurrent Simulations
- Configurable limit (default: 3) for concurrent simulations
- Prevents overwhelming the WorldQuant Brain API
- Can be adjusted via command line argument

### 3. Process Management
- Proper process lifecycle management
- Automatic restart of failed processes
- Graceful shutdown with cleanup

### 4. Better Error Handling
- Improved error handling for corrupted `hopeful_alphas.json`
- Better logging and monitoring
- Retry mechanisms for failed operations

## Usage

### Basic Usage
```bash
# Run with default settings (3 max concurrent simulations)
python alpha_orchestrator.py --mode continuous

# Run with custom max concurrent simulations
python alpha_orchestrator.py --mode continuous --max-concurrent 5

# Run with custom mining interval (in hours)
python alpha_orchestrator.py --mode continuous --mining-interval 4
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `continuous` | Operation mode: `daily`, `continuous`, `miner`, `submitter`, `generator` |
| `--max-concurrent` | `3` | Maximum concurrent simulations |
| `--mining-interval` | `6` | Mining interval in hours for continuous mode |
| `--batch-size` | `3` | Batch size for operations |
| `--credentials` | `./credential.txt` | Path to credentials file |
| `--ollama-url` | `http://localhost:11434` | Ollama API URL |

### Modes

1. **continuous** (default): Runs both generator and miner concurrently
2. **daily**: Runs a complete daily workflow (generator → miner → submitter)
3. **miner**: Runs only the alpha expression miner
4. **submitter**: Runs only the alpha submitter
5. **generator**: Runs only the alpha generator

## How It Works

### Concurrent Execution Flow

1. **Alpha Generator Process**
   - Runs continuously in background
   - Generates alpha expressions using Ollama
   - Tests alphas and logs promising ones to `hopeful_alphas.json`
   - Respects max concurrent simulation limit

2. **Alpha Expression Miner Process**
   - Runs in separate thread
   - Monitors `hopeful_alphas.json` for new alphas
   - Processes alphas when found (every 6 hours by default)
   - Mines variations of promising alphas

3. **Coordination**
   - Both processes run independently
   - Generator creates data, miner consumes it
   - No blocking or waiting between processes

### File Dependencies

- `hopeful_alphas.json`: Created by generator, consumed by miner
- `credential.txt`: Authentication credentials
- `submission_log.json`: Tracks daily submissions

## Testing

Run the test script to verify the orchestrator works correctly:

```bash
python test_orchestrator.py
```

This will test:
- Orchestrator initialization
- Command line argument handling
- Concurrent execution capabilities

## Monitoring

### Log Files
- `alpha_orchestrator.log`: Main orchestrator logs
- `alpha_generator_ollama.log`: Generator logs
- `alpha_miner.log`: Miner logs

### Key Log Messages
- `"Both alpha generator and expression miner are running concurrently"`
- `"Max concurrent simulations: 3"`
- `"Found X alphas to mine"`
- `"Alpha generator started with PID: X"`

## Troubleshooting

### Common Issues

1. **hopeful_alphas.json not found**
   - Normal during startup - generator needs time to create promising alphas
   - Check generator logs for errors

2. **Authentication failures**
   - Verify `credential.txt` exists and has correct format
   - Check WorldQuant Brain API status

3. **Process crashes**
   - Orchestrator will automatically restart failed processes
   - Check logs for specific error messages

4. **Too many concurrent simulations**
   - Reduce `--max-concurrent` value
   - Check WorldQuant Brain API limits

### Performance Tuning

- **Increase throughput**: Increase `--max-concurrent` (but respect API limits)
- **Reduce resource usage**: Decrease `--batch-size`
- **Faster mining**: Decrease `--mining-interval`
- **More frequent checks**: Modify check interval in `start_alpha_expression_miner_continuous`

## Docker Support

The orchestrator works with the existing Docker setup. Use the same commands but ensure the orchestrator has access to all required files and the Ollama service.

## Future Enhancements

- Web dashboard integration for monitoring
- Dynamic adjustment of concurrent limits based on API response times
- Machine learning-based alpha prioritization
- Integration with additional mining strategies
