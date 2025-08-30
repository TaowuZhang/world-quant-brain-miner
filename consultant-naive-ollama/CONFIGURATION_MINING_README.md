# Alpha Expression Miner with Configuration Traversal

This enhanced version of the Alpha Expression Miner now supports testing alpha expressions across multiple simulation configurations based on the WorldQuant Brain API schema.

## New Features

### 1. Multi-Configuration Testing
Instead of using a single hardcoded simulation configuration, the miner now generates and tests multiple configurations that traverse through:

- **Regions**: USA, GLB, EUR, ASI, CHN
- **Universes**: Region-specific universes (TOP3000, TOP1000, etc.)
- **Neutralization Strategies**: NONE, INDUSTRY, SECTOR, MARKET, etc.
- **Truncation Values**: 0.01 to 0.30
- **Decay Values**: 0 to 512
- **Other Parameters**: Pasteurization, Nan handling, etc.

### 2. API Schema Compliance
All generated configurations strictly follow the WorldQuant Brain API schema, ensuring:
- Region-specific universe and delay constraints
- Valid neutralization strategies for each region
- Proper parameter ranges and types

### 3. Configuration Management
- **Automatic Generation**: Configurations are generated based on the API schema
- **Configurable Limits**: Control the number of configurations per alpha
- **Configuration Logging**: Summary of all configurations used
- **Configuration Export**: Save configurations to JSON files for reference

## Usage

### Basic Usage
```bash
python alpha_expression_miner.py --expression "your_alpha_expression" --auto-mode
```

### Advanced Usage with Configuration Control
```bash
python alpha_expression_miner.py \
    --expression "your_alpha_expression" \
    --auto-mode \
    --max-configs 20 \
    --pool-size 3 \
    --save-configs "my_configs.json" \
    --output "results.json"
```

### Command Line Arguments

- `--max-configs`: Maximum number of simulation configurations per alpha (default: 10)
- `--pool-size`: Number of alphas to process concurrently in each pool (default: 5)
- `--save-configs`: File to save simulation configurations (default: simulation_configs.json)
- `--auto-mode`: Run in automated mode without user interaction
- `--expression`: Base alpha expression to mine variations from
- `--output`: Output file for results (default: mined_expressions.json)

## Concurrency Control

The miner processes alphas in pools to control the number of concurrent simulations:

- **Pool Size**: Number of alphas processed simultaneously in each pool (default: 5)
- **Total Concurrent Simulations**: `pool_size × max_configs_per_alpha`
- **Example**: With `--pool-size 3` and `--max-configs 10`, you'll have 30 concurrent simulations per pool

This helps balance between speed and API rate limits. Lower pool sizes are recommended if you encounter rate limiting issues.

## Configuration Generation Strategy

The miner generates configurations in the following priority order:

1. **Base Configuration**: USA, TOP3000, INDUSTRY neutralization, standard parameters
2. **Regional Variations**: Different regions with their corresponding universes and delays
3. **Neutralization Strategies**: Various neutralization approaches for main regions
4. **Truncation Values**: Different truncation levels (0.05, 0.10, 0.15, 0.20)
5. **Decay Values**: Different decay periods (25, 50, 256, 512)
6. **Processing Options**: Different pasteurization and nan handling strategies

## Output Format

Each result now includes configuration information:

```json
{
  "expression": "your_alpha_expression",
  "config": {
    "type": "REGULAR",
    "settings": {
      "region": "USA",
      "universe": "TOP3000",
      "neutralization": "INDUSTRY",
             "truncation": 0.08,
       "decay": 0,
       // ... other settings
    }
  },
  "config_id": "config_0",
  "result": {
    // WorldQuant Brain simulation result
  }
}
```

## Testing

Run the test script to verify configuration generation:

```bash
python test_config_generation.py
```

This will:
- Test configuration generation with different limits
- Verify API schema compliance
- Save sample configurations to files
- Display configuration summaries

## Configuration Files

The miner generates several files:

- `simulation_configs.json`: All simulation configurations used
- `mined_expressions.json`: Results with configuration information
- `alpha_miner.log`: Detailed logging of the mining process

## Benefits

1. **Comprehensive Testing**: Test alphas across multiple market conditions and parameter sets
2. **Discovery**: Find configurations where alphas perform better
3. **Robustness**: Ensure alphas work across different regions and settings
4. **Compliance**: All configurations follow API constraints
5. **Scalability**: Control the number of configurations to balance speed vs. coverage

## Example Output

```
2024-01-15 10:30:00 - INFO - Max configs per alpha: 15
2024-01-15 10:30:00 - INFO - Pool size (concurrent alphas): 3
2024-01-15 10:30:00 - INFO - Using 15 different simulation configurations
2024-01-15 10:30:00 - INFO - Configuration summary: {
  'total_configs': 15,
  'regions': ['USA', 'GLB', 'EUR', 'ASI', 'CHN'],
  'universes': ['TOP3000', 'TOP1000', 'MINVOL1M', 'TOP2500'],
  'neutralizations': ['INDUSTRY', 'NONE', 'SECTOR', 'MARKET'],
     'truncations': [0.08, 0.05, 0.10, 0.15, 0.20],
   'decays': [0, 25, 50, 256, 512]
}
2024-01-15 10:30:00 - INFO - Saved 15 configurations to simulation_configs.json
2024-01-15 10:30:00 - INFO - Using pool size of 3 alphas per pool
2024-01-15 10:30:00 - INFO - Total concurrent simulations per pool: 45 (3 alphas × 15 configs)
2024-01-15 10:30:00 - INFO - Created 2 pools of size 3
```

This enhanced miner provides much more comprehensive testing of alpha expressions across different market conditions and parameter settings, helping to identify the most robust and profitable configurations.
