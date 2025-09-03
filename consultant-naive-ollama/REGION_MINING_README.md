# Region-Specific Mining Functionality

## Overview

The `alpha_expression_miner.py` script now supports region-specific mining, allowing you to generate ALL possible valid simulation configurations for a specific region instead of using the default handpicked approach across all regions.

## New Command Line Arguments

### `--region`
- **Type**: String
- **Default**: None (uses original behavior)
- **Values**: USA, GLB, EUR, ASI, CHN
- **Description**: When specified, generates ALL possible valid combinations for the specified region

### `--pool-size`
- **Type**: Integer
- **Default**: 5
- **Description**: Number of alphas to process concurrently in each pool

### `--max-concurrent-sims`
- **Type**: Integer
- **Default**: 50
- **Description**: Maximum number of concurrent simulations across all pools (helps avoid API limits)

## How It Works

### Without `--region` (Default Behavior)
- Uses the original handpicked approach
- Generates configurations across multiple regions
- Focuses on key parameter variations
- Limited number of configurations for efficiency

### With `--region` (New Behavior)
- Generates ALL possible valid combinations for the specified region
- Respects region-specific constraints (available universes, delays, neutralizations)
- Can generate a very large number of configurations
- Uses the `--max-configs` parameter to limit total configurations

## Usage Examples

### 1. Mine ALL USA Combinations
```bash
python alpha_expression_miner.py \
  --expression "your_alpha_expression" \
  --region USA \
  --max-configs 1000 \
  --pool-size 3 \
  --max-concurrent-sims 50 \
  --auto-mode
```

### 2. Mine ALL Global Combinations
```bash
python alpha_expression_miner.py \
  --expression "your_alpha_expression" \
  --region GLB \
  --max-configs 1000 \
  --pool-size 2 \
  --max-concurrent-sims 40 \
  --auto-mode
```

### 3. Mine ALL Europe Combinations
```bash
python alpha_expression_miner.py \
  --expression "your_alpha_expression" \
  --region EUR \
  --max-configs 1000 \
  --pool-size 4 \
  --max-concurrent-sims 60 \
  --auto-mode
```

### 4. Use Default Behavior (All Regions)
```bash
python alpha_expression_miner.py \
  --expression "your_alpha_expression" \
  --auto-mode
```

### 5. Conservative Settings for Testing
```bash
python alpha_expression_miner.py \
  --expression "your_alpha_expression" \
  --region USA \
  --max-configs 100 \
  --pool-size 2 \
  --max-concurrent-sims 25 \
  --auto-mode
```

## Region-Specific Constraints

Each region has different available options:

### USA
- **Universes**: TOP3000, TOP1000, TOP500, TOP200, ILLIQUID_MINVOL1M, TOPSP500
- **Delays**: 1, 0
- **Neutralizations**: NONE, REVERSION_AND_MOMENTUM, STATISTICAL, CROWDING, FAST, SLOW, MARKET, SECTOR, INDUSTRY, SUBINDUSTRY, SLOW_AND_FAST

### GLB (Global)
- **Universes**: TOP3000, MINVOL1M, TOPDIV3000
- **Delays**: 1
- **Neutralizations**: NONE, REVERSION_AND_MOMENTUM, STATISTICAL, CROWDING, FAST, SLOW, MARKET, SECTOR, INDUSTRY, SUBINDUSTRY, COUNTRY, SLOW_AND_FAST

### EUR (Europe)
- **Universes**: TOP2500, TOP1200, TOP800, TOP400, ILLIQUID_MINVOL1M
- **Delays**: 1, 0
- **Neutralizations**: NONE, REVERSION_AND_MOMENTUM, STATISTICAL, CROWDING, FAST, SLOW, MARKET, SECTOR, INDUSTRY, SUBINDUSTRY, COUNTRY, SLOW_AND_FAST

### ASI (Asia)
- **Universes**: MINVOL1M, ILLIQUID_MINVOL1M
- **Delays**: 1
- **Neutralizations**: NONE, REVERSION_AND_MOMENTUM, STATISTICAL, CROWDING, FAST, SLOW, MARKET, SECTOR, INDUSTRY, SUBINDUSTRY, COUNTRY, SLOW_AND_FAST

### CHN (China)
- **Universes**: TOP2000U
- **Delays**: 0, 1
- **Neutralizations**: NONE, REVERSION_AND_MOMENTUM, CROWDING, FAST, SLOW, MARKET, SECTOR, INDUSTRY, SUBINDUSTRY, SLOW_AND_FAST

## Common Parameters (All Regions)

These parameters are available across all regions:
- **Truncations**: 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20, 0.25, 0.30
- **Decays**: 0 to 512 (all integer values)
- **Pasteurization**: ON, OFF
- **Nan Handling**: ON, OFF

## Configuration Generation

When `--region` is specified, the system generates configurations by combining:

1. **All available universes** for the region
2. **All available delays** for the region  
3. **All available neutralizations** for the region
4. **All truncation values** (14 values)
5. **All decay values** (513 values)
6. **All pasteurization options** (2 values)
7. **All nan handling options** (2 values)

**Total possible combinations** = universes × delays × neutralizations × 14 × 513 × 2 × 2

## Important Considerations

### 1. Configuration Count
- Region-specific mining can generate thousands of configurations
- Use `--max-configs` to limit the total number
- The system will respect this limit and generate as many as possible

### 2. API Limits
- WorldQuant Brain has rate limits and simulation limits
- Large numbers of configurations may hit these limits
- Monitor the logs for rate limit warnings

### 3. Processing Time
- More configurations = longer processing time
- Consider using smaller `--max-configs` values for testing
- Increase gradually based on your needs and API limits

### 4. Memory Usage
- Large numbers of configurations consume more memory
- Monitor system resources during execution

## Concurrent Simulation Management

The system now includes intelligent management of concurrent simulations to avoid hitting WorldQuant Brain API limits:

### Automatic Pool Size Adjustment
- The system automatically calculates the optimal pool size based on `--max-concurrent-sims`
- Formula: `optimal_pool_size = min(pool_size, max_concurrent_sims // max_configs_per_alpha)`
- If the calculated pool size is less than 1, it defaults to 1 with a warning

### Example Pool Size Calculation
```bash
# With default values:
# --pool-size 5, --max-configs 10, --max-concurrent-sims 50
# optimal_pool_size = min(5, 50 // 10) = min(5, 5) = 5
# Total concurrent simulations = 5 × 10 = 50

# With higher configs:
# --pool-size 5, --max-configs 20, --max-concurrent-sims 50
# optimal_pool_size = min(5, 50 // 20) = min(5, 2) = 2
# Total concurrent simulations = 2 × 20 = 40
```

### Error Handling
- **CONCURRENT_SIMULATION_LIMIT_EXCEEDED**: The system waits 5 seconds and continues
- **Rate Limiting**: Automatic retry with exponential backoff
- **Authentication Errors**: Automatic re-authentication

### Best Practices for Concurrent Simulations
1. **Start Conservative**: Begin with `--max-concurrent-sims 25` and increase gradually
2. **Monitor Logs**: Watch for warnings about exceeding limits
3. **Balance Parameters**: Higher `--max-configs` requires lower `--pool-size`
4. **Test First**: Use small values to test your setup before scaling up

## Testing

Use the included test script to verify the functionality:

```bash
python test_region_mining.py
```

This script will:
- Show usage examples
- Test configuration generation for each region
- Display sample configurations
- Show unique parameter values per region

## Logging

When using `--region`, the system provides detailed logging:

```
INFO - Focusing on region: USA - will generate ALL possible combinations
INFO - Generating ALL possible combinations for region: USA
INFO - Available options for USA:
INFO -   Universes: ['TOP3000', 'TOP1000', 'TOP500', 'TOP200', 'ILLIQUID_MINVOL1M', 'TOPSP500']
INFO -   Delays: [1, 0]
INFO -   Neutralizations: ['NONE', 'REVERSION_AND_MOMENTUM', 'STATISTICAL', 'CROWDING', 'FAST', 'SLOW', 'MARKET', 'SECTOR', 'INDUSTRY', 'SUBINDUSTRY', 'SLOW_AND_FAST']
INFO -   Truncations: 14 values
INFO -   Decays: 513 values
INFO - Total possible combinations for USA: 1,845,360
INFO - Note: This may exceed max_configs limit of 1000
```

## Troubleshooting

### Common Issues

1. **Invalid Region**
   ```
   WARNING - Specified region 'INVALID' not found, using all available regions
   ```
   - Solution: Use one of the valid regions: USA, GLB, EUR, ASI, CHN

2. **Configuration Limit Reached**
   ```
   INFO - Reached max configs limit (1000)
   INFO - Generated 1000 configurations (out of 1,845,360 possible)
   ```
   - Solution: Increase `--max-configs` value

3. **Rate Limiting**
   ```
   INFO - Rate limit hit, waiting 60 seconds...
   ```
   - Solution: Wait for the specified time or reduce concurrent requests

### Best Practices

1. **Start Small**: Begin with smaller `--max-configs` values
2. **Monitor Logs**: Watch for warnings and errors
3. **Test Regions**: Test each region individually before large-scale mining
4. **Use Auto Mode**: Combine with `--auto-mode` for automated parameter selection
5. **Backup Results**: Save configurations to files for later use

## Migration from Old Version

The new functionality is backward compatible:
- Existing scripts will continue to work unchanged
- The `--region` parameter is optional
- Default behavior remains the same when `--region` is not specified

## Support

For issues or questions about the region-specific mining functionality:
1. Check the logs for detailed error messages
2. Verify region names are correct
3. Test with smaller configuration limits first
4. Review the API documentation for WorldQuant Brain limits
