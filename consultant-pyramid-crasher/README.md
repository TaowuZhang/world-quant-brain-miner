# Consultant Pyramid Crasher

A specialized system for cracking financial pyramids using 3 concurrent simulation strategies, designed to work alongside the consultant-templates-api.

## Overview

The Pyramid Crasher implements three specialized strategies for breaking through financial pyramid structures:

1. **Aggregate Breaker** - Breaks through aggregation layers using multi-level ranking and scaling
2. **Correlation Hunter** - Finds hidden correlations using time-series correlation analysis
3. **Volatility Exploiter** - Exploits volatility patterns using standard deviation and delta analysis

## Features

- **3 Concurrent Strategies**: Runs 3 specialized pyramid-cracking algorithms simultaneously
- **Pyramid Multiplier Integration**: Uses region-specific pyramid multipliers for optimal targeting
- **Breakthrough Detection**: Identifies and tracks breakthrough simulations with enhanced scoring
- **Template Integration**: Works alongside consultant-templates-api for comprehensive analysis
- **Real-time Progress Tracking**: Monitors progress with detailed statistics and breakthrough counts

## Architecture

### Core Components

- `pyramid_crasher.py` - Main pyramid cracking engine with 3 concurrent strategies
- `run_pyramid_crasher.py` - Command-line interface for running pyramid cracking
- `integrated_orchestrator.py` - Coordinates pyramid cracking with template generation

### Strategy Details

#### 1. Aggregate Breaker
- **Purpose**: Break through aggregation layers in financial data
- **Method**: Multi-layer ranking and scaling operations
- **Template Pattern**: `rank(ts_rank(scale(ts_mean(field1, 20)), 5)) - ts_rank(scale(ts_mean(field2, 10)), 3)`

#### 2. Correlation Hunter
- **Purpose**: Find hidden correlations between financial instruments
- **Method**: Time-series correlation analysis with ranking
- **Template Pattern**: `ts_corr(field1, field2, 20) * rank(ts_rank(scale(field1), 5))`

#### 3. Volatility Exploiter
- **Purpose**: Exploit volatility patterns for profit
- **Method**: Standard deviation and delta analysis
- **Template Pattern**: `rank(ts_std(field1, 20)) - rank(ts_std(field2, 10))`

## Installation

1. **Prerequisites**:
   ```bash
   pip install requests numpy
   ```

2. **Credentials Setup**:
   Create a `credential.txt` file with your WorldQuant Brain credentials:
   ```json
   ["username", "password"]
   ```

## Usage

### Basic Pyramid Cracking

```bash
# Run 3 concurrent pyramid cracking simulations
python run_pyramid_crasher.py --iterations 100

# Run with specific regions
python run_pyramid_crasher.py --regions USA EUR --iterations 50

# Resume from previous progress
python run_pyramid_crasher.py --resume --iterations 200
```

### Integrated Orchestration

```bash
# Run integrated simulation with template generation
python integrated_orchestrator.py --deepseek-key YOUR_API_KEY --iterations 100
```

### Advanced Options

```bash
# Custom breakthrough threshold
python run_pyramid_crasher.py --breakthrough-threshold 2.5 --iterations 100

# Maximum concurrent simulations
python run_pyramid_crasher.py --max-concurrent 3 --iterations 100

# Specific output files
python run_pyramid_crasher.py --output custom_results.json --progress-file custom_progress.json
```

## Configuration

### Region Settings

The system supports 5 regions with different pyramid multipliers:

- **USA**: Multiplier 1.8 (delay=0), 1.2 (delay=1)
- **GLB**: Multiplier 1.0 (delay=0), 1.5 (delay=1) 
- **EUR**: Multiplier 1.7 (delay=0), 1.4 (delay=1)
- **ASI**: Multiplier 1.0 (delay=0), 1.5 (delay=1)
- **CHN**: Multiplier 1.0 (delay=0), 1.8 (delay=1)

### Breakthrough Scoring

Breakthrough scores are calculated using:
- **Sharpe Ratio** (50% weight)
- **Fitness Score** (30% weight)  
- **Turnover Efficiency** (20% weight)

Default breakthrough threshold: 2.0

## Output Files

### Results File (`pyramid_results.json`)
```json
{
  "metadata": {
    "generated_at": "2024-01-01 12:00:00",
    "strategies": ["aggregate_breaker", "correlation_hunter", "volatility_exploiter"],
    "max_concurrent": 3,
    "version": "1.0"
  },
  "pyramid_results": {
    "USA": [...],
    "EUR": [...]
  },
  "breakthrough_analysis": {
    "total_simulations": 100,
    "successful_simulations": 85,
    "breakthrough_simulations": 12,
    "strategy_performance": {...},
    "region_performance": {...},
    "best_breakthroughs": [...]
  }
}
```

### Progress File (`pyramid_progress.json`)
```json
{
  "completed_count": 100,
  "successful_count": 85,
  "failed_count": 15,
  "breakthrough_count": 12,
  "best_breakthrough": 3.45,
  "timestamp": 1704067200.0
}
```

## Integration with Templates API

The Pyramid Crasher is designed to work alongside the consultant-templates-api:

1. **Concurrent Execution**: Runs 3 pyramid strategies while templates API runs 5 template generation slots
2. **Shared Resources**: Uses same credentials and region configurations
3. **Unified Results**: Integrated orchestrator combines results from both systems
4. **Cross-Pollination**: Successful pyramid templates can inform template generation

## Monitoring and Logging

- **Real-time Progress**: Shows completion count, success rate, and breakthrough count
- **Detailed Logging**: All operations logged to `pyramid_crasher.log`
- **Breakthrough Alerts**: Special notifications for breakthrough simulations
- **Performance Metrics**: Strategy and region performance analysis

## Example Output

```
🚀 Starting Pyramid Crasher - 3 Concurrent Simulations...
🎯 Strategies: Aggregate Breaker, Correlation Hunter, Volatility Exploiter
🌍 Regions: ['USA', 'GLB', 'EUR', 'ASI', 'CHN']
🔄 Iterations: 100
================================================================================

🔄 Progress: 50/100 completed | ✅ 42 successful | ❌ 8 failed | 📈 84.0% success rate | 💥 5 breakthroughs | 🏆 Best: 2.34

💥 BREAKTHROUGH! aggregate_breaker: Sharpe=2.45, Score=2.67
💥 BREAKTHROUGH! correlation_hunter: Sharpe=2.12, Score=2.34

================================================================================
🎉 PYRAMID CRACKING COMPLETE!
================================================================================
📊 Final Statistics:
   Total concurrent simulations: 100
   Successful simulations: 85
   Failed simulations: 15
   Breakthrough simulations: 12
   Success rate: 85.0%
   Breakthrough rate: 14.1%
   Best breakthrough score: 3.45
   Results saved to: pyramid_results.json
   Progress saved to: pyramid_progress.json
   Strategies used: ['aggregate_breaker', 'correlation_hunter', 'volatility_exploiter']
   Max Concurrent: 3
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Ensure `credential.txt` is properly formatted
2. **Network Issues**: Check internet connection and WorldQuant Brain access
3. **Memory Usage**: Large datasets may require increased memory allocation
4. **Concurrent Limits**: WorldQuant Brain has limits on concurrent simulations

### Performance Tips

1. **Optimal Regions**: Focus on USA and EUR for highest pyramid multipliers
2. **Breakthrough Threshold**: Adjust threshold based on your risk tolerance
3. **Iteration Count**: Start with 50-100 iterations for testing
4. **Resume Capability**: Use `--resume` to continue from previous progress

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `pyramid_crasher.log`
3. Open an issue with detailed error information
4. Include your configuration and error messages
