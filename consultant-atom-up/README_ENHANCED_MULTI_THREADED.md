# Enhanced Multi-Threaded Atom Tester

A sophisticated multi-threaded atom testing system that combines operator stacking with intelligent combination generation using Ollama AI.

## Features

### 🚀 Multi-Threading
- **8-thread management system** similar to consultant-templates-ollama
- Parallel processing for maximum efficiency
- Configurable worker count

### 🧠 Intelligent Operator Combinations
- **Ollama AI integration** for smart operator sequencing
- **Operator stacking levels**: 0, 1, 2, and 3 operators
- Fallback combinations when Ollama is unavailable
- Categories: momentum, mean reversion, volatility, correlation, trend

### 🌍 Region-Based Testing
- **Sequential region processing**: ASI → CHN → EUR → GLB → USA
- Comprehensive data field coverage
- Region-specific optimization

### 💾 Progress Management
- **Automatic progress saving** every 10 tests
- **Resume functionality** - pick up where you left off
- Detailed progress tracking with completion statistics

### 🔍 Quality Assurance
- **Too good to be true PnL checks** - marks unrealistic results as RED
- **Submission validation** with color coding:
  - 🟢 **GREEN**: All checks PASS
  - 🟡 **YELLOW**: Some WARNINGs present
  - 🔴 **RED**: FAIL conditions detected
- **PROD_CORRELATION tracking** in result names

### 📊 Comprehensive Results
- **Detailed JSON output** with all metrics
- **PnL data preservation** for each data field
- **Sharpe ratio, fitness, returns, turnover** tracking
- **Execution time and error logging**

## Usage

### Basic Usage
```bash
# Run with default 8 workers
python run_enhanced_multi_threaded_atom_tester.py

# Run with 4 workers
python run_enhanced_multi_threaded_atom_tester.py --workers 4

# Resume from previous progress
python run_enhanced_multi_threaded_atom_tester.py --resume

# Test specific region
python run_enhanced_multi_threaded_atom_tester.py --region ASI
```

### Advanced Usage
```bash
# Resume with custom worker count
python run_enhanced_multi_threaded_atom_tester.py --workers 8 --resume

# Test only CHN region with 6 workers
python run_enhanced_multi_threaded_atom_tester.py --region CHN --workers 6
```

## Requirements

### Required Files
- `credential.txt` - Your WorldQuant Brain credentials
- `operatorRAW.json` - Operator definitions
- `data_fields_cache_*.json` - Cached data fields for each region

### Optional Dependencies
- **Ollama** (recommended) - For intelligent operator combination generation
  - Install: https://ollama.ai/
  - Model: `ollama pull llama3.1`

## Output Files

### Results
- `enhanced_atom_results.json` - Complete test results with all metrics
- `atom_test_progress.json` - Progress state for resuming
- `enhanced_multi_threaded_atom_tester.log` - Detailed execution log

### Result Structure
```json
{
  "atom_id": "alpha_id",
  "expression": "generated_expression",
  "data_field_id": "field_id",
  "data_field_name": "field_description",
  "dataset_id": "dataset_id",
  "dataset_name": "dataset_name",
  "region": "ASI",
  "universe": "TOP3000",
  "operator_combination": {
    "combination_id": "stack_2_5",
    "operators": ["ts_rank", "delta"],
    "stack_level": 2,
    "description": "Ranks time series and calculates momentum",
    "category": "momentum",
    "complexity": "simple"
  },
  "status": "success",
  "sharpe_ratio": 1.25,
  "fitness": 0.85,
  "returns": 0.12,
  "color_status": "GREEN",
  "prod_correlation": 0.45,
  "submission_checks": {...},
  "pnl_data": {...}
}
```

## Operator Combination Examples

### Stack Level 0 (Raw Data)
- No operators applied - uses data field directly

### Stack Level 1 (Single Operator)
- `ts_rank(data_field, 20)` - Time series ranking
- `delta(data_field, 1)` - First difference
- `abs(data_field)` - Absolute value

### Stack Level 2 (Two Operators)
- `ts_rank(delta(data_field, 1), 20)` - Ranked momentum
- `abs(ts_mean(data_field, 20))` - Absolute moving average
- `delta(ts_rank(data_field, 20), 1)` - Ranked difference

### Stack Level 3 (Three Operators)
- `abs(delta(ts_rank(data_field, 20), 1))` - Complex momentum signal
- `ts_rank(abs(delta(data_field, 1)), 20)` - Ranked absolute momentum

## Color Status System

### 🟢 GREEN
- All submission checks PASS
- No warnings or failures
- Production-ready alphas

### 🟡 YELLOW
- Some WARNING conditions present
- May need attention before production
- Still potentially valuable

### 🔴 RED
- FAIL conditions detected
- Too good to be true results
- Not suitable for production

## Performance Monitoring

The system provides real-time monitoring:
```
✅ ASI | Price to Book Ratio... | stack_2_5 | Sharpe: 1.250 | Color: GREEN
❌ CHN | Market Cap... | stack_1_3 | Sharpe: 0.000 | Color: RED
```

## Error Handling

- **Graceful degradation** when Ollama is unavailable
- **Automatic retry** for transient failures
- **Progress preservation** on interruption
- **Detailed error logging** for debugging

## Best Practices

1. **Start with ASI region** for initial testing
2. **Use 8 workers** for optimal performance
3. **Enable progress saving** for long-running tests
4. **Monitor color status** for quality assessment
5. **Review logs** for error patterns

## Troubleshooting

### Common Issues
- **Ollama not available**: System falls back to predefined combinations
- **Network timeouts**: Automatic retry with exponential backoff
- **Memory issues**: Reduce worker count
- **Progress corruption**: Delete progress file to start fresh

### Performance Tips
- Use SSD storage for faster I/O
- Ensure stable internet connection
- Monitor system resources during execution
- Consider running during off-peak hours
