# NWS77 Template for Consultant-Templates-Bruteforce-Ollama

## Overview

This implementation adds support for the nws77 sentiment analysis template based on real-world experience with the WorldQuant Brain platform. The template is designed to test sentiment-based alpha strategies using the nws77 dataset.

## Template Structure

The nws77 template uses a sophisticated structure with variable placeholders:

```
ts_decay_linear(<ts_Statistical_op/>(<dataprocess_op/>(-group_backfill(vec_max(<nws77/>),country,60, std = 4.0)), 90),5, dense = false)
```

### Variables

1. **`<nws77/>`**: EUR region TOP2500 universe, DataSet ID: nws77, DataFields Type: Vector
2. **`<dataprocess_op/>`**: Data processing functions (quantile, winsorize, normalize, rank, ts_rank)
3. **`<ts_Statistical_op/>`**: Time series statistical functions (ts_ir, ts_zscore, ts_entropy, ts_mean, ts_std_dev, ts_corr)

### Search Space

- **Total combinations**: 29 × 2 × 3 = 174 variations
- **Data fields**: 29 fields from nws77 dataset
- **Data processing options**: 2 functions
- **Time series statistical options**: 3 functions

## Data Characteristics

Based on analysis of nws77_sentiment_impact_projection field:

### Distribution
- Highly concentrated and negatively skewed distribution
- Prominent mode around -0.18
- Predominantly negative values (cautious/bearish sentiment)

### Sparsity & Coverage
- Extremely sparse with high prevalence of NaN values
- Strong upward trend in data coverage since 2013
- Clear quarterly seasonality (earnings reporting cycles)

### Time Series Nature
- Event-driven, not continuous time series
- Updates correspond to discrete events (earnings calls)
- Non-NaN and Non-Zero counts are nearly identical

### Group Behavior
- Group-level averages show strong co-movement
- Suggests common macro/market-wide influence on sentiment

## Implementation Details

### Signal Processing
- `vec_max`: Selects maximum element from vector
- `group_backfill`: Fills missing values to reduce unexpected position closures
- `dataprocess_op`: Adjusts data distribution and reduces extreme values
- `ts_Statistical_op`: Extracts temporal information from preprocessed data
- `ts_decay_linear`: Further smooths and enhances the signal

### Performance Notes
- **Turnover**: Generally high (consistently high over time, not extreme values)
- **Margin**: Generally low
- **Success Rate**: 1 out of 174 backtests produced a submittable Alpha
- **Signal Quality**: Remaining alphas show clear signals, proving template versatility

## Usage

### Basic Usage

```bash
# Run nws77 template testing
python run_nws77_template.py
```

### Advanced Usage

```bash
# Run with specific dataset targeting
python bruteforce_template_generator.py --credentials credential.json --custom-template nws77_template.json --target-dataset nws77
```

### Command Line Options

- `--credentials`: Path to credentials JSON file
- `--custom-template`: Path to nws77_template.json
- `--target-dataset`: Specific dataset to test (e.g., nws77)
- `--max-concurrent`: Maximum concurrent simulations (default: 8)
- `--resume`: Resume from previous progress

## Files

- `nws77_template.json`: Template definition with variables and metadata
- `run_nws77_template.py`: Dedicated runner script for nws77 testing
- `bruteforce_template_generator.py`: Enhanced with nws77 support
- `README_NWS77_TEMPLATE.md`: This documentation

## Template Variations

The system generates all possible combinations:

1. **Data Processing Functions**:
   - `quantile`
   - `winsorize`
   - `normalize`
   - `rank`
   - `ts_rank`

2. **Time Series Statistical Functions**:
   - `ts_ir`
   - `ts_zscore`
   - `ts_entropy`
   - `ts_mean`
   - `ts_std_dev`
   - `ts_corr`

3. **Data Fields**: 29 fields from nws77 dataset

## Alternative Decay Functions

The template can be extended with alternative decay functions:
- `jump_decay`
- `ts_decay_exp_window(x, d, factor = f)`

## Expected Results

Based on the original experience:
- **Total tests**: 174 combinations
- **Submittable alphas**: 1 (0.57% success rate)
- **Signal quality**: Most alphas show clear signals
- **Template versatility**: Proven effective on nws77 dataset

## Future Exploration Directions

1. **Turnover Analysis**: Address high turnover issues
2. **Margin Improvement**: Focus on low margin challenges
3. **Field-Specific Optimization**: Tailor information extraction for each data field
4. **Alternative Datasets**: Test template on other sentiment datasets

## Technical Notes

- The template uses variable substitution for flexibility
- Dataset-specific filtering ensures only relevant fields are tested
- Progress tracking allows resuming interrupted runs
- Results are saved in JSON format for analysis
- Logging provides detailed execution information

## Troubleshooting

1. **Credentials**: Ensure `credential.json` exists with valid WorldQuant Brain credentials
2. **Dataset Access**: Verify nws77 dataset is available in your region
3. **Field Availability**: Check that nws77 fields are accessible in EUR region
4. **Memory**: Large search spaces may require sufficient memory for concurrent execution

## Performance Optimization

- Use `--max-concurrent` to control parallel execution
- Enable `--resume` for long-running tests
- Monitor logs for progress and error information
- Save results regularly to avoid data loss
