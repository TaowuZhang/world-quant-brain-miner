# Alpha ICU - WorldQuant Brain Alpha Analysis System

Alpha ICU is a comprehensive system for fetching, analyzing, and evaluating alphas from the WorldQuant Brain API. It automatically identifies successful alphas based on performance metrics and checks their correlations with production alphas.

## Features

- **Alpha Fetching**: Automatically fetches alphas from WorldQuant Brain API with authentication
- **Success Analysis**: Filters alphas based on configurable success criteria (Sharpe ratio, fitness, returns, etc.)
- **Correlation Checking**: Analyzes correlations with production alphas to identify potential issues
- **Comprehensive Reporting**: Generates detailed reports with recommendations
- **Top Performers**: Identifies and ranks the best performing alphas
- **Flexible Configuration**: Customizable thresholds and criteria

## Installation

1. Clone or download the alpha-icu directory
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your credentials:
   - Copy `credential.txt` and update it with your WorldQuant Brain credentials:
   ```
   [your.email@worldquant.com, your_password]
   ```

## Usage

### Basic Usage

Run the full analysis with default settings:
```bash
python main.py
```

### Advanced Usage

```bash
# Analyze alphas from the last 7 days
python main.py --days 7

# Limit to 50 alphas for testing
python main.py --max-alphas 50

# Skip correlation checks for faster analysis
python main.py --no-correlations

# Don't save results to files
python main.py --no-save

# Show top 20 performers sorted by fitness
python main.py --top-n 20 --sort-by fitness

# Use custom credential file
python main.py --credential-file my_credentials.txt
```

### Command Line Options

- `--days`: Number of days back to fetch alphas (default: 3)
- `--max-alphas`: Maximum number of alphas to process
- `--status`: Status filter for alphas (default: "UNSUBMITTED,IS_FAIL")
- `--no-correlations`: Skip correlation checks
- `--no-save`: Don't save results to files
- `--top-n`: Number of top performers to show (default: 10)
- `--sort-by`: Sort metric for top performers (sharpe, fitness, returns, pnl)
- `--credential-file`: Path to credential file (default: credential.txt)

## Success Criteria

The system evaluates alphas based on the following criteria:

- **Sharpe Ratio**: Minimum 1.5
- **Fitness**: Minimum 1.0
- **Returns**: Minimum 0.05 (5%)
- **Drawdown**: Maximum 0.5 (50%)
- **Turnover**: Between 0.01 and 0.7
- **Check Failures**: Less than 30% of total checks

## Correlation Analysis

The correlation checker analyzes each successful alpha against production alphas:

- **High Correlation**: > 0.5 (potential issue)
- **Medium Correlation**: 0.2 to 0.5 (monitor)
- **Low Correlation**: -0.2 to 0.2 (acceptable)
- **Negative Correlation**: < -0.2 (good diversification)

Risk levels are assigned based on correlation patterns:
- **LOW**: Minimal correlation concerns
- **MEDIUM**: Some correlation issues present
- **HIGH**: Significant correlation risks

## Output Files

The system generates several output files with timestamps:

- `alpha_icu_results_YYYYMMDD_HHMMSS.json`: Complete analysis results
- `successful_alphas_YYYYMMDD_HHMMSS.json`: Summary of successful alphas
- `correlation_report_YYYYMMDD_HHMMSS.json`: Detailed correlation analysis
- `alpha_icu.log`: System log file

## Example Output

```
============================================================
ALPHA ICU ANALYSIS SUMMARY
============================================================
Total alphas processed: 1179
Successful alphas: 45
Success rate: 3.8%

Top 10 performers (sorted by sharpe):
 1. NZ9g967 - Sharpe: 2.35, Fitness: 1.15, Returns: 0.095
 2. EdJrbw1 - Sharpe: 2.43, Fitness: 1.19, Returns: 0.095
 3. AAKYPWl - Sharpe: 2.23, Fitness: 0.89, Returns: 0.069
 ...

Correlation Analysis:
  Risk distribution: {'LOW': 32, 'MEDIUM': 10, 'HIGH': 3}
  High risk alphas: 3
  Medium risk alphas: 10

Analysis completed successfully!
```

## API Endpoints Used

The system interacts with the following WorldQuant Brain API endpoints:

- `POST /auth/login`: Authentication
- `GET /users/self/alphas`: Fetch user's alphas
- `GET /alphas/{alpha_id}/correlations/prod`: Get correlation data

## Configuration

You can modify the success criteria by editing the `AlphaAnalyzer` initialization in `main.py`:

```python
self.analyzer = AlphaAnalyzer(
    min_sharpe=1.5,        # Minimum Sharpe ratio
    min_fitness=1.0,       # Minimum fitness score
    max_drawdown=0.5,      # Maximum drawdown
    min_turnover=0.01,     # Minimum turnover
    max_turnover=0.7,      # Maximum turnover
    min_returns=0.05       # Minimum returns
)
```

## Error Handling

The system includes comprehensive error handling:

- Authentication failures are logged and reported
- API rate limiting is handled with delays
- Individual alpha processing errors don't stop the entire analysis
- Correlation check failures are logged but don't affect other alphas

## Logging

All operations are logged to both console and `alpha_icu.log` file. Log levels include:

- INFO: Normal operations and progress
- WARNING: Non-critical issues (e.g., failed correlation checks)
- ERROR: Critical failures that stop processing

## Contributing

To extend the system:

1. **Add new success criteria**: Modify `AlphaAnalyzer.is_successful_alpha()`
2. **Add new correlation metrics**: Extend `CorrelationChecker`
3. **Add new output formats**: Modify `AlphaICU._save_results()`
4. **Add new API endpoints**: Extend `AlphaFetcher`

## Troubleshooting

### Common Issues

1. **Authentication Error**: Check your credentials in `credential.txt`
2. **No Alphas Found**: Verify date range and status filter
3. **API Rate Limiting**: The system includes delays, but you may need to reduce `max_alphas`
4. **Correlation Check Failures**: Some alphas may not have correlation data available

### Debug Mode

For detailed debugging, modify the logging level in `main.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

## License

This project is part of the WorldQuant Miner suite. Please refer to the main project license.

## Support

For issues or questions, please check the logs first and ensure your credentials are correct. The system provides detailed error messages to help diagnose problems.
