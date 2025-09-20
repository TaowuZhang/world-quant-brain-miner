# Atom-Based Alpha Testing System

A comprehensive system for testing **Atom Alphas** on WorldQuant Brain, designed to understand the performance characteristics of single-dataset alphas across different regions and configurations.

## What is an Atom Alpha?

An **Atom** is the simplest type of Alpha in WorldQuant Brain:
- It is a signal built using **only one dataset**, without combining multiple datasets
- Examples: `rank(returns)`, `abs(volume)`, `log(price)`
- **NOT an Atom**: `ts_co_skewness(news_session_range_pct, returns, 20)` (uses two datasets!)

### Why are Atom Alphas Important?

1. **Clean and Specialized Behavior**: Atoms provide focused signals from single data sources
2. **Easier to Analyze**: Simple structure makes it easier to understand which datasets drive performance
3. **Better Focus**: Mastering Atoms helps fine-tune understanding of data behavior
4. **Foundation for Higher Achievements**: Critical skill for Master and Grandmaster levels

## Features

### 🧪 Comprehensive Testing
- Tests atoms across all cached data fields from multiple regions (USA, EUR, ASI, CHN, GLB)
- Multiple universe configurations (TOP3000, TOP2000, TOP1000)
- Various neutralization options (INDUSTRY, SUBINDUSTRY, SECTOR, COUNTRY, NONE)
- Different delay settings (1, 2 days)

### 📊 Statistical Analysis
- **Performance Metrics**: Sharpe ratio, returns, max drawdown, hit ratio
- **PnL Tracking**: Complete profit/loss data for each successful atom
- **Dataset Analysis**: Performance comparison across different datasets
- **Success Rate Tracking**: Monitor which datasets and operators work best

### 🔧 Smart Operator Selection
- Automatically selects operators suitable for single-input atoms
- Includes: rank, abs, log, sqrt, sign, max, min, mean, std, sum, prod
- Excludes complex operators requiring multiple datasets

### 💾 Data Storage
- **Detailed Results**: Complete test results with timestamps and execution times
- **Statistical Summary**: Aggregated performance metrics by dataset
- **Resume Capability**: Can resume testing from previous results
- **JSON Format**: Easy to analyze and integrate with other tools

## Installation

1. **Clone and Setup**:
   ```bash
   cd consultant-atom-up
   pip install -r requirements.txt
   ```

2. **Copy Required Files**:
   ```bash
   # Copy operators from consultant-templates-api
   copy ..\consultant-templates-api\operatorRAW.json .
   
   # Ensure you have cached data fields files:
   # data_fields_cache_USA_1.json
   # data_fields_cache_EUR_1.json
   # data_fields_cache_ASI_1.json
   # data_fields_cache_CHN_1.json
   # data_fields_cache_GLB_1.json
   ```

3. **Setup Credentials**:
   Create `credential.txt` with your WorldQuant Brain credentials:
   
   **Format 1 (JSON):**
   ```
   ["your_username", "your_password"]
   ```
   
   **Format 2 (Two-line):**
   ```
   your_username
   your_password
   ```

## Usage

### Quick Start
```bash
python run_atom_tests.py
```

### Advanced Usage
```bash
# Run with custom parameters
python atom_tester.py --max-tests 200 --max-workers 4

# Run with custom credential file
python atom_tester.py --credential-file my_credentials.txt
```

### Parameters
- `--max-tests`: Maximum number of atom tests to run (default: 100)
- `--max-workers`: Number of concurrent workers (default: 4)
- `--credential-file`: Path to credential file (default: credential.txt)

## Output Files

### 📄 atom_test_results.json
Detailed results for each atom test:
```json
{
  "atom_id": "PV7Lxb7",
  "expression": "rank(returns)",
  "dataset_id": "analyst14",
  "dataset_name": "Estimations of Key Fundamentals",
  "region": "USA",
  "universe": "TOP3000",
  "delay": 1,
  "neutralization": "INDUSTRY",
  "status": "success",
  "sharpe_ratio": 1.234,
  "returns": 0.156,
  "max_drawdown": -0.089,
  "hit_ratio": 0.523,
  "pnl_data": {...},
  "test_timestamp": "2024-01-15T10:30:00",
  "execution_time": 12.5
}
```

### 📊 atom_statistics.json
Statistical summary by dataset:
```json
{
  "analyst14": {
    "dataset_id": "analyst14",
    "dataset_name": "Estimations of Key Fundamentals",
    "total_tests": 45,
    "successful_tests": 38,
    "failed_tests": 7,
    "avg_sharpe": 0.892,
    "max_sharpe": 2.145,
    "min_sharpe": -0.234,
    "avg_returns": 0.123,
    "avg_max_drawdown": -0.067,
    "avg_hit_ratio": 0.512,
    "best_atom": "rank(anl14_actvalue_bvps_fp0)",
    "worst_atom": "abs(anl14_actvalue_bvps_fp0)",
    "success_rate": 0.844
  }
}
```

### 📝 atom_tester.log
Execution log with detailed progress and error information.

## Analysis Scripts

### analyze_atom_results.py
```bash
python analyze_atom_results.py
```
Provides detailed analysis of test results including:
- Top performing atoms by region
- Dataset performance comparison
- Operator effectiveness analysis
- Risk-return characteristics

## Example Results

### Top Performing Atoms
```
🏆 Top 5 Atom Performers:
  1. rank(anl14_actvalue_bvps_fp0)
     Dataset: Estimations of Key Fundamentals
     Sharpe: 2.145, Returns: 0.234

  2. log(fundamental13_pe_ratio)
     Dataset: Comprehensive Fundamentals Dataset
     Sharpe: 1.892, Returns: 0.189

  3. abs(model55_technical_indicator_1)
     Dataset: Fundamental Indicators
     Sharpe: 1.756, Returns: 0.167
```

### Dataset Performance Summary
```
📈 Dataset Performance Summary:
  analyst14: Estimations of Key Fundamentals
    Tests: 45, Success Rate: 84.4%
    Avg Sharpe: 0.892, Max Sharpe: 2.145

  fundamental13: Comprehensive Fundamentals Dataset
    Tests: 38, Success Rate: 78.9%
    Avg Sharpe: 0.756, Max Sharpe: 1.892
```

## Key Insights

### Dataset Categories
- **Analyst Data**: High success rate, good for ranking operations
- **Fundamental Data**: Consistent performance, suitable for mathematical operations
- **Model Data**: Variable performance, requires careful operator selection
- **News Data**: Lower success rate, but high potential when successful

### Operator Effectiveness
- **Rank**: Most effective for analyst and fundamental data
- **Log**: Good for price-related fields with wide ranges
- **Abs**: Useful for volatility and momentum indicators
- **Mathematical**: Work well with fundamental ratios and technical indicators

### Regional Differences
- **USA**: Highest data coverage, most consistent results
- **EUR**: Good performance with analyst data
- **ASI/CHN**: Specialized datasets, unique opportunities
- **GLB**: Broad coverage but variable quality

## Best Practices

1. **Start Small**: Begin with 50-100 tests to understand system behavior
2. **Monitor Resources**: Use 2-4 workers to avoid API rate limits
3. **Regular Analysis**: Review results frequently to identify patterns
4. **Dataset Focus**: Concentrate on high-performing datasets first
5. **Operator Selection**: Use simple operators for initial exploration

## Troubleshooting

### Common Issues
- **API Rate Limits**: Reduce max_workers or add delays
- **Memory Issues**: Process results in smaller batches
- **Credential Errors**: Verify credential.txt format
- **Missing Files**: Ensure all required files are present

### Performance Tips
- Use SSD storage for faster file I/O
- Monitor system resources during testing
- Save results frequently to avoid data loss
- Use appropriate max_tests based on your system capacity

## Contributing

This system is designed to be extensible. Key areas for enhancement:
- Additional operator types
- More sophisticated atom generation
- Advanced statistical analysis
- Integration with other WorldQuant tools
- Real-time monitoring and alerting

## License

Part of the WorldQuant Miner project. See main project LICENSE for details.
