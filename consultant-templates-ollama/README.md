# Enhanced Template Generator for WorldQuant Brain Alpha Expressions

This tool uses the DeepSeek API to generate comprehensive alpha expression templates and **multi-simulates** them using the powerhouse approach, combining operators and data fields from different regions.

## Features

### Core Features
- **Region-aware**: Generates templates specific to different regions (USA, GLB, EUR, ASI, CHN)
- **Data-driven**: Uses actual data fields available for each region
- **Operator-rich**: Incorporates all available WorldQuant Brain operators
- **AI-powered**: Uses DeepSeek API for intelligent template generation
- **Multi-simulation**: Tests all generated templates using WorldQuant Brain's simulation API
- **Performance Analysis**: Provides detailed performance metrics and rankings
- **Comprehensive**: Traverses through all data field combinations with real simulation results

### Enhanced v2 Features (Recommended)
- **Progress Saving**: Automatically saves progress during execution
- **Resume Functionality**: Resume from where you left off if interrupted
- **Dynamic Progress Display**: Real-time progress updates with statistics
- **Template Validation**: Improved template quality with syntax and field validation
- **Error Handling**: Better error handling and recovery
- **Best Template Tracking**: Tracks and displays the best performing template in real-time

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up credentials**:
   - Create a `credential.txt` file with your WorldQuant Brain credentials:
     ```json
     ["your-username", "your-password"]
     ```

3. **Set DeepSeek API key**:
   - Create a `.env` file with your DeepSeek API key:
     ```
     DEEPSEEK_API_KEY=your-deepseek-api-key
     ```
   - Or set as environment variable:
     ```bash
     export DEEPSEEK_API_KEY='your-deepseek-api-key'
     ```

## Usage

### Quick Start (Basic Template Generation)
```bash
python run_generator.py
```

### Enhanced Multi-Simulation Testing
```bash
# Generate templates AND test them with multi-simulation
python run_enhanced_generator.py

# Enhanced v2 with progress saving and resume functionality
python run_enhanced_generator_v2.py
```

### Advanced Usage
```bash
# Basic template generation
python template_generator.py --credentials credential.txt --ollama-model llama3.1

# Enhanced v2 with progress saving and resume (Recommended)
python run_enhanced_generator_v2.py

# Generate templates for specific regions with testing
python run_enhanced_generator_v2.py --region USA

# Custom number of templates per region
python run_enhanced_generator_v2.py --templates 15

# Resume from previous progress
python run_enhanced_generator_v2.py --resume
```

## Output Format

### Basic Template Generation (`generatedTemplate.json`)
```json
{
  "metadata": {
    "generated_at": "2025-01-05 10:30:00",
    "total_operators": 79,
    "regions": ["USA", "GLB", "EUR", "ASI", "CHN"]
  },
  "templates": {
    "USA": [
      {
        "region": "USA",
        "template": "ts_rank(ts_delta(close, 1), 20)",
        "operators_used": ["ts_rank", "ts_delta"],
        "fields_used": ["close"]
      }
    ]
  }
}
```

### Enhanced Multi-Simulation Results (`enhanced_generatedTemplate.json`)
```json
{
  "metadata": {
    "generated_at": "2025-01-05 10:30:00",
    "total_operators": 79,
    "regions": ["USA", "GLB", "EUR", "ASI", "CHN"],
    "templates_per_region": 10
  },
  "templates": { /* Same as basic format */ },
  "simulation_results": {
    "USA": [
      {
        "template": "ts_rank(ts_delta(close, 1), 20)",
        "region": "USA",
        "sharpe": 1.25,
        "fitness": 0.85,
        "turnover": 0.12,
        "returns": 0.08,
        "drawdown": -0.05,
        "margin": 0.15,
        "longCount": 150,
        "shortCount": 100,
        "success": true,
        "error_message": "",
        "timestamp": 1704456000.0
      }
    ]
  },
  "analysis": {
    "total_templates": 50,
    "successful_simulations": 45,
    "failed_simulations": 5,
    "success_rate": 0.9,
    "performance_metrics": {
      "sharpe": {"mean": 1.15, "std": 0.25, "min": 0.8, "max": 1.8},
      "fitness": {"mean": 0.82, "std": 0.12, "min": 0.6, "max": 0.95},
      "turnover": {"mean": 0.15, "std": 0.08, "min": 0.05, "max": 0.35}
    }
  }
}
```

## Region Configurations

- **USA**: TOP3000 universe, delay=1
- **GLB**: TOP3000 universe, delay=1  
- **EUR**: TOP2500 universe, delay=1
- **ASI**: MINVOL1M universe, delay=1, maxTrade=True
- **CHN**: TOP2000U universe, delay=1, maxTrade=True

## Template Generation Process

### Basic Process
1. **Data Collection**: Fetches available datasets and data fields for each region
2. **Operator Selection**: Randomly selects relevant operators from the full operator set
3. **Field Selection**: Randomly selects data fields available in the region
4. **AI Generation**: Uses DeepSeek API to generate creative alpha expression templates
5. **Validation**: Extracts and validates operators and fields used in each template
6. **Output**: Saves structured templates with metadata

### Enhanced Multi-Simulation Process
1. **Template Generation**: Same as basic process
2. **Multi-Simulation Setup**: Groups templates into pools for efficient processing
3. **Parallel Submission**: Submits multiple simulations simultaneously to WorldQuant Brain API
4. **Progress Monitoring**: Tracks simulation progress across all submitted templates
5. **Result Collection**: Gathers performance metrics (Sharpe, fitness, turnover, etc.)
6. **Performance Analysis**: Analyzes results and ranks templates by performance
7. **Comprehensive Output**: Saves both templates and simulation results with analysis

## Files

### Basic Template Generation
- `template_generator.py`: Main generator class
- `operatorRAW.json`: Available operators database
- `templateRAW.txt`: Raw template examples
- `generatedTemplate.json`: Generated output (created after running)

### Enhanced Multi-Simulation v2 (Recommended)
- `enhanced_template_generator_v2.py`: Enhanced generator v2 with progress saving and resume
- `run_enhanced_generator_v2.py`: Enhanced runner script v2
- `enhanced_results_v2.json`: Enhanced output with simulation results (created after running)
- `template_progress_v2.json`: Progress file for resume functionality (created during running)

### Utilities
- `test_setup.py`: Test script to verify setup
- `example_usage.py`: Examples showing how to use generated templates
- `requirements.txt`: Python dependencies

## Logging

The generator creates detailed logs in `template_generator.log` including:
- API call attempts and responses
- Data field fetching progress
- Template generation statistics
- Error handling and retries

## Error Handling

- Automatic retry for API failures
- Graceful handling of missing data fields
- Region-specific error recovery
- Comprehensive logging for debugging

## Example Templates

The generator creates diverse templates like:
- Time series operations: `ts_rank(ts_delta(volume, 1), 60)`
- Group operations: `group_normalize(ts_zscore(returns, 120), industry)`
- Complex combinations: `winsorize(ts_regression(close, volume, 250), std=3)`
- Arithmetic operations: `add(ts_mean(high, 20), ts_mean(low, 20))`

## Troubleshooting

1. **Authentication errors**: Check your WorldQuant Brain credentials
2. **API key errors**: Verify your DeepSeek API key is valid
3. **No data fields**: Some regions may have limited data availability
4. **Rate limiting**: The generator includes delays between API calls

## Contributing

Feel free to extend the generator with:
- Additional template patterns
- More sophisticated operator combinations
- Custom region configurations
- Enhanced validation logic
