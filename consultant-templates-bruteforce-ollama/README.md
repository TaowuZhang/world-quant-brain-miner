# Bruteforce Template Generator for WorldQuant Brain

A specialized template generator that creates **4 atom/light-weight templates at a time** using Ollama AI, then tests each template across ALL data fields and ALL regions using **2 subprocesses per template** for optimal distribution.

## Features

- **Batch Template Generation**: Uses Ollama to generate 4 atomic/light-weight templates per batch
- **Operator Integration**: Provides Ollama with all available operators and their descriptions
- **Dual Subprocess Distribution**: Each template is tested with 2 subprocesses for balanced workload
- **Bruteforce Testing**: Tests each template across ALL data fields and ALL regions
- **Success Criteria Filtering**: Automatically expands neutralization for templates meeting criteria (Sharpe > 1.25, Margin > 10bps, Fitness > 1.0)
- **Template Validation**: Validates templates before testing and regenerates if needed
- **Progress Persistence**: Saves progress periodically for continuation
- **Custom Template Support**: Can input your own template JSON for testing
- **Concurrent Execution**: Uses ThreadPoolExecutor for efficient parallel simulation
- **Comprehensive Coverage**: Tests across USA, EUR, CHN, GLB, and ASI regions
- **Multiple Neutralization Options**: Tests with different neutralization strategies

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Setup Ollama (see consultant-templates-ollama for detailed setup)

3. Create credentials file:
```json
{
    "username": "your_username",
    "password": "your_password"
}
```

## Usage

### Generate and Test Multiple Batches (4 templates per batch)
```bash
python bruteforce_template_generator.py --credentials credentials.json --max-batches 3
```

### Test Custom Template
```bash
python bruteforce_template_generator.py --credentials credentials.json --custom-template my_template.json
```

### Advanced Options
```bash
python bruteforce_template_generator.py \
    --credentials credentials.json \
    --ollama-model llama3.1 \
    --max-concurrent 8 \
    --max-batches 5
```

## Custom Template Format

Create a JSON file with your template:
```json
{
    "template": "rank(close, 20)"
}
```

## Output

The system generates:
- `bruteforce_results.json`: Complete results with all simulation data
- `bruteforce_template_generator.log`: Detailed execution log

## Thread Management

Uses similar thread management to consultant-templates-ollama:
- ThreadPoolExecutor for concurrent execution
- Future tracking and timeout handling
- Progress monitoring and health checks
- Automatic cleanup of stuck futures

## Regions and Data Fields

Tests across all available regions:
- USA, EUR, CHN, GLB, ASI

For each region, tests with all available data fields and neutralization options:
- INDUSTRY, SUBINDUSTRY, SECTOR, COUNTRY, NONE

## Example Output

```
🎯 Starting bruteforce test for 4 templates
📁 Loading cached data fields for USA delay=1
📊 Loaded 1250 cached fields for USA delay=1
📁 Loading cached data fields for EUR delay=1
📊 Loaded 980 cached fields for EUR delay=1
📊 Region USA: 1250 data fields
📊 Region EUR: 980 data fields
📊 Region CHN: 850 data fields
🚀 Created 4 template groups with 2 subprocesses each
🚀 Started subprocess 1 for template 1: rank(close, 20)...
🚀 Started subprocess 2 for template 1: rank(close, 20)...
🚀 Started subprocess 1 for template 2: sma(volume, 10)...
🚀 Started subprocess 2 for template 2: sma(volume, 10)...
...
🎮 MONITORING: Starting to monitor simulation progress (max 300s)
🎮 MONITORING: Check #1 (elapsed: 5.0s)
🎮 MONITORING: Simulation status: COMPLETE
✅ Subprocess completed with 1850 results
✅ Subprocess completed with 1850 results
🏁 Template_1_Subprocess_1: Completed 1850 tasks with 45 successes
🏁 Template_1_Subprocess_2: Completed 1850 tasks with 38 successes
...
🏆 Best result: Sharpe 1.456 in USA
📊 Tested 4 unique templates
🎯 Template: rank(close, 20)... - Best Sharpe: 1.456 in USA
```
