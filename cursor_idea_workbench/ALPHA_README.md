# Alpha Expression Analyzer & Recommender

A comprehensive tool for analyzing WorldQuant alpha expressions and suggesting improvements to reduce turnover, improve robustness, and optimize performance.

## Features

- **Expression Parsing**: Automatically parses alpha expressions and identifies operators and fields
- **Complexity Analysis**: Calculates complexity scores and identifies potential issues
- **Flexible Improvement Requests**: Provides targeted suggestions based on custom improvement requests
- **Vector Database Integration**: Finds related fields and suggests combinations using Pinecone
- **Multiple Expression Support**: Analyze and combine multiple expressions separated by semicolons
- **Conditional Expressions**: Support for `if_else` operators and complex logic
- **Smart Recommendations**: 
  - **Turnover Reduction**: Add `hump`, `ts_decay_linear`, `ts_delay`
  - **Robustness Improvement**: Add `winsorize`, `ts_zscore`, `ts_rank`
  - **Performance Optimization**: Break down complex expressions
  - **Conditional Logic**: Add `if_else`, `trade_when` for filtering
  - **Risk Management**: Add `scale`, `group_neutralize` for position control
  - **Field Combinations**: Suggest related fields from vector database
- **Interactive CLI**: User-friendly command-line interface
- **Batch Processing**: Analyze multiple expressions at once
- **JSON Output**: Machine-readable output for integration

## Quick Start

### Installation

No additional dependencies required beyond standard Python libraries:

```bash
# The tool uses only standard libraries
python alpha_cli.py --help
```

### Flexible Improvement Requests

The system now supports custom improvement requests instead of prescriptive types. You can ask for any improvement you want:

**Examples:**
- `"Reduce turnover by adding conditions"`
- `"Make it more robust against outliers"`
- `"Add time decay for news signals"`
- `"Improve performance and efficiency"`
- `"Add risk management controls"`
- `"Add filtering conditions"`
- `"Make it more stable over time"`
- `"Add sector neutralization"`
- `"Reduce noise in the signal"`

### Basic Usage

#### Interactive Mode
```bash
python alpha_cli.py --interactive
```

#### Single Expression Analysis
```bash
python alpha_cli.py --expression "-abs(subtract(news_max_up_ret, news_max_dn_ret))" --improvement-request "Reduce turnover by adding conditions"
```

#### Multiple Expression Analysis
```bash
python alpha_cli.py --expression "ts_mean(close, 20); rank(volume); subtract(high, low)" --improvement-request "Make more robust"
```

#### Conditional Expression Analysis
```bash
python alpha_cli.py --expression "if_else(greater(close, ts_mean(close, 20)), 1, -1); ts_std_dev(volume, 10)" --improvement-request "Add risk controls"
```

#### Batch Analysis
```bash
python alpha_cli.py --batch "ts_mean(close, 20)" "rank(volume)" "subtract(high, low)" --improvement-request "Find related fields"
```

## Example Usage

### Your Example

**Input:**
- Expression: `-abs(subtract(news_max_up_ret, news_max_dn_ret))`
- Improvement Request: "Reduce turnover by adding conditions over some relevant fields"

**Analysis:**
```
Original Expression: -abs(subtract(news_max_up_ret, news_max_dn_ret))

ANALYSIS:
  Operators Used: abs, subtract
  Fields Used: news_max_up_ret, news_max_dn_ret
  Complexity Score: 8
  Categories: Arithmetic

SPECIFIC IMPROVEMENTS:
  1. Add 'hump' operator: hump(-abs(subtract(news_max_up_ret, news_max_dn_ret)), 0.01)
  2. Consider using 'ts_decay_linear' for time-weighted calculations: ts_decay_linear(-abs(subtract(news_max_up_ret, news_max_dn_ret)), 20)

RELATED FIELD SUGGESTIONS:
  1. news_max_ret (Score: 0.850)
     Category: Fundamental
     Description: News sentiment maximum return
     Suggested Combination: subtract(news_max_up_ret, news_max_ret)

IMPROVED EXPRESSION:
  hump(ts_decay_linear(-abs(subtract(news_max_up_ret, news_max_dn_ret)), 20), 0.01)
```

### More Examples

#### 1. Simple Moving Average
```bash
python alpha_cli.py --expression "ts_mean(close, 20)" --improvement-request "Make it more robust against outliers"
```

**Output:**
```
IMPROVED EXPRESSION:
  winsorize(ts_zscore(ts_mean(close, 20), 20), 4)
```

#### 2. Multiple Expression Analysis
```bash
python alpha_cli.py --expression "ts_mean(close, 20); rank(volume)" --improvement-request "Add filtering conditions"
```

**Output:**
```
EXPRESSION COMBINATIONS:
  1. if_else(greater(ts_mean(close, 20), 0), rank(volume), 0)
  2. add(ts_mean(close, 20), rank(volume))
  3. subtract(ts_mean(close, 20), rank(volume))
```

#### 3. Conditional Expression
```bash
python alpha_cli.py --expression "if_else(greater(close, ts_mean(close, 20)), 1, -1)" --improvement-request "Add risk management controls"
```

**Output:**
```
IMPROVED EXPRESSION:
  scale(if_else(greater(close, ts_mean(close, 20)), 1, -1), 1)
```

## Available Operators

The analyzer supports all WorldQuant operators from the `__operator__.json` file:

### Arithmetic Operators
- `add`, `subtract`, `multiply`, `divide`
- `abs`, `sqrt`, `log`, `power`
- `min`, `max`, `sign`, `reverse`

### Time Series Operators
- `ts_mean`, `ts_std_dev`, `ts_zscore`
- `ts_rank`, `ts_corr`, `ts_delay`
- `ts_decay_linear`, `ts_backfill`

### Cross Sectional Operators
- `rank`, `normalize`, `winsorize`
- `zscore`, `quantile`, `scale`

### Group Operators
- `group_neutralize`, `group_rank`
- `group_mean`, `group_scale`

### Logical Operators
- `if_else`, `greater`, `less`
- `and`, `or`, `not`
- `trade_when`, `condition`

## Improvement Types

### 1. Reduce Turnover (`reduce_turnover`)
**Goal:** Minimize transaction costs and position changes

**Added Operators:**
- `hump(x, 0.01)` - Limits changes in input
- `ts_decay_linear(x, 20)` - Time-weighted calculations
- `ts_delay(x, 1)` - Adds delay to reduce noise

**Best For:**
- High-frequency signals
- News-based strategies
- Momentum strategies

### 2. Improve Robustness (`improve_robustness`)
**Goal:** Handle outliers and improve stability

**Added Operators:**
- `winsorize(x, 4)` - Handles outliers
- `ts_zscore(x, 20)` - Normalizes over time
- `ts_rank(x, 20)` - Robust ranking

**Best For:**
- Volatile markets
- Outlier-prone data
- Risk management

### 3. Optimize Performance (`optimize_performance`)
**Goal:** Improve computational efficiency

**Suggestions:**
- Break down complex expressions
- Use vectorized operations
- Simplify nested functions

## Command Line Options

### Alpha CLI Tool
```bash
python alpha_cli.py [OPTIONS]

Options:
  --expression, -e TEXT     Alpha expression to analyze
  --improvement-request, -i Custom improvement request (e.g., "Reduce turnover by adding conditions")
  --operators-file, -f      Path to operators JSON file (default: __operator__.json)
  --output-format, -o       Output format (text, json)
  --batch, -b               Analyze multiple expressions
  --interactive, -t         Run in interactive mode
  --help                    Show help message
```

### Alpha Analyzer Tool
```bash
python alpha_analyzer.py [EXPRESSION] [OPTIONS]

Options:
  --improvement-request     Custom improvement request to focus on
  --operators-file          Path to operators JSON file
  --output-format           Output format (text, json)
  --help                    Show help message
```

## API Usage

### Basic Analysis
```python
from alpha_analyzer import AlphaAnalyzer

# Initialize analyzer
analyzer = AlphaAnalyzer("__operator__.json")

# Analyze expression
expression = "-abs(subtract(news_max_up_ret, news_max_dn_ret))"
suggestions = analyzer.suggest_improvements(expression, "Reduce turnover by adding conditions")

# Get improved expression
improved = analyzer.generate_improved_expression(expression, "Reduce turnover by adding conditions")
print(f"Improved: {improved}")
```

### Detailed Analysis
```python
# Parse expression for detailed analysis
parsed = analyzer.parse_expression(expression)
print(f"Operators: {parsed.operators}")
print(f"Fields: {parsed.fields}")
print(f"Complexity: {parsed.complexity}")
print(f"Categories: {parsed.categories}")
```

## Output Formats

### Text Output (Default)
```
============================================================
ALPHA EXPRESSION ANALYSIS
============================================================
Original Expression: -abs(subtract(news_max_up_ret, news_max_dn_ret))

ANALYSIS:
  Operators Used: abs, subtract
  Fields Used: news_max_up_ret, news_max_dn_ret
  Complexity Score: 8
  Categories: Arithmetic

GENERAL SUGGESTIONS:
  1. Consider adding outlier handling with 'winsorize' or 'zscore' to improve robustness
  2. Consider adding 'hump' to reduce turnover or 'ts_decay_linear' for time-weighted calculations

SPECIFIC IMPROVEMENTS:
  1. Add 'hump' operator: hump(-abs(subtract(news_max_up_ret, news_max_dn_ret)), 0.01)
  2. Consider using 'ts_decay_linear' for time-weighted calculations: ts_decay_linear(-abs(subtract(news_max_up_ret, news_max_dn_ret)), 20)

IMPROVED EXPRESSION:
  hump(ts_decay_linear(-abs(subtract(news_max_up_ret, news_max_dn_ret)), 20), 0.01)
```

### JSON Output
```bash
python alpha_cli.py --expression "ts_mean(close, 20)" --improvement-request "Make it more robust" --output-format json
```

```json
{
  "original_expression": "ts_mean(close, 20)",
  "analysis": {
    "operators_used": ["ts_mean"],
    "fields_used": ["close"],
    "complexity_score": 6,
    "categories": ["Time Series"]
  },
  "general_suggestions": [
    "Consider adding outlier handling with 'winsorize' or 'zscore' to improve robustness"
  ],
  "specific_improvements": [
    "Add outlier handling: winsorize(ts_mean(close, 20), 4)",
    "Add normalization: ts_zscore(ts_mean(close, 20), 20)"
  ]
}
```

## Best Practices

### 1. Choose the Right Improvement Request
- **"Reduce turnover by adding conditions"**: For strategies with high transaction costs
- **"Make it more robust against outliers"**: For volatile or outlier-prone data
- **"Improve performance and efficiency"**: For computationally intensive expressions
- **"Add filtering conditions"**: For conditional logic and thresholds
- **"Add risk management controls"**: For position sizing and risk control
- **"Find related fields for combination"**: To discover related data fields from vector database
- **"Add time decay for news signals"**: For time-weighted calculations
- **"Make it more stable over time"**: For smoothing and stability improvements

### 2. Understand the Trade-offs
- **Turnover reduction** may reduce signal responsiveness
- **Robustness improvements** may smooth out some alpha
- **Performance optimizations** may affect readability

### 3. Test Improvements
- Always backtest improved expressions
- Monitor performance impact
- Validate that improvements achieve intended goals

### 4. Iterative Improvement
- Start with one improvement request
- Gradually add more improvements
- Monitor the cumulative effect

## Troubleshooting

### Common Issues

1. **Operator Not Found**
   - Check that `__operator__.json` is in the correct location
   - Verify operator names match exactly

2. **Complexity Too High**
   - Break down expressions into smaller parts
   - Use intermediate variables

3. **No Improvements Suggested**
   - Expression may already be optimal for the selected criteria
   - Try a different improvement request

### Getting Help

```bash
# Show help for CLI tool
python alpha_cli.py --help

# Show help for analyzer tool
python alpha_analyzer.py --help

# Run examples
python example_usage.py
```

## Vector Database Integration

### Field Discovery and Suggestions

The alpha analyzer integrates with the Pinecone vector database to find related fields and suggest combinations:

**Features:**
- **Automatic Field Discovery**: Searches for related fields based on your expression
- **Relevance Scoring**: Ranks suggestions by relevance to your improvement request
- **Smart Combinations**: Suggests how to combine fields based on your request
- **Category Filtering**: Finds fields from specific categories (Fundamental, Analyst, etc.)

**Example Field Suggestions:**
```
RELATED FIELD SUGGESTIONS:
  1. news_max_ret (Score: 0.850)
     Category: Fundamental
     Description: News sentiment maximum return
     Suggested Combination: subtract(news_max_up_ret, news_max_ret)
  
  2. news_sentiment_score (Score: 0.820)
     Category: Fundamental
     Description: Overall news sentiment score
     Suggested Combination: ts_corr(news_max_up_ret, news_sentiment_score, 20)
```

### Setup Requirements

1. **Environment Variables**: Set `PINECONE_API_KEY` in your `.env` file
2. **Dependencies**: Install `pinecone-client` and `python-dotenv`
3. **Automatic Connection**: The analyzer automatically connects to the vector database

### Usage Examples

**Find Related Fields:**
```bash
python alpha_cli.py --expression "ts_mean(close, 20)" --improvement-request "Find related fields for combination"
```

**Combine with Vector Database:**
```python
# The analyzer automatically uses vector database when available
analyzer = AlphaAnalyzer()
suggestions = analyzer.suggest_improvements("news_max_up_ret", "Find related fields")
```

## Integration
```python
from alpha_analyzer import AlphaAnalyzer
from query_vector_database import WorldQuantMinerQuery

# Analyze alpha expression with vector database integration
analyzer = AlphaAnalyzer()  # Automatically connects to vector database

# Get suggestions including related fields
suggestions = analyzer.suggest_improvements(
    "-abs(subtract(news_max_up_ret, news_max_dn_ret))", 
    "Find related fields for combination"
)

# Access field suggestions
for field_suggestion in suggestions['field_suggestions']:
    print(f"Field: {field_suggestion['field_name']}")
    print(f"Combination: {field_suggestion['suggested_combination']}")

# Generate improved expression
improved_expression = analyzer.generate_improved_expression(
    "-abs(subtract(news_max_up_ret, news_max_dn_ret))", 
    "Reduce turnover by adding conditions"
)

# Analyze multiple expressions
expressions = ["ts_mean(close, 20)", "rank(volume)", "subtract(high, low)"]
multi_suggestions = analyzer.suggest_multiple_expressions(
    expressions, 
    "Add filtering conditions"
)

# Get expression combinations
for combination in multi_suggestions['combinations']:
    print(f"Combination: {combination}")
```

### With Other Tools
The JSON output format makes it easy to integrate with:
- Backtesting frameworks
- Portfolio optimization tools
- Risk management systems
- Reporting dashboards

## License

This tool is provided as-is for analyzing and improving WorldQuant alpha expressions.
