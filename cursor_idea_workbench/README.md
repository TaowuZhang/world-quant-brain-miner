# WorldQuant Miner Pinecone Query Tool

For pinecone api key, please contact me via discord~

This tool provides a comprehensive interface to query the WorldQuant Miner Pinecone vector database containing financial metrics and fundamental data.

## Features

- **Text-based Search**: Search for financial metrics using natural language queries
- **Category Filtering**: Filter results by metric categories (Fundamental, Analyst, etc.)
- **Specific Metric Lookup**: Retrieve specific metrics by their unique ID
- **Database Analysis**: Get insights about the database structure and content
- **Data Export**: Export metrics to CSV format for further analysis
- **Advanced Filtering**: Combine text search with metadata filters

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. **Set up environment variables** (REQUIRED):
   
   **Option A: Using .env file (Recommended)**
   ```bash
   # Copy the example environment file
   cp env.example .env
   
   # Edit .env file with your actual API keys
   # PINECONE_API_KEY=your-actual-pinecone-api-key-here
   ```
   
   **Option B: Using environment variables**
   ```bash
   # On Windows (PowerShell)
   $env:PINECONE_API_KEY="your-pinecone-api-key-here"
   
   # On Windows (Command Prompt)
   set PINECONE_API_KEY=your-pinecone-api-key-here
   
   # On Linux/Mac
   export PINECONE_API_KEY="your-pinecone-api-key-here"
   ```

   **Option C: Using python-dotenv (Automatic .env loading)**
   ```bash
   # Install python-dotenv
   pip install python-dotenv
   
   # Create .env file (will be automatically loaded)
   echo "PINECONE_API_KEY=your-pinecone-api-key-here" > .env
   ```

## Quick Start

### Basic Usage

```python
from query_vector_database import WorldQuantMinerQuery

# Initialize the client (API key loaded from environment)
client = WorldQuantMinerQuery()

# Search for cash flow related metrics
results = client.search_by_text("cash flow", top_k=5)
for result in results:
    print(f"{result.metadata.get('name')} - {result.score:.4f}")
```

**Note**: The client automatically loads the API key from the `PINECONE_API_KEY` environment variable. Make sure you have set up your environment variables as described in the Installation section.

### Running Examples

Run the example script to see various query patterns:

```bash
python example_queries.py
```

## API Reference

### WorldQuantMinerQuery Class

#### `__init__(api_key=None)`
Initialize the Pinecone client and connect to the index.

**Parameters:**
- `api_key` (str): Pinecone API key. If None, will try to get from `PINECONE_API_KEY` environment variable.

#### `search_by_text(query_text, top_k=10, filter_dict=None)`
Search for similar financial metrics by text query.

**Parameters:**
- `query_text` (str): The text to search for
- `top_k` (int): Number of results to return
- `filter_dict` (dict): Optional metadata filters

**Returns:**
- List of search results with scores and metadata

#### `search_by_category(category, top_k=50)`
Search for metrics within a specific category.

**Parameters:**
- `category` (str): Category to filter by (e.g., "Fundamental", "Analyst")
- `top_k` (int): Number of results to return

**Returns:**
- List of search results

#### `get_metric_by_id(metric_id)`
Retrieve a specific metric by its ID.

**Parameters:**
- `metric_id` (str): The ID of the metric to retrieve

**Returns:**
- Metric data if found, None otherwise

#### `get_metrics_summary()`
Get a summary of the metrics database.

**Returns:**
- Dictionary with database statistics

#### `export_metrics_to_csv(filename)`
Export all metrics to a CSV file.

**Parameters:**
- `filename` (str): Output filename

## Database Structure

The WorldQuant Miner database contains:

- **Total Records**: 1,883
- **Dimensions**: 1024
- **Metric**: Cosine similarity
- **Model**: llama-text-embed-v2
- **Categories**: Fundamental, Analyst, Technical, Market

### Sample Data Structure

Each metric record contains:
- `id`: Unique identifier
- `name`: Metric name
- `category`: Metric category
- `description`: Detailed description
- `timestamp`: When the metric was added

## Usage Examples

### 1. Basic Search Operations

**Search for Revenue Metrics:**
```python
results = client.search_by_text("revenue", top_k=5)
```

**Get All Fundamental Metrics:**
```python
results = client.search_by_category("Fundamental", top_k=50)
```

**Look Up Specific Metric:**
```python
metric = client.get_metric_by_id("fnd6_newa1v1300_ivncf")
```

### 2. Advanced Search with Filters

**Search with Category Filter:**
```python
filter_dict = {"category": {"$eq": "Fundamental"}}
results = client.search_by_text("capital", top_k=5, filter_dict=filter_dict)
```

**Search with Multiple Filters:**
```python
filter_dict = {
    "category": {"$eq": "Analyst"},
    "timestamp": {"$gte": "2025-01-01"}
}
results = client.search_by_text("earnings", top_k=10, filter_dict=filter_dict)
```

### 3. Data Export and Analysis

**Export to CSV:**
```python
client.export_metrics_to_csv("my_metrics.csv")
```

**Get Database Summary:**
```python
summary = client.get_metrics_summary()
print(f"Total records: {summary['total_records']}")
print(f"Categories: {summary['categories']}")
```

**Analyze Trends:**
```python
df = client.analyze_metric_trends()
print(df.head())
```

### 4. Command Line Usage

**Run the main script:**
```bash
python query_vector_database.py
```

**Run example queries:**
```bash
python example_queries.py
```

**Interactive mode:**
```bash
python alpha_cli.py --interactive
```

### 5. Integration Examples

**With pandas for analysis:**
```python
import pandas as pd
from query_vector_database import WorldQuantMinerQuery

client = WorldQuantMinerQuery()
results = client.search_by_text("cash flow", top_k=100)

# Convert to DataFrame
df = pd.DataFrame([
    {
        'name': r.metadata.get('name'),
        'category': r.metadata.get('category'),
        'score': r.score
    }
    for r in results
])

print(df.head())
```

**With matplotlib for visualization:**
```python
import matplotlib.pyplot as plt

# Get category distribution
summary = client.get_metrics_summary()
categories = summary['categories']

plt.figure(figsize=(10, 6))
plt.bar(categories.keys(), categories.values())
plt.title('Metric Categories Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Filter Syntax

Pinecone supports various filter operators:

- `$eq`: Equal to
- `$ne`: Not equal to
- `$gt`: Greater than
- `$gte`: Greater than or equal to
- `$lt`: Less than
- `$lte`: Less than or equal to
- `$in`: In array
- `$nin`: Not in array

Example:
```python
filter_dict = {
    "category": {"$eq": "Fundamental"},
    "timestamp": {"$gte": "2025-01-01"}
}
```

## Error Handling

The tool includes comprehensive error handling:

- API key validation
- Connection error handling
- Missing dependency warnings
- Graceful fallbacks for embedding generation

## Dependencies

- `pinecone-client`: Pinecone Python client
- `sentence-transformers`: For text embedding generation
- `pandas`: For data manipulation and export
- `numpy`: For numerical operations
- `torch`: Required by sentence-transformers
- `transformers`: Required by sentence-transformers

## Security Best Practices

### API Key Management

1. **Never commit API keys to version control**
   - Use `.env` files (add to `.gitignore`)
   - Use environment variables
   - Use secret management services

2. **Use .gitignore**
   ```bash
   # Add to .gitignore
   .env
   *.key
   credentials/
   ```

3. **Rotate API keys regularly**
   - Update your Pinecone API keys periodically
   - Monitor API usage for unusual activity

4. **Use least privilege principle**
   - Only grant necessary permissions to API keys
   - Use read-only keys when possible

### Environment Setup Checklist

- [ ] Created `.env` file from `env.example`
- [ ] Added `.env` to `.gitignore`
- [ ] Set `PINECONE_API_KEY` in `.env` file
- [ ] Verified API key permissions
- [ ] Tested connection with a simple query

## Troubleshooting

### Common Issues

1. **API Key Not Found Error**
   ```bash
   ValueError: Pinecone API key is required. Set PINECONE_API_KEY environment variable or pass api_key parameter.
   ```
   **Solution:**
   - Check if `.env` file exists and contains `PINECONE_API_KEY`
   - Verify environment variable is set: `echo $PINECONE_API_KEY`
   - Restart your terminal/IDE after setting environment variables

2. **Import Error for sentence-transformers**
   ```bash
   ModuleNotFoundError: No module named 'sentence_transformers'
   ```
   **Solution:**
   ```bash
   pip install sentence-transformers
   ```

3. **Connection Issues**
   ```bash
   ConnectionError: Failed to connect to Pinecone
   ```
   **Solution:**
   - Verify internet connectivity
   - Check if Pinecone service is available
   - Verify API key is valid and has proper permissions

4. **Memory Issues**
   ```bash
   MemoryError: Not enough memory
   ```
   **Solution:**
   - Reduce `top_k` parameter for large queries
   - Use pagination for large datasets
   - Process results in smaller batches

5. **Authentication Errors**
   ```bash
   AuthenticationError: Invalid API key
   ```
   **Solution:**
   - Verify API key is correct and not expired
   - Check if key has proper permissions
   - Ensure no extra spaces or characters in the key

### Getting Help

If you encounter issues:

1. **Check the error messages** for specific details
2. **Verify your API key and permissions** in Pinecone console
3. **Ensure all dependencies are installed**: `pip install -r requirements.txt`
4. **Test with a simple query** to isolate the issue
5. **Check the Pinecone documentation** for API changes
6. **Review the logs** for additional error details

### Debug Mode

Enable debug mode to get more detailed error information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from query_vector_database import WorldQuantMinerQuery
client = WorldQuantMinerQuery()
```

## Quick Reference

### Environment Setup
```bash
# 1. Copy environment template
cp env.example .env

# 2. Edit .env file with your API key
# PINECONE_API_KEY=your-actual-pinecone-api-key

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test connection
python query_vector_database.py
```

### Common Commands
```bash
# Basic search
python query_vector_database.py

# Run examples
python example_queries.py

# Alpha expression analysis
python alpha_cli.py --interactive

# Single alpha expression
python alpha_cli.py --expression "ts_mean(close, 20)" --improvement-request "Make it more robust"
```

### API Key Sources (in order of preference)
1. `.env` file (automatically loaded)
2. Environment variable `PINECONE_API_KEY`
3. Passed directly to constructor

## License

This tool is provided as-is for querying the WorldQuant Miner Pinecone database. 