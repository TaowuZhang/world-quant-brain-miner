# Integrated Alpha Mining System

## Overview

The Integrated Alpha Mining System combines **Adaptive Alpha Mining** with **Alpha Generation** to create a comprehensive alpha discovery pipeline. The system uses expanded data fields from WorldQuant Brain API and leverages multi-arm bandit optimization and genetic algorithms to find high-performing alphas with good Sharpe ratios and fitness scores.

## Key Features

### 🎯 **Adaptive Alpha Mining**
- **Multi-Arm Bandit Optimization**: Automatically optimizes simulation settings (region, universe, neutralization, truncation, etc.)
- **Genetic Algorithm Evolution**: Evolves alpha expressions based on performance feedback
- **Expanded Data Fields**: Uses 10,000+ data fields from WorldQuant Brain API
- **Lateral Movement**: Explores variations of successful alphas
- **Performance-based Learning**: Adapts strategy based on Sharpe ratio, fitness, turnover, and returns

### 🤖 **Alpha Generation Integration**
- **Ollama AI Models**: Uses DeepSeek-R1 models for alpha idea generation
- **Multi-Simulate Support**: Efficient batch processing with parallel simulations
- **Model Fleet Management**: Automatic model downgrading based on VRAM usage
- **Continuous Generation**: Scheduled alpha generation sessions

### 📊 **Data Field Management**
- **10,000+ Data Fields**: Access to comprehensive financial data
- **High Coverage Fields**: Prioritizes fields with >70% coverage
- **Popular Fields**: Uses fields that appear in many successful alphas
- **Category-based Selection**: Groups fields by category (Analyst, Market, etc.)

## Architecture

```
IntegratedAlphaMiner
├── AdaptiveAlphaMiner
│   ├── DataFieldManager (10,000+ fields)
│   ├── MultiArmBandit (Settings Optimization)
│   ├── GeneticAlgorithm (Expression Evolution)
│   └── LateralMovement (Variation Generation)
├── AlphaGenerator Integration
│   ├── Ollama AI Models
│   ├── Multi-Simulate Processing
│   └── Model Fleet Management
└── PerformanceTracker (Metrics & Analytics)
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- WorldQuant Brain consultant access
- Ollama with DeepSeek-R1 models

### Setup
1. **Install Dependencies**:
```bash
pip install requests numpy schedule
```

2. **Configure Credentials**:
```bash
# Create credential.txt with your WorldQuant Brain credentials
echo '["your_username", "your_password"]' > credential.txt
```

3. **Start Ollama**:
```bash
# Pull required models
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:7b
ollama pull deepseek-r1:1.5b
```

## Usage

### Quick Start

#### 1. **Continuous Integrated Mining**
```bash
# Start continuous mining with both adaptive and generator sessions
python integrated_alpha_miner.py --mode continuous --mining-interval 6
```

#### 2. **Single Mining Cycle**
```bash
# Run a single cycle with both adaptive and generator
python integrated_alpha_miner.py --mode single
```

#### 3. **Adaptive Mining Only**
```bash
# Run only adaptive mining with expanded data fields
python integrated_alpha_miner.py --mode adaptive-only --adaptive-batch-size 5 --adaptive-iterations 3
```

#### 4. **Alpha Generator Only**
```bash
# Run only alpha generator with multi-simulate
python integrated_alpha_miner.py --mode generator-only --generator-batch-size 10
```

### Advanced Usage

#### **Direct Adaptive Miner**
```bash
# Run adaptive miner directly with expanded data fields
python adaptive_alpha_miner.py --mode mine --batch-size 5 --iterations 10
```

#### **Check System Status**
```bash
# Check comprehensive system status
python integrated_alpha_miner.py --mode status
```

#### **Submit Best Alpha**
```bash
# Submit the best alpha found by adaptive mining
python integrated_alpha_miner.py --mode submit
```

## Configuration

### **Expanded Data Fields**

The system now supports 10,000+ data fields across multiple categories:

| Category | Examples | Coverage |
|----------|----------|----------|
| **Analyst** | `act_12m_eps_value`, `act_12m_net_value` | High |
| **Market** | `close`, `volume`, `returns` | Very High |
| **Fundamental** | `pe_ratio`, `book_value` | Medium |
| **Technical** | `rsi`, `macd` | Medium |
| **Alternative** | `sentiment`, `news_count` | Low |

### **Simulation Settings Optimization**

The Multi-Arm Bandit optimizes these expanded settings:

| Setting | Options | Description |
|---------|---------|-------------|
| Region | USA, GLB, EUR, ASI, CHN | Geographic region |
| Universe | TOP3000, TOP1000, TOP500, TOP200, TOPSP500, etc. | Stock universe |
| Neutralization | INDUSTRY, SECTOR, MARKET, NONE, SLOW_AND_FAST, FAST, SLOW | Risk neutralization |
| Truncation | 0.05, 0.08, 0.1, 0.15 | Position truncation |
| Lookback Days | 25, 50, 128, 256, 384, 512 | Historical data window |

### **Genetic Algorithm Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Population Size | 30 | Number of expressions in population |
| Mutation Rate | 0.15 | Probability of mutation |
| Crossover Rate | 0.8 | Probability of crossover |
| Elite Size | 20% | Top performers to preserve |
| Data Field Mutation | Enabled | Mutate data fields in expressions |

### **Integrated Mining Configuration**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Adaptive Batch Size | 5 | Number of alphas per adaptive batch |
| Adaptive Iterations | 3 | Number of adaptive mining iterations |
| Lateral Count | 3 | Number of lateral movements |
| Generator Batch Size | 10 | Number of alphas per generator batch |
| Mining Interval | 6 hours | Interval between mining cycles |

## Data Field Integration

### **High-Quality Field Selection**

The system automatically selects high-quality data fields:

```python
# High coverage fields (>70% coverage)
high_coverage_fields = data_field_manager.get_high_coverage_fields(0.7)

# Popular fields (used in >100 alphas)
popular_fields = data_field_manager.get_popular_fields(100)

# Category-based selection
analyst_fields = data_field_manager.get_fields_by_category("Analyst")
market_fields = data_field_manager.get_fields_by_category("Market")
```

### **Field Categories**

The system groups fields by category for better organization:

- **Analyst**: Broker estimates, earnings forecasts
- **Market**: Price, volume, returns data
- **Fundamental**: Financial ratios, balance sheet data
- **Technical**: Technical indicators
- **Alternative**: News, sentiment, ESG data

### **Example High-Performing Fields**

Based on the expanded data fields:

```python
# High-coverage analyst fields
"act_12m_eps_value"      # Actual EPS (87% coverage)
"act_12m_net_value"      # Actual net income (86% coverage)
"act_12m_sal_value"      # Actual sales (88% coverage)
"act_12m_ebi_value"      # Actual EBIT (72% coverage)

# Popular fields (used in many alphas)
"act_12m_eps_value"      # 481 alphas
"act_12m_net_value"      # 538 alphas
"act_12m_ebi_value"      # 362 alphas
"act_12m_ebt_value"      # 148 alphas
```

## Performance Optimization

### **1. Data Field Strategy**
- **High Coverage Priority**: Use fields with >70% coverage for reliability
- **Popular Field Leverage**: Use fields that appear in many successful alphas
- **Category Diversity**: Mix fields from different categories
- **Field Evolution**: Genetic algorithm can mutate data fields

### **2. Multi-Arm Bandit Tuning**
- **Exploration Rate**: 0.2 for initial learning, 0.1 for mature systems
- **Settings Diversity**: Test different region/universe combinations
- **Neutralization Testing**: Try different risk neutralization strategies
- **Lookback Optimization**: Test different historical windows

### **3. Genetic Algorithm Optimization**
- **Population Diversity**: Maintain diverse expression population
- **Mutation Strategy**: Include data field mutations
- **Elite Preservation**: Keep top 20% performers
- **Crossover Innovation**: Combine successful expression parts

### **4. Lateral Movement**
- **Parameter Variation**: Adjust std, lookback periods
- **Operator Swapping**: Test different operators
- **Wrapper Addition**: Add winsorize, ts_backfill wrappers
- **Data Field Substitution**: Replace data fields with similar ones

## File Structure

```
consultant-multi-arm-bandit-ollama/
├── integrated_alpha_miner.py      # Main integrated mining system
├── adaptive_alpha_miner.py        # Core adaptive mining engine
├── alpha_generator_ollama.py      # Alpha generation with multi-simulate
├── alpha_orchestrator.py          # Original orchestrator
├── credential.txt                 # WorldQuant Brain credentials
├── bandit_state.pkl              # Multi-arm bandit state
├── adaptive_miner_state.json     # Adaptive miner state
├── integrated_miner_state.json   # Integrated miner state
├── adaptive_alpha_miner.log      # Adaptive mining logs
├── integrated_alpha_miner.log    # Integrated mining logs
└── INTEGRATED_README.md          # This file
```

## Monitoring & Analytics

### **Status Check**
```bash
python integrated_alpha_miner.py --mode status
```

Output includes:
- Adaptive mining activity status
- Generator activity status
- Total alphas tested (adaptive + generator)
- Best alpha performance scores
- Bandit arms count
- Genetic algorithm generation
- Data fields loaded count

### **Log Files**
- `integrated_alpha_miner.log` - Integrated system operations
- `adaptive_alpha_miner.log` - Adaptive mining operations
- `alpha_generator_ollama.log` - Alpha generation operations

### **State Files**
- `bandit_state.pkl` - Multi-arm bandit learning state
- `adaptive_miner_state.json` - Complete adaptive mining state
- `integrated_miner_state.json` - Integrated system metrics

## Advanced Features

### **1. Data Field Intelligence**
The system intelligently selects data fields:

```python
# Automatic field selection based on:
# - Coverage (>70% for reliability)
# - Popularity (>100 alphas for proven value)
# - Category diversity (mix of analyst, market, fundamental)
# - Field evolution (genetic algorithm can mutate fields)
```

### **2. Multi-Simulate Integration**
Efficient batch processing:

```bash
# Use multi-simulate for faster processing
python integrated_alpha_miner.py --mode generator-only --generator-batch-size 20
```

### **3. Adaptive Learning**
The system learns and adapts over time:

- **Bandit Learning**: Improves settings selection based on rewards
- **Genetic Evolution**: Evolves expressions based on performance
- **Data Field Optimization**: Learns which fields work best
- **State Persistence**: Maintains learning across sessions

### **4. Lateral Movement Strategies**
Sophisticated variation generation:

- **Parameter Variation**: Adjusts std, lookback periods, etc.
- **Operator Swapping**: Tests different operators (ts_rank vs ts_zscore)
- **Wrapper Addition**: Adds winsorize, ts_backfill wrappers
- **Data Field Substitution**: Replaces fields with similar ones
- **Expression Combination**: Combines successful expression parts

## Best Practices

### **1. Initial Setup**
- Start with continuous mode for 24-48 hours
- Monitor success rates and adjust parameters
- Let the system learn before expecting results
- Check data field loading status

### **2. Parameter Tuning**
- Adjust exploration rate based on performance
- Modify genetic algorithm parameters for your data
- Fine-tune reward function weights
- Optimize batch sizes for your environment

### **3. Data Field Strategy**
- Monitor which data fields perform best
- Use high-coverage fields for reliability
- Mix popular and novel fields
- Let genetic algorithm evolve field selection

### **4. Maintenance**
- Regularly check system status
- Monitor log files for errors
- Backup state files periodically
- Reset system if performance degrades

## Troubleshooting

### **Common Issues**

1. **Data Field Loading Errors**
```bash
# Check API access
python integrated_alpha_miner.py --mode status
# Ensure consultant access for data fields
```

2. **Simulation Failures**
```bash
# Check simulation limits
python integrated_alpha_miner.py --mode status
# Reduce batch size if hitting limits
```

3. **VRAM Issues**
```bash
# Check model fleet status
python alpha_orchestrator.py --mode fleet-status
# Force model downgrade if needed
```

4. **State Corruption**
```bash
# Reset integrated system
python integrated_alpha_miner.py --mode reset
```

### **Performance Monitoring**

Monitor these key metrics:
- **Success Rate**: Should be > 50% for good performance
- **Best Alpha Score**: Tracks improvement over time
- **Data Fields Loaded**: Should be > 1000 fields
- **Bandit Arms**: Should have diverse settings tested
- **Genetic Generations**: Should show evolution progress

## Integration Examples

### **1. Hybrid Mining Strategy**
```bash
# Run integrated mining alongside traditional mining
python integrated_alpha_miner.py --mode continuous &
python alpha_orchestrator.py --mode continuous &
```

### **2. Pipeline Integration**
```bash
# Use adaptive system for initial exploration
python integrated_alpha_miner.py --mode adaptive-only --adaptive-iterations 5

# Follow with generator for AI-generated ideas
python integrated_alpha_miner.py --mode generator-only --generator-batch-size 20
```

### **3. Data Field Exploration**
```bash
# Focus on specific data field categories
# Modify adaptive_alpha_miner.py to prioritize specific categories
```

## Future Enhancements

### **Planned Features**
1. **Advanced Data Field Selection**: ML-based field selection
2. **Cross-Region Mining**: Multi-region alpha discovery
3. **Real-time Adaptation**: Dynamic parameter adjustment
4. **Ensemble Methods**: Combine multiple mining strategies
5. **Advanced Genetic Operators**: More sophisticated crossover and mutation

### **Research Areas**
1. **Meta-Learning**: Learning to learn better mining strategies
2. **Transfer Learning**: Applying knowledge across different markets
3. **Reinforcement Learning**: Advanced reward-based learning
4. **Evolutionary Strategies**: More sophisticated genetic algorithms

## Support & Community

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Monitor system status regularly
4. Consider resetting if performance degrades
5. Check data field loading status

## License

This system is part of the WorldQuant Brain alpha mining toolkit and follows the same licensing terms as the parent project.
