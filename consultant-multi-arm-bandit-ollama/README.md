# Naive-Ollama Alpha Generator

A sophisticated alpha factor generation system that uses Ollama with financial language models to generate, test, and submit alpha factors to WorldQuant Brain. This system replaces the previous Kimi interface with a local Ollama-based solution for better performance and control.

## 🚀 Features

- **Local LLM Integration**: Uses Ollama with llama3.2:3b or llama2:7b models
- **GPU Acceleration**: Full NVIDIA GPU support for faster inference
- **Web Dashboard**: Real-time monitoring and control interface
- **Automated Orchestration**: Continuous alpha generation, mining, and submission
- **WorldQuant Brain Integration**: Direct API integration for testing and submission
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Daily Rate Limiting**: Ensures compliance with WorldQuant submission limits

## 📋 Prerequisites

- **Docker & Docker Compose**: For containerized deployment
- **NVIDIA GPU** (optional): For GPU acceleration
- **NVIDIA Container Toolkit**: For GPU support in Docker
- **WorldQuant Brain Account**: For alpha testing and submission

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Dashboard │    │ Alpha Generator │    │  WorldQuant API │
│   (Flask)       │◄──►│   (Ollama)      │◄──►│   (External)    │
│   Port 5000     │    │   Port 11434    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────►│ Alpha Orchestrator │◄─────────────┘
                        │   (Python)      │
                        └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Results &     │
                        │   Logs Storage  │
                        └─────────────────┘
```

## 🚀 Quick Start

### 1. Setup Credentials

Create `credential.txt` with your WorldQuant Brain credentials:
```json
["your.email@worldquant.com", "your_password"]
```

### 2. Start with GPU Support (Recommended)

```bash
# Start the complete system with GPU acceleration
docker-compose -f docker-compose.gpu.yml up -d

# Or use the convenience script
start_gpu.bat
```

### 3. Access the Web Dashboard

Open your browser and navigate to:
- **Main Dashboard**: http://localhost:5000
- **Ollama WebUI**: http://localhost:3000
- **Ollama API**: http://localhost:11434

## 📊 Web Dashboard Features

The web dashboard provides real-time monitoring and control:

### Status Monitoring
- **GPU Status**: Memory usage, utilization, temperature
- **Ollama Status**: Model loading, API connectivity
- **Orchestrator Status**: Generation activity, mining schedule
- **WorldQuant Status**: API connectivity, authentication
- **Statistics**: Generated alphas, success rates, 24h metrics

### Manual Controls
- **Generate Alpha**: Trigger single alpha generation
- **Trigger Mining**: Run alpha expression mining
- **Trigger Submission**: Submit successful alphas
- **Refresh Status**: Update all metrics

### Real-time Logs
- **Alpha Generator Logs**: Filtered logs showing alpha generation activity
- **System Logs**: Complete system activity
- **Recent Activity**: Timeline of recent events

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `MODEL_NAME` | `llama3.2:3b` | LLM model to use |
| `MINING_INTERVAL` | `6` | Hours between mining runs |
| `BATCH_SIZE` | `3` | Alphas per generation batch |

### Docker Compose Services

#### GPU Version (`docker-compose.gpu.yml`)
- **naive-ollma**: Main alpha generation container with GPU support
- **ollama-webui**: Web interface for Ollama management
- **alpha-dashboard**: Flask web dashboard for monitoring

#### CPU Version (`docker-compose.yml`)
- **naive-ollma**: Main alpha generation container (CPU only)
- **ollama-webui**: Web interface for Ollama management

## 📁 File Structure

```
naive-ollama/
├── alpha_generator_ollama.py      # Main alpha generation script
├── alpha_orchestrator.py          # Orchestration and scheduling
├── alpha_expression_miner.py      # Alpha expression mining
├── successful_alpha_submitter.py  # Alpha submission to WorldQuant
├── web_dashboard.py               # Flask web dashboard
├── templates/
│   └── dashboard.html             # Dashboard HTML template
├── results/                       # Generated alpha results
├── logs/                          # System logs
├── Dockerfile                     # Docker image definition
├── docker-compose.gpu.yml         # GPU-enabled deployment
├── docker-compose.yml             # CPU-only deployment
├── requirements.txt               # Python dependencies
├── credential.txt                 # WorldQuant credentials
├── start_gpu.bat                  # Windows GPU startup script
├── start_dashboard.bat            # Windows dashboard startup script
└── README_Docker.md               # Detailed Docker documentation
```

## 🔄 Workflow

### 1. Alpha Generation
- **Continuous Mode**: Generates alphas every 6 hours
- **Batch Processing**: Generates 3 alphas per batch
- **Ollama Integration**: Uses local LLM for alpha idea generation
- **WorldQuant Testing**: Tests each alpha immediately

### 2. Alpha Mining
- **Expression Mining**: Analyzes promising alphas for variations
- **Pattern Recognition**: Identifies successful alpha patterns
- **Optimization**: Suggests improvements to existing alphas

### 3. Alpha Submission
- **Daily Limit**: Submits only once per day
- **Success Filtering**: Only submits alphas with good performance
- **Rate Limiting**: Respects WorldQuant API limits

## 📈 Monitoring

### Real-time Metrics
- **Alpha Generation Rate**: Alphas generated per hour
- **Success Rate**: Percentage of successful alphas
- **GPU Utilization**: Memory and compute usage
- **API Response Times**: WorldQuant API performance

### Log Analysis
- **Alpha Generator Logs**: Filtered for alpha-related activity
- **System Logs**: Complete system activity
- **Error Tracking**: Failed generations and API calls

## 🛠️ Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

#### Ollama Connection Issues
```bash
# Check Ollama service
docker logs naive-ollma-gpu

# Test Ollama API
curl http://localhost:11434/api/tags
```

#### WorldQuant Authentication
```bash
# Verify credentials format
cat credential.txt

# Check authentication in logs
docker logs naive-ollma-gpu | grep "Authentication"
```

### Performance Optimization

#### GPU Memory Issues
- Reduce `BATCH_SIZE` in environment variables
- Monitor GPU memory usage in dashboard
- Consider using smaller model (llama2:7b)

#### Generation Speed
- Increase `BATCH_SIZE` for faster processing
- Use GPU acceleration for better performance
- Monitor Ollama response times

## 🔒 Security

- **Local Processing**: All LLM inference happens locally
- **Credential Protection**: Credentials stored in mounted volume
- **Network Isolation**: Docker network isolation
- **API Rate Limiting**: Respects external API limits

## 📝 Logging

### Log Levels
- **INFO**: Normal operation messages
- **WARNING**: Non-critical issues
- **ERROR**: Critical failures
- **DEBUG**: Detailed debugging information

### Log Locations
- **Container Logs**: `docker logs naive-ollma-gpu`
- **Application Logs**: `./logs/` directory
- **Web Dashboard**: Real-time log display

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker setup
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **WorldQuant Brain**: For providing the alpha testing platform
- **Ollama**: For the local LLM serving framework
- **NVIDIA**: For GPU acceleration support
- **Flask**: For the web dashboard framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker logs
3. Check the web dashboard for real-time status
4. Open an issue on GitHub

---

**Happy Alpha Generation! 🚀**
