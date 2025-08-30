# 🚀 Integrated Alpha Mining System - Docker Deployment

This repository contains a complete integrated alpha mining system that combines adaptive mining with multi-arm bandit optimization, genetic algorithms, and Ollama-powered alpha generation for WorldQuant Brain.

## 🎯 System Overview

The integrated system consists of:

- **Adaptive Alpha Miner**: Uses Multi-Arm Bandit and Genetic Algorithm to optimize simulation settings and evolve alpha expressions
- **Alpha Generator**: Leverages Ollama AI models to generate alpha expressions using expanded data fields
- **Integrated Orchestrator**: Coordinates both mining approaches for continuous alpha discovery
- **Web Dashboard**: Real-time monitoring and status tracking

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Ollama AI     │    │  Integrated      │    │  Web Dashboard  │
│   Service       │◄──►│  Alpha Miner     │◄──►│  (Port 8080)    │
│  (Port 11434)   │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  WorldQuant     │    │  State Files     │    │  Results &      │
│  Brain API      │    │  (Persistent)    │    │  Logs           │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🐳 Docker Services

### Core Services

1. **ollama** - AI model serving with GPU support
2. **integrated-miner** - Main mining service with adaptive algorithms
3. **web-dashboard** - Monitoring interface
4. **alpha-orchestrator** - Advanced orchestration (optional)

### Optional Services

- **ollama-webui** - Ollama model management interface (Port 3000)

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- WorldQuant Brain consultant access
- Ollama models (will be downloaded automatically)

### 1. Setup Credentials

Create your `credential.txt` file:

```bash
echo '["your-username@domain.com", "your-password"]' > credential.txt
```

### 2. Start the System

#### Development Mode
```bash
docker-compose up -d
```

#### Production Mode (GPU Optimized)
```bash
docker-compose -f docker-compose.prod.yml up -d
```

#### GPU Pod Deployment
```bash
# Make scripts executable
chmod +x deploy.sh deploy-gpu.sh

# Deploy with GPU optimization
./deploy-gpu.sh deploy production
```

### 3. Access Services

- **Web Dashboard**: http://localhost:8080
- **Ollama API**: http://localhost:11434
- **Ollama WebUI**: http://localhost:3000 (optional)

## 🎮 GPU Optimization

### GPU Requirements

- **Minimum**: NVIDIA GPU with 8GB VRAM
- **Recommended**: NVIDIA GPU with 16GB+ VRAM (RTX 4090, A100, H100)
- **Multi-GPU**: Support for multiple GPUs with automatic load balancing

### GPU Environment Variables

```yaml
# GPU Configuration
NVIDIA_VISIBLE_DEVICES: all
NVIDIA_DRIVER_CAPABILITIES: compute,utility,graphics
CUDA_VISIBLE_DEVICES: all
CUDA_LAUNCH_BLOCKING: 0

# Ollama GPU Settings
OLLAMA_GPU_LAYERS: 50  # Use GPU for more layers
OLLAMA_NUM_PARALLEL: 4  # Parallel processing
OLLAMA_KEEP_ALIVE: 5m  # Keep models in memory

# CPU Optimization
OMP_NUM_THREADS: 16  # OpenMP threads
MKL_NUM_THREADS: 16  # Intel MKL threads
NUMBA_NUM_THREADS: 16  # Numba threads
```

### GPU Resource Allocation

#### Production Configuration
```yaml
deploy:
  resources:
    limits:
      memory: 48G
      cpus: '24.0'
    reservations:
      memory: 24G
      cpus: '12.0'
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu, compute, utility, graphics]
```

#### Multi-GPU Configuration
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # Use 2 GPUs
          capabilities: [gpu, compute, utility, graphics]
```

### GPU Monitoring

```bash
# Monitor GPU utilization
./deploy-gpu.sh gpu

# Check GPU status
nvidia-smi

# Real-time monitoring
watch -n 1 nvidia-smi
```

## 📊 Monitoring

### Web Dashboard Features

- Real-time system status
- Mining performance metrics
- GPU and resource utilization
- Recent logs and activity
- Auto-refresh every 30 seconds

### Key Metrics

- **Total Adaptive Alphas**: Number of alphas mined by adaptive algorithm
- **Total Generator Alphas**: Number of alphas generated by Ollama
- **Best Scores**: Performance metrics for discovered alphas
- **System Resources**: CPU, Memory, GPU utilization

## ⚙️ Configuration

### Environment Variables

```yaml
# Ollama Configuration
OLLAMA_HOST: 0.0.0.0
OLLAMA_ORIGINS: "*"
OLLAMA_MODEL: deepseek-r1:8b

# Mining Configuration
ADAPTIVE_BATCH_SIZE: 10  # Increased for GPU pods
ADAPTIVE_ITERATIONS: 5   # Increased for GPU pods
GENERATOR_BATCH_SIZE: 20 # Increased for GPU pods
MINING_INTERVAL: 4       # Reduced for more frequent mining

# GPU Configuration
NVIDIA_VISIBLE_DEVICES: all
NVIDIA_DRIVER_CAPABILITIES: compute,utility,graphics
CUDA_VISIBLE_DEVICES: all
```

### Command Line Arguments

The integrated miner supports various modes:

```bash
# Continuous mining (default)
python integrated_alpha_miner.py --mode continuous

# Single mining cycle
python integrated_alpha_miner.py --mode single

# Adaptive mining only
python integrated_alpha_miner.py --mode adaptive-only

# Generator only
python integrated_alpha_miner.py --mode generator-only

# Submit best alpha
python integrated_alpha_miner.py --mode submit

# Get system status
python integrated_alpha_miner.py --mode status

# Reset system
python integrated_alpha_miner.py --mode reset
```

## 🔧 Advanced Configuration

### Custom Mining Parameters

Edit the docker-compose file to customize mining behavior:

```yaml
command: [
  "python", "integrated_alpha_miner.py",
  "--credentials", "/app/credential.txt",
  "--mode", "continuous",
  "--ollama-url", "http://ollama:11434",
  "--ollama-model", "deepseek-r1:8b",
  "--adaptive-batch-size", "15",        # Customize batch size
  "--adaptive-iterations", "8",         # Customize iterations
  "--lateral-count", "8",               # Customize lateral movements
  "--generator-batch-size", "30",       # Customize generator batch
  "--generator-sleep-time", "15",       # Customize sleep time
  "--mining-interval", "3"              # Customize mining interval
]
```

### GPU Configuration

For multi-GPU setups:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # Use 2 GPUs
          capabilities: [gpu, compute, utility, graphics]
```

### Resource Limits

Add resource constraints:

```yaml
deploy:
  resources:
    limits:
      memory: 64G
      cpus: '32.0'
    reservations:
      memory: 32G
      cpus: '16.0'
```

## 📁 Data Persistence

### Volumes

- **ollama_data**: Ollama models and cache
- **miner_state**: Mining state and progress
- **orchestrator_state**: Orchestrator state (if used)
- **./results**: Alpha results and submissions
- **./logs**: Application logs

### State Files

- `integrated_miner_state.json` - Integrated miner state
- `adaptive_miner_state.json` - Adaptive miner state
- `bandit_state.pkl` - Multi-arm bandit state

## 🔍 Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check NVIDIA Docker runtime
   docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
   
   # Check GPU prerequisites
   ./deploy-gpu.sh optimize
   ```

2. **Ollama Service Not Starting**
   ```bash
   # Check logs
   docker-compose logs ollama
   
   # Restart service
   docker-compose restart ollama
   ```

3. **Authentication Errors**
   ```bash
   # Verify credentials format
   cat credential.txt
   # Should be: ["username", "password"]
   ```

4. **Memory Issues**
   ```bash
   # Check resource usage
   docker stats
   
   # Check GPU memory
   nvidia-smi
   
   # Increase memory limits in docker-compose.yml
   ```

5. **GPU Memory Issues**
   ```bash
   # Check GPU utilization
   nvidia-smi
   
   # Reduce batch sizes
   # Edit docker-compose.yml to reduce --adaptive-batch-size and --generator-batch-size
   ```

### Health Checks

The system includes health checks for all services:

```bash
# Check service health
docker-compose ps

# View health check logs
docker-compose logs --tail=50 integrated-miner
```

### Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f integrated-miner
docker-compose logs -f ollama
docker-compose logs -f web-dashboard
```

## 🚀 Production Deployment

### Production Docker Compose

Use `docker-compose.prod.yml` for production:

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# With custom configuration
docker-compose -f docker-compose.prod.yml -f docker-compose.override.yml up -d
```

### Production Features

- Health checks for all services
- Resource limits and reservations
- Optimized build context (.dockerignore)
- Persistent state management
- Monitoring and logging
- GPU optimization

### Scaling

For high-throughput mining:

```bash
# Scale integrated miner
docker-compose up -d --scale integrated-miner=2

# Scale with resource limits
docker-compose -f docker-compose.prod.yml up -d --scale integrated-miner=3
```

### GPU Pod Deployment

For dedicated GPU pods:

```bash
# Deploy with GPU optimization
./deploy-gpu.sh deploy production

# Monitor GPU utilization
./deploy-gpu.sh gpu

# Check status
./deploy-gpu.sh status
```

## 🔄 Updates and Maintenance

### Updating the System

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Backup and Restore

```bash
# Backup state
tar -czf backup-$(date +%Y%m%d).tar.gz \
  integrated_miner_state.json \
  adaptive_miner_state.json \
  bandit_state.pkl \
  results/ \
  logs/

# Restore state
tar -xzf backup-20241201.tar.gz
```

### Cleanup

```bash
# Remove unused containers and images
docker system prune -a

# Remove unused volumes
docker volume prune

# Clean logs
docker-compose logs --tail=0
```

## 📈 Performance Optimization

### GPU Optimization

1. **Model Selection**: Use smaller models for faster inference
2. **Batch Processing**: Adjust batch sizes based on GPU memory
3. **Concurrent Processing**: Use multiple GPU instances
4. **Memory Management**: Monitor GPU memory usage
5. **CUDA Optimization**: Use CUDA-optimized libraries

### Memory Optimization

1. **Resource Limits**: Set appropriate memory limits
2. **Garbage Collection**: Monitor Python memory usage
3. **State Management**: Regular state cleanup

### Network Optimization

1. **API Rate Limiting**: Respect WorldQuant Brain API limits
2. **Connection Pooling**: Reuse HTTP connections
3. **Caching**: Cache frequently accessed data

## 🔐 Security

### Best Practices

1. **Credentials**: Never commit credentials to version control
2. **Network**: Use internal Docker networks
3. **Volumes**: Mount credentials as read-only
4. **Updates**: Regularly update base images

### Access Control

```yaml
# Restrict dashboard access
web-dashboard:
  environment:
    - FLASK_ENV=production
  networks:
    - internal-network  # No external access
```

## 📞 Support

For issues and questions:

1. Check the logs: `docker-compose logs`
2. Verify configuration: `docker-compose config`
3. Test connectivity: `docker-compose exec integrated-miner python -c "import requests; print(requests.get('http://ollama:11434/api/tags').status_code)"`
4. Check GPU: `nvidia-smi`
5. Monitor GPU: `./deploy-gpu.sh gpu`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
