# WorldQuant Alpha Mining - Multi-Service Docker Setup

This setup runs three different alpha mining simulations in parallel using Docker Compose:

1. **Machine Miner** - Traditional alpha mining using predefined operators and data fields
2. **Alpha Generator** - AI-powered alpha generation using Ollama models
3. **Alpha Expression Miner** - Continuous mining of alpha expression variations

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (recommended)
- WorldQuant Brain credentials

## Quick Start

### 1. Setup Credentials

Ensure your `credential.txt` file contains your WorldQuant credentials in JSON format:
```json
["your_email@example.com", "your_password"]
```

### 2. Start All Services

**Windows:**
```bash
start_all_services.bat
```

**Linux/Mac:**
```bash
chmod +x start_all_services.sh
./start_all_services.sh
```

**Manual:**
```bash
docker-compose up --build -d
```

## Services Overview

### 1. Ollama Service (`ollama`)
- **Purpose**: AI model serving for alpha generation
- **Port**: 11434
- **Models**: Automatically pulls and uses financial models (llama3.2:3b, llama2:7b)
- **GPU**: Full GPU access for model inference

### 2. Machine Miner (`machine-miner`)
- **Purpose**: Traditional alpha mining using WorldQuant Brain API
- **Method**: Uses predefined operators and data fields to generate alphas
- **Region**: USA
- **Universe**: TOP3000
- **Neutralization**: INDUSTRY
- **Logs**: `machine_mining.log`

### 3. Alpha Generator (`alpha-generator`)
- **Purpose**: AI-powered alpha generation using Ollama models
- **Method**: Uses AI models to generate creative alpha expressions
- **Batch Size**: 3 alphas per batch
- **Interval**: 6 hours between batches
- **Logs**: `alpha_generator_ollama.log`

### 4. Alpha Expression Miner (`alpha-expression-miner`)
- **Purpose**: Continuous mining of alpha expression variations
- **Method**: Takes alphas from `hopeful_alphas.json` and generates variations
- **Interval**: 6 hours between mining cycles
- **Logs**: `alpha_expression_miner_continuous.log`

### 5. Web UI (`ollama-webui`)
- **Purpose**: Web interface for monitoring Ollama models
- **Port**: 3000
- **URL**: http://localhost:3000

## Monitoring and Logs

### View Service Status
```bash
docker-compose ps
```

### View Logs for Specific Services
```bash
# Machine Miner logs
docker-compose logs -f machine-miner

# Alpha Generator logs
docker-compose logs -f alpha-generator

# Alpha Expression Miner logs
docker-compose logs -f alpha-expression-miner

# Ollama service logs
docker-compose logs -f ollama
```

### View All Logs
```bash
docker-compose logs -f
```

## File Structure

```
naive-ollama/
├── docker-compose.yml              # Multi-service configuration
├── Dockerfile                      # Base image with CUDA and Python
├── start_ollama.sh                 # Ollama service startup script
├── start_all_services.bat          # Windows startup script
├── start_all_services.sh           # Linux/Mac startup script
├── machine_miner.py                # Traditional alpha mining
├── alpha_generator_ollama.py       # AI-powered alpha generation
├── alpha_expression_miner_continuous.py  # Continuous expression mining
├── machine_lib.py                  # WorldQuant Brain API wrapper
├── credential.txt                  # WorldQuant credentials
├── results/                        # Output directory for results
├── logs/                           # Log files directory
└── hopeful_alphas.json             # Shared alpha storage
```

## Configuration

### Environment Variables
All services have access to:
- `PYTHONUNBUFFERED=1` - Ensures Python output is not buffered
- `NVIDIA_VISIBLE_DEVICES=all` - Full GPU access
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility` - GPU compute capabilities

### Volume Mounts
- `./credential.txt:/app/credential.txt:ro` - Read-only credentials
- `./results:/app/results` - Shared results directory
- `./logs:/app/logs` - Shared logs directory
- `.:/app` - Full application directory for file sharing
- `ollama_data:/root/.ollama` - Persistent Ollama model storage

## Stopping Services

### Stop All Services
```bash
docker-compose down
```

### Stop Specific Service
```bash
docker-compose stop [service-name]
```

### Stop and Remove Volumes
```bash
docker-compose down -v
```

## Troubleshooting

### GPU Issues
If GPU is not detected:
```bash
# Check GPU availability
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

### Service Restart
If a service fails, it will automatically restart. To manually restart:
```bash
docker-compose restart [service-name]
```

### Credential Issues
Ensure `credential.txt` is properly formatted:
```json
["email@example.com", "password"]
```

### Port Conflicts
If ports 11434 or 3000 are already in use:
```bash
# Check what's using the ports
netstat -tulpn | grep :11434
netstat -tulpn | grep :3000

# Stop conflicting services or modify docker-compose.yml
```

## Performance Optimization

### GPU Memory
- Monitor GPU memory usage: `nvidia-smi`
- Services include automatic VRAM cleanup
- Consider reducing batch sizes if GPU memory is limited

### Network
- Services communicate via Docker network
- Ollama API calls use internal network addresses
- External API calls (WorldQuant) use host network

### Storage
- Results are persisted in `./results` directory
- Logs are persisted in `./logs` directory
- Ollama models are cached in Docker volume

## Security Notes

- **Credentials are mounted as read-only** and read from credential.txt file
- **No hardcoded credentials** in any configuration files
- Services run in isolated containers
- No sensitive data is exposed in logs
- Network communication is internal to Docker network
- Credential file should be kept secure and not committed to version control

## Support

For issues or questions:
1. Check service logs: `docker-compose logs [service-name]`
2. Verify credentials in `credential.txt`
3. Ensure Docker and NVIDIA drivers are up to date
4. Check GPU availability and memory
