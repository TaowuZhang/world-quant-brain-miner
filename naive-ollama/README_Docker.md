# Naive-Ollma Docker Setup

This Docker setup replaces the Kimi interface with Ollama and FinGPT for generating alpha factors using WorldQuant Brain API.

## Prerequisites

- Docker and Docker Compose installed
- WorldQuant Brain credentials
- At least 8GB RAM (16GB recommended for FinGPT model)
- 20GB+ free disk space for models

### GPU Support (Optional but Recommended)

For GPU acceleration:
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed
- CUDA 12.1+ drivers
- At least 8GB GPU VRAM (16GB+ recommended)

## Quick Start

1. **Set up credentials:**
   ```bash
   # Copy the example credentials file
   cp credential.example.txt credential.txt
   
   # Edit credential.txt with your WorldQuant Brain credentials
   # Format: ["username", "password"]
   ```

2. **Create necessary directories:**
   ```bash
   mkdir -p results logs
   ```

3. **Build and run with Docker Compose:**

   **CPU Only:**
   ```bash
   docker-compose up --build
   ```

   **With GPU Acceleration:**
   ```bash
   # Windows
   start_gpu.bat
   
   # Linux/Mac
   docker-compose -f docker-compose.gpu.yml up --build
   ```

## What's Included

### Services

1. **naive-ollma**: Main application container
   - Runs the alpha orchestrator with integrated workflow
   - Alpha generation using Ollama with llama3.2:3b
   - Alpha expression mining on promising alphas
   - Daily alpha submission with rate limiting
   - GPU acceleration support

2. **ollama-webui** (Optional): Web interface for Ollama
   - Access at http://localhost:3000
   - Monitor model status and chat with models

3. **alpha-dashboard** (Optional): Alpha Generator Dashboard
   - Access at http://localhost:5000
   - Real-time monitoring of GPU, Ollama, orchestrator status
   - Manual controls for mining and submission
   - Statistics and activity logs
   - Alpha generator logs with filtering
   - Real-time system activity monitoring

### Alpha Orchestrator Features

The system now includes an integrated orchestrator that manages:

- **Concurrent Execution**: Alpha generator and expression miner run simultaneously for maximum efficiency
- **Continuous Alpha Generation**: Uses Ollama with llama3.2:3b for generating alpha ideas
- **Expression Mining**: Automatically mines promising alphas every 6 hours to find parameter variations
- **Daily Submission**: Submits successful alphas once per day (2 PM) to avoid rate limits
- **GPU Acceleration**: Full GPU support for faster model inference
- **Automated Workflow**: No manual intervention required
- **Concurrent Simulation Control**: Configurable limit (default: 3) for concurrent simulations to respect API limits

### Operation Modes

- **Continuous Mode** (default): Runs all components with intelligent scheduling
- **Daily Mode**: Single daily workflow execution
- **Individual Modes**: Run specific components (generator, miner, submitter)

### Volumes

- `./credential.txt`: WorldQuant Brain credentials (read-only)
- `./results/`: Generated alpha results and batch files
- `./logs/`: Application logs
- `ollama_data`: Persistent Ollama models and data

## Web Dashboard

The alpha dashboard provides comprehensive monitoring and control:

### Access URLs
- **Main Dashboard**: http://localhost:5000
- **Ollama WebUI**: http://localhost:3000
- **Ollama API**: http://localhost:11434

### Dashboard Features

#### Status Monitoring
- **GPU Status**: Real-time GPU memory, utilization, temperature
- **Ollama Status**: Model loading status, API connectivity
- **Orchestrator Status**: Generation activity, mining schedule, submission status
- **WorldQuant Status**: API connectivity, authentication status
- **Statistics**: Total generated alphas, success rates, 24h metrics

#### Manual Controls
- **Generate Alpha**: Trigger single alpha generation for testing
- **Trigger Mining**: Run alpha expression mining manually
- **Trigger Submission**: Submit successful alphas immediately
- **Refresh Status**: Update all metrics and logs

#### Real-time Logs
- **Alpha Generator Logs**: Filtered logs showing alpha generation activity
- **System Logs**: Complete system activity and Docker container logs
- **Recent Activity**: Timeline of recent events and activities

#### Auto-refresh
- Dashboard automatically refreshes every 30 seconds
- Real-time log updates
- Live status indicators

## Configuration

### Environment Variables

You can modify the `docker-compose.yml` to adjust:

- `OLLAMA_HOST`: Ollama host (default: 0.0.0.0)
- `OLLAMA_ORIGINS`: CORS origins (default: *)
- `PYTHONUNBUFFERED`: Python output buffering (default: 1)

### Command Line Arguments

The application accepts these arguments (modify in docker-compose.yml):

- `--batch-size`: Number of alphas per batch (default: 3)
- `--sleep-time`: Sleep between batches in seconds (default: 30)
- `--max-concurrent`: Maximum concurrent simulations (default: 3)
- `--log-level`: Logging level (default: INFO)
- `--ollama-url`: Ollama API URL (default: http://localhost:11434)
- `--mode`: Operation mode (continuous, daily, generator, miner, submitter)
- `--mining-interval`: Hours between mining runs (default: 6)

## Concurrent Execution Workflow

The Docker setup now runs the alpha generator and expression miner concurrently:

1. **Alpha Generator**: Continuously generates new alpha expressions using Ollama
2. **Expression Miner**: Monitors for promising alphas and mines variations
3. **Coordination**: Both components respect the `--max-concurrent` limit (default: 3)
4. **File Sharing**: Uses `hopeful_alphas.json` as the coordination mechanism
5. **Process Management**: Orchestrator manages both processes and restarts if needed

### Benefits of Concurrent Execution

- **Increased Throughput**: Generator and miner work simultaneously
- **Better Resource Utilization**: No idle time waiting for sequential completion
- **Faster Alpha Discovery**: Mining starts as soon as promising alphas are found
- **Resilient Operation**: Automatic restart of failed components

## Usage

### Starting the Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f naive-ollma

# Stop services
docker-compose down
```

### Web Dashboard

The system includes a comprehensive web dashboard for monitoring and control:

**Access URLs:**
- **Alpha Dashboard**: http://localhost:5000 (Main monitoring interface)
- **Ollama WebUI**: http://localhost:3000 (Ollama chat interface)
- **Ollama API**: http://localhost:11434 (API endpoint)

**Dashboard Features:**
- **Real-time Status**: GPU utilization, Ollama status, orchestrator activity
- **Manual Controls**: Trigger mining and submission operations
- **Statistics**: Alpha generation metrics and success rates
- **Activity Logs**: Recent system activity and error tracking
- **Auto-refresh**: Updates every 30 seconds automatically

**Manual Controls:**
- **Trigger Mining**: Manually start alpha expression mining
- **Trigger Submission**: Manually submit successful alphas
- **Refresh Status**: Update dashboard data immediately

### Monitoring

1. **Application Logs:**
   ```bash
   docker-compose logs -f naive-ollma
   ```

2. **Web Interface:**
   - Open http://localhost:3000 for Ollama WebUI
   - Monitor model status and performance

3. **Results:**
   - Check `./results/` directory for batch results
   - Check `hopeful_alphas.json` for promising alphas

### Troubleshooting

1. **Ollama Model Issues:**
   ```bash
   # Check if FinGPT model is available
   docker-compose exec naive-ollma ollama list
   
   # Pull model manually if needed
   docker-compose exec naive-ollma ollama pull fingpt
   ```

2. **Authentication Issues:**
   - Verify `credential.txt` format: `["username", "password"]`
   - Check WorldQuant Brain API status

3. **Resource Issues:**
   - Increase Docker memory limit (8GB+ recommended)
   - Check available disk space for models

4. **Network Issues:**
   ```bash
   # Test Ollama connectivity
   curl http://localhost:11434/api/tags
   
   # Test WorldQuant Brain connectivity
   docker-compose exec naive-ollma python -c "
   import requests
   from requests.auth import HTTPBasicAuth
   import json
   
   with open('credential.txt') as f:
       creds = json.load(f)
   
   sess = requests.Session()
   sess.auth = HTTPBasicAuth(creds[0], creds[1])
   resp = sess.post('https://api.worldquantbrain.com/authentication')
   print(f'Auth status: {resp.status_code}')
   "
   ```

## Advanced Configuration

### Custom Models

To use a different model instead of FinGPT:

1. Modify the `generate_alpha_ideas_with_ollama` method in `alpha_generator_ollama.py`
2. Change the model name in the Ollama API request
3. Update the Dockerfile to pull your preferred model

### Scaling

For production use:

1. **Increase batch size:**
   ```yaml
   command: ["--batch-size", "10", "--sleep-time", "60"]
   ```

2. **Add resource limits:**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 16G
         cpus: '4.0'
   ```

3. **Use external Ollama service:**
   ```yaml
   environment:
     - OLLAMA_API_BASE_URL=http://external-ollama:11434/api
   ```

### Backup and Recovery

1. **Backup models:**
   ```bash
   docker run --rm -v naive-ollma_ollama_data:/data -v $(pwd):/backup alpine tar czf /backup/ollama_backup.tar.gz -C /data .
   ```

2. **Restore models:**
   ```bash
   docker run --rm -v naive-ollma_ollama_data:/data -v $(pwd):/backup alpine tar xzf /backup/ollama_backup.tar.gz -C /data
   ```

## Performance Tips

1. **Model Optimization:**
   - Use quantized models for faster inference
   - Consider using smaller models for testing

2. **Resource Management:**
   - Monitor memory usage during model loading
   - Adjust batch sizes based on available resources

3. **Network Optimization:**
   - Use local Ollama instance for faster responses
   - Consider caching frequently used data fields

## Security Notes

- Credentials are mounted as read-only
- Ollama API is exposed only locally
- Consider using Docker secrets for production credentials
- Regularly update base images and dependencies

## Support

For issues related to:
- **Ollama**: Check [Ollama documentation](https://ollama.ai/docs)
- **FinGPT**: Check [FinGPT repository](https://github.com/ms-dot-k/FinGPT)
- **WorldQuant Brain**: Check [WorldQuant Brain documentation](https://platform.worldquantbrain.com/)
