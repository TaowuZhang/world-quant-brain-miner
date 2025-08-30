# 🚀 Integrated Alpha Mining System - Docker Hub Deployment Script (PowerShell)
# This script builds and pushes the integrated alpha mining system to Docker Hub

param(
    [string]$DockerUsername = "",
    [string]$ImageName = "integrated-alpha-miner",
    [string]$Version = "latest",
    [string]$Command = "deploy"
)

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Red
}

# Function to check prerequisites
function Test-Prerequisites {
    Write-Status "Checking prerequisites..."
    
    # Check if Docker is installed
    try {
        $dockerVersion = docker --version
        Write-Success "Docker found: $dockerVersion"
    }
    catch {
        Write-Error "Docker is not installed. Please install Docker Desktop first."
        exit 1
    }
    
    # Check if Docker daemon is running
    try {
        docker info | Out-Null
        Write-Success "Docker daemon is running"
    }
    catch {
        Write-Error "Docker daemon is not running. Please start Docker Desktop."
        exit 1
    }
    
    # Check if logged into Docker Hub
    try {
        $dockerInfo = docker info
        if ($dockerInfo -match "Username") {
            Write-Success "Logged into Docker Hub"
        } else {
            Write-Warning "Not logged into Docker Hub"
            $login = Read-Host "Do you want to login to Docker Hub now? (y/N)"
            if ($login -eq "y" -or $login -eq "Y") {
                docker login
            } else {
                Write-Error "Docker Hub login required to push images."
                exit 1
            }
        }
    }
    catch {
        Write-Error "Could not check Docker Hub login status"
        exit 1
    }
    
    Write-Success "Prerequisites check completed"
}

# Function to get Docker Hub username
function Get-DockerUsername {
    if ([string]::IsNullOrEmpty($DockerUsername)) {
        try {
            $dockerInfo = docker info
            if ($dockerInfo -match "Username:\s+(.+)") {
                $DockerUsername = $matches[1].Trim()
            } else {
                Write-Error "Could not determine Docker Hub username. Please set DockerUsername parameter."
                exit 1
            }
        }
        catch {
            Write-Error "Could not get Docker Hub username"
            exit 1
        }
    }
    Write-Status "Using Docker Hub username: $DockerUsername"
    return $DockerUsername
}

# Function to build images
function Build-Images {
    param([string]$Username)
    
    Write-Status "Building Docker images..."
    
    # Build main image
    Write-Status "Building main image..."
    docker build -t "${Username}/${ImageName}:${Version}" .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build main image"
        exit 1
    }
    
    # Build GPU optimized image (same image, different tag)
    Write-Status "Building GPU optimized image..."
    docker tag "${Username}/${ImageName}:${Version}" "${Username}/${ImageName}:${Version}-gpu"
    
    # Build specific version tags
    if ($Version -ne "latest") {
        Write-Status "Building version-specific tags..."
        docker tag "${Username}/${ImageName}:${Version}" "${Username}/${ImageName}:${Version}-gpu"
    }
    
    Write-Success "All images built successfully"
}

# Function to test images
function Test-Images {
    param([string]$Username)
    
    Write-Status "Testing Docker images..."
    
    # Test main image
    Write-Status "Testing main image..."
    docker run --rm "${Username}/${ImageName}:${Version}" python -c "print('Main image test successful')"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Main image test failed"
        exit 1
    }
    
    # Test GPU image
    Write-Status "Testing GPU image..."
    docker run --rm "${Username}/${ImageName}:${Version}-gpu" python -c "print('GPU image test successful')"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "GPU image test failed"
        exit 1
    }
    
    Write-Success "All image tests passed"
}

# Function to push images to Docker Hub
function Push-Images {
    param([string]$Username)
    
    Write-Status "Pushing images to Docker Hub..."
    
    # Push main image
    Write-Status "Pushing main image..."
    docker push "${Username}/${ImageName}:${Version}"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to push main image"
        exit 1
    }
    
    # Push GPU image
    Write-Status "Pushing GPU image..."
    docker push "${Username}/${ImageName}:${Version}-gpu"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to push GPU image"
        exit 1
    }
    
    # Push version-specific tags
    if ($Version -ne "latest") {
        Write-Status "Pushing version-specific tags..."
        docker push "${Username}/${ImageName}:${Version}-gpu"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to push version-specific tags"
            exit 1
        }
    }
    
    Write-Success "All images pushed to Docker Hub successfully"
}

# Function to create Docker Hub README
function New-DockerHubReadme {
    param([string]$Username)
    
    Write-Status "Creating Docker Hub README..."
    
    $readmeContent = @"
# 🚀 Integrated Alpha Mining System

A complete integrated alpha mining system that combines adaptive mining with multi-arm bandit optimization, genetic algorithms, and Ollama-powered alpha generation for WorldQuant Brain.

## 🎯 Features

- **Adaptive Alpha Mining**: Multi-Arm Bandit and Genetic Algorithm optimization
- **AI-Powered Generation**: Ollama integration for intelligent alpha expression generation
- **GPU Acceleration**: Full NVIDIA GPU support with CUDA optimization
- **Real-time Monitoring**: Web dashboard for system monitoring
- **Model Fleet Management**: Automatic model switching based on VRAM availability
- **Continuous Mining**: 24/7 automated alpha discovery

## 🐳 Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- WorldQuant Brain consultant access

### 1. Setup Credentials

Create your `credential.txt` file:

```bash
echo '["your-username@domain.com", "your-password"]' > credential.txt
```

### 2. Run with Docker Compose

```bash
# Development mode
docker-compose up -d

# Production mode (GPU optimized)
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Access Services

- **Web Dashboard**: http://localhost:8080
- **Ollama API**: http://localhost:11434
- **Ollama WebUI**: http://localhost:3000 (optional)

## 🎮 GPU Support

### GPU Requirements

- **Minimum**: NVIDIA GPU with 8GB VRAM
- **Recommended**: NVIDIA GPU with 16GB+ VRAM (RTX 4090, A100, H100)
- **Multi-GPU**: Support for multiple GPUs with automatic load balancing

### GPU Deployment

```bash
# Deploy with GPU optimization
./deploy-gpu.sh deploy production

# Monitor GPU utilization
./deploy-gpu.sh gpu
```

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Ollama AI     │    │  Alpha           │    │  Web Dashboard  │
│   Service       │◄──►│  Orchestrator    │◄──►│  (Port 8080)    │
│  (Port 11434)   │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  WorldQuant     │    │  Integrated      │    │  Results &      │
│  Brain API      │    │  Alpha Miner     │    │  Logs           │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### Environment Variables

```yaml
# GPU Configuration
NVIDIA_VISIBLE_DEVICES: all
NVIDIA_DRIVER_CAPABILITIES: compute,utility,graphics
CUDA_VISIBLE_DEVICES: all

# Ollama Settings
OLLAMA_GPU_LAYERS: 50
OLLAMA_NUM_PARALLEL: 4
OLLAMA_KEEP_ALIVE: 5m

# Mining Configuration
ADAPTIVE_BATCH_SIZE: 10
ADAPTIVE_ITERATIONS: 5
GENERATOR_BATCH_SIZE: 20
MINING_INTERVAL: 4
```

### Command Line Arguments

```bash
# Continuous mining (default)
python integrated_alpha_miner.py --mode continuous

# Single mining cycle
python integrated_alpha_miner.py --mode single

# Adaptive mining only
python integrated_alpha_miner.py --mode adaptive-only

# Generator only
python integrated_alpha_miner.py --mode generator-only
```

## 📈 Performance Optimization

### GPU Optimization

1. **Model Selection**: Use smaller models for faster inference
2. **Batch Processing**: Adjust batch sizes based on GPU memory
3. **Concurrent Processing**: Use multiple GPU instances
4. **Memory Management**: Monitor GPU memory usage
5. **CUDA Optimization**: Use CUDA-optimized libraries

### Resource Allocation

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

## 🔍 Monitoring

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

## 🔐 Security

### Best Practices

1. **Credentials**: Never commit credentials to version control
2. **Network**: Use internal Docker networks
3. **Volumes**: Mount credentials as read-only
4. **Updates**: Regularly update base images

## 📞 Support

For issues and questions:

1. Check the logs: `docker-compose logs`
2. Verify configuration: `docker-compose config`
3. Test connectivity: `docker-compose exec alpha-orchestrator python -c "import requests; print(requests.get('http://ollama:11434/api/tags').status_code)"`
4. Check GPU: `nvidia-smi`
5. Monitor GPU: `./deploy-gpu.sh gpu`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- **GitHub Repository**: [Integrated Alpha Mining System](https://github.com/your-username/integrated-alpha-miner)
- **Docker Hub**: [${Username}/${ImageName}](https://hub.docker.com/r/${Username}/${ImageName})
- **Documentation**: [Full Documentation](https://github.com/your-username/integrated-alpha-miner/blob/main/README.md)
"@

    $readmeContent | Out-File -FilePath "DOCKERHUB_README.md" -Encoding UTF8
    Write-Success "Docker Hub README created"
}

# Function to show usage
function Show-Usage {
    Write-Host "🚀 Integrated Alpha Mining System - Docker Hub Deployment (PowerShell)" -ForegroundColor $Blue
    Write-Host ""
    Write-Host "Usage: .\dockerhub-deploy.ps1 [OPTIONS] [COMMAND]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -DockerUsername USERNAME    Docker Hub username"
    Write-Host "  -ImageName IMAGE_NAME       Image name (default: integrated-alpha-miner)"
    Write-Host "  -Version VERSION           Version tag (default: latest)"
    Write-Host "  -Command COMMAND           Command to execute (default: deploy)"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build                      Build Docker images"
    Write-Host "  test                       Test built images"
    Write-Host "  push                       Push images to Docker Hub"
    Write-Host "  deploy                     Build, test, and push (default)"
    Write-Host "  readme                     Create Docker Hub README"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\dockerhub-deploy.ps1"
    Write-Host "  .\dockerhub-deploy.ps1 -DockerUsername myusername -Version v1.0.0"
    Write-Host "  .\dockerhub-deploy.ps1 -ImageName alpha-miner -Command build"
    Write-Host "  .\dockerhub-deploy.ps1 -Command test"
}

# Main deployment function
function Start-Deployment {
    param([string]$Username)
    
    Write-Status "Starting Docker Hub deployment..."
    
    Test-Prerequisites
    Build-Images -Username $Username
    Test-Images -Username $Username
    Push-Images -Username $Username
    
    Write-Success "🎉 Docker Hub deployment completed successfully!"
    Write-Status "Images available at:"
    Write-Status "  - ${Username}/${ImageName}:${Version}"
    Write-Status "  - ${Username}/${ImageName}:${Version}-gpu"
    
    if ($Version -ne "latest") {
        Write-Status "  - ${Username}/${ImageName}:${Version}-gpu"
    }
}

# Main script logic
function Main {
    # Show help if requested
    if ($Command -eq "help" -or $Command -eq "-h" -or $Command -eq "--help") {
        Show-Usage
        return
    }
    
    # Get Docker username
    $username = Get-DockerUsername
    
    # Execute command
    switch ($Command.ToLower()) {
        "build" {
            Test-Prerequisites
            Build-Images -Username $username
            Write-Success "Build completed"
        }
        "test" {
            Test-Prerequisites
            Test-Images -Username $username
            Write-Success "Tests completed"
        }
        "push" {
            Test-Prerequisites
            Push-Images -Username $username
            Write-Success "Push completed"
        }
        "deploy" {
            Start-Deployment -Username $username
        }
        "readme" {
            New-DockerHubReadme -Username $username
            Write-Success "README creation completed"
        }
        default {
            Write-Error "Unknown command: $Command"
            Show-Usage
            exit 1
        }
    }
}

# Run main function
Main
