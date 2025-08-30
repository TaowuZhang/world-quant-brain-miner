# 🚀 Docker Hub Deployment Guide

## Quick Upload to Docker Hub

### Prerequisites

1. **Docker Desktop** installed and running
2. **Docker Hub account** created
3. **Logged into Docker Hub** via `docker login`

### Step 1: Navigate to Project Directory

```powershell
cd consultant-multi-arm-bandit-ollama
```

### Step 2: Run the PowerShell Deployment Script

#### Option A: Automatic Deployment (Recommended)
```powershell
.\dockerhub-deploy.ps1
```

#### Option B: Specify Your Docker Hub Username
```powershell
.\dockerhub-deploy.ps1 -DockerUsername "your-dockerhub-username"
```

#### Option C: Custom Image Name and Version
```powershell
.\dockerhub-deploy.ps1 -DockerUsername "your-username" -ImageName "alpha-miner" -Version "v1.0.0"
```

### Step 3: Available Commands

```powershell
# Build images only
.\dockerhub-deploy.ps1 -Command build

# Test built images
.\dockerhub-deploy.ps1 -Command test

# Push to Docker Hub only
.\dockerhub-deploy.ps1 -Command push

# Create Docker Hub README
.\dockerhub-deploy.ps1 -Command readme

# Show help
.\dockerhub-deploy.ps1 -Command help
```

### Step 4: Manual Docker Commands (Alternative)

If you prefer manual commands:

```powershell
# 1. Build the image
docker build -t your-username/integrated-alpha-miner:latest .

# 2. Tag for GPU version
docker tag your-username/integrated-alpha-miner:latest your-username/integrated-alpha-miner:latest-gpu

# 3. Push to Docker Hub
docker push your-username/integrated-alpha-miner:latest
docker push your-username/integrated-alpha-miner:latest-gpu
```

### Step 5: Verify Upload

1. Go to [Docker Hub](https://hub.docker.com)
2. Navigate to your repository
3. Verify images are available:
   - `your-username/integrated-alpha-miner:latest`
   - `your-username/integrated-alpha-miner:latest-gpu`

## 🎯 What Gets Uploaded

### Docker Images
- **Main Image**: `integrated-alpha-miner:latest`
- **GPU Image**: `integrated-alpha-miner:latest-gpu`

### Features Included
- ✅ Integrated Alpha Mining System
- ✅ Ollama AI Integration
- ✅ GPU Optimization (CUDA)
- ✅ Web Dashboard
- ✅ Model Fleet Management
- ✅ Multi-Arm Bandit & Genetic Algorithm
- ✅ WorldQuant Brain API Integration

### System Architecture
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

## 🚀 After Upload

### Users Can Deploy With:

```bash
# Pull the image
docker pull your-username/integrated-alpha-miner:latest

# Run with Docker Compose
docker-compose up -d

# Or run directly
docker run -d \
  -p 8080:8080 \
  -p 11434:11434 \
  --gpus all \
  your-username/integrated-alpha-miner:latest
```

### Quick Start for Users

1. **Setup Credentials**:
   ```bash
   echo '["username@domain.com", "password"]' > credential.txt
   ```

2. **Run System**:
   ```bash
   docker-compose up -d
   ```

3. **Access Dashboard**:
   - Web Dashboard: http://localhost:8080
   - Ollama API: http://localhost:11434

## 🔧 Troubleshooting

### Common Issues

1. **Docker not running**:
   ```powershell
   # Start Docker Desktop
   Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
   ```

2. **Not logged into Docker Hub**:
   ```powershell
   docker login
   ```

3. **Permission denied**:
   ```powershell
   # Run PowerShell as Administrator
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. **Build fails**:
   ```powershell
   # Clean Docker cache
   docker system prune -a
   docker build --no-cache -t your-username/integrated-alpha-miner:latest .
   ```

### Support

- Check logs: `docker-compose logs`
- Verify image: `docker images`
- Test image: `docker run --rm your-username/integrated-alpha-miner:latest python -c "print('Test successful')"`

## 📊 Success Indicators

✅ **Build Success**: Images created locally
✅ **Test Success**: Images run without errors
✅ **Push Success**: Images uploaded to Docker Hub
✅ **Verification**: Images visible on Docker Hub website

## 🎉 Deployment Complete!

Your integrated alpha mining system is now available on Docker Hub for users worldwide!
