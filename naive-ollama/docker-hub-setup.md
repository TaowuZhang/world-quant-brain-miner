# Docker Hub Hosting Guide for Naive-Ollama

This guide will help you host the naive-ollama project on Docker Hub.

## Prerequisites

1. **Docker Hub Account**: Create an account at [hub.docker.com](https://hub.docker.com)
2. **Docker Desktop**: Install Docker Desktop on your machine
3. **Git**: Ensure you have Git installed

## Step 1: Prepare Your Docker Image

### 1.1 Create a Production Dockerfile

Create `Dockerfile.prod` in the naive-ollama directory:

```dockerfile
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Set environment variables for CUDA
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /app

# Install system dependencies including Python and CUDA tools
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy application code
COPY . .

# Copy the start scripts and make them executable
COPY start.sh /app/start.sh
COPY start_ollama.sh /app/start_ollama.sh
RUN chmod +x /app/start.sh /app/start_ollama.sh

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose Ollama port
EXPOSE 11434

# Set the entrypoint
ENTRYPOINT ["/app/start.sh"]
```

### 1.2 Update .dockerignore

Ensure your `.dockerignore` excludes sensitive files:

```
# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

# Results and logs
results/
logs/
*.log

# Credentials (will be mounted)
credential.txt
credential.example.txt

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
Dockerfile
.dockerignore
docker-compose.yml
docker-compose.gpu.yml
```

## Step 2: Build and Test Locally

### 2.1 Build the Image

```bash
cd naive-ollama
docker build -f Dockerfile.prod -t naive-ollama:latest .
```

### 2.2 Test the Image

```bash
# Test with GPU support (if available)
docker run --gpus all -p 11434:11434 -p 5000:5000 naive-ollama:latest

# Test without GPU
docker run -p 11434:11434 -p 5000:5000 naive-ollama:latest
```

## Step 3: Push to Docker Hub

### 3.1 Login to Docker Hub

```bash
docker login
# Enter your Docker Hub username and password
```

### 3.2 Tag Your Image

Replace `your-username` with your Docker Hub username:

```bash
docker tag naive-ollama:latest your-username/naive-ollama:latest
docker tag naive-ollama:latest your-username/naive-ollama:v1.0.0
```

### 3.3 Push to Docker Hub

```bash
docker push your-username/naive-ollama:latest
docker push your-username/naive-ollama:v1.0.0
```

## Step 4: Create Docker Hub Repository

1. Go to [hub.docker.com](https://hub.docker.com)
2. Click "Create Repository"
3. Repository name: `naive-ollama`
4. Description: "A sophisticated alpha factor generation system using Ollama with financial language models"
5. Visibility: Choose Public or Private
6. Click "Create"

## Step 5: Add Repository Documentation

### 5.1 Create README.md for Docker Hub

Create a `README.dockerhub.md` file:

```markdown
# Naive-Ollama Alpha Generator

A sophisticated alpha factor generation system that uses Ollama with financial language models to generate, test, and submit alpha factors to WorldQuant Brain.

## Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit (for GPU support)
- WorldQuant Brain account

### Run with GPU Support

```bash
docker run --gpus all -p 11434:11434 -p 5000:5000 \
  -v $(pwd)/credential.txt:/app/credential.txt \
  your-username/naive-ollama:latest
```

### Run without GPU

```bash
docker run -p 11434:11434 -p 5000:5000 \
  -v $(pwd)/credential.txt:/app/credential.txt \
  your-username/naive-ollama:latest
```

### Using Docker Compose

```yaml
version: '3.8'
services:
  naive-ollama:
    image: your-username/naive-ollama:latest
    ports:
      - "11434:11434"
      - "5000:5000"
    volumes:
      - ./credential.txt:/app/credential.txt
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Features

- Local LLM Integration with Ollama
- GPU Acceleration support
- Web Dashboard for monitoring
- Automated alpha generation and submission
- WorldQuant Brain integration

## Configuration

Create a `credential.txt` file with your WorldQuant credentials:
```
["your.email@worldquant.com", "your_password"]
```

## Access Points

- Web Dashboard: http://localhost:5000
- Ollama API: http://localhost:11434

## Support

For issues and questions, please visit the GitHub repository.
```

## Step 6: Automated Builds (Optional)

### 6.1 Connect GitHub Repository

1. In Docker Hub, go to your repository
2. Click "Builds" tab
3. Click "Link to GitHub"
4. Select your GitHub repository
5. Configure build rules:
   - Source: `/naive-ollama`
   - Docker Tag: `latest`
   - Dockerfile location: `/naive-ollama/Dockerfile.prod`

### 6.2 Set up GitHub Actions (Alternative)

Create `.github/workflows/docker-publish.yml`:

```yaml
name: Docker Build and Push

on:
  push:
    branches: [ main ]
    paths: [ 'naive-ollama/**' ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v1
    
    - name: Login to Docker Hub
      uses: docker/login-action@v1
      with:
        username: ${{ secrets.DOCKER_HUB_USERNAME }}
        password: ${{ secrets.DOCKER_HUB_ACCESS_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v2
      with:
        context: ./naive-ollama
        file: ./naive-ollama/Dockerfile.prod
        push: true
        tags: your-username/naive-ollama:latest
```

## Step 7: Usage Instructions

### 7.1 For Users

Users can now pull and run your image:

```bash
# Pull the image
docker pull your-username/naive-ollama:latest

# Run with GPU support
docker run --gpus all -p 11434:11434 -p 5000:5000 \
  -v $(pwd)/credential.txt:/app/credential.txt \
  your-username/naive-ollama:latest
```

### 7.2 Docker Compose Example

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  naive-ollama:
    image: your-username/naive-ollama:latest
    container_name: naive-ollama
    ports:
      - "11434:11434"
      - "5000:5000"
    volumes:
      - ./credential.txt:/app/credential.txt
      - ./results:/app/results
      - ./logs:/app/logs
    environment:
      - OLLAMA_URL=http://localhost:11434
      - MODEL_NAME=llama3.2:3b
      - MINING_INTERVAL=6
      - BATCH_SIZE=3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

## Step 8: Maintenance

### 8.1 Update the Image

When you make changes:

```bash
# Build new version
docker build -f Dockerfile.prod -t your-username/naive-ollama:latest .

# Push to Docker Hub
docker push your-username/naive-ollama:latest
```

### 8.2 Version Management

Use semantic versioning:

```bash
# Tag with version
docker tag naive-ollama:latest your-username/naive-ollama:v1.1.0

# Push version
docker push your-username/naive-ollama:v1.1.0
```

## Troubleshooting

### Common Issues

1. **GPU not detected**: Ensure NVIDIA Container Toolkit is installed
2. **Permission denied**: Check credential.txt file permissions
3. **Port conflicts**: Change exposed ports if needed
4. **Memory issues**: Increase Docker memory limits

### Support

- Check the logs: `docker logs <container_id>`
- Access container shell: `docker exec -it <container_id> /bin/bash`
- Monitor resources: `docker stats <container_id>`

## Next Steps

1. Set up automated builds
2. Add CI/CD pipeline
3. Create multiple tags for different versions
4. Add health checks
5. Optimize image size
6. Add security scanning
