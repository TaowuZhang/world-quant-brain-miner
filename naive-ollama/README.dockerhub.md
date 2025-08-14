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
