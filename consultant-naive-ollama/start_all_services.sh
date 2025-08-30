#!/bin/bash

echo "Starting WorldQuant Alpha Mining Services with Docker Compose"
echo "=============================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed. Please install it first."
    exit 1
fi

# Check if credential.txt exists
if [ ! -f "credential.txt" ]; then
    echo "Error: credential.txt not found. Please create it with your WorldQuant credentials."
    exit 1
fi

echo "Building and starting services..."
echo "This will start:"
echo "  - Ollama service (AI model serving)"
echo "  - Machine Miner (traditional alpha mining)"
echo "  - Alpha Generator (AI-powered alpha generation)"
echo "  - Alpha Expression Miner (continuous expression mining)"
echo "  - Web UI (monitoring interface at http://localhost:3000)"
echo ""

# Build and start all services
docker-compose up --build -d

echo ""
echo "Services started successfully!"
echo ""
echo "Service Status:"
docker-compose ps
echo ""
echo "Logs can be viewed with:"
echo "  docker-compose logs -f [service-name]"
echo ""
echo "Available services:"
echo "  - ollama"
echo "  - machine-miner"
echo "  - alpha-generator"
echo "  - alpha-expression-miner"
echo "  - ollama-webui"
echo ""
echo "Web UI available at: http://localhost:3000"
echo "Ollama API available at: http://localhost:11434"
echo ""
echo "To stop all services:"
echo "  docker-compose down"
echo ""
echo "To view logs for a specific service:"
echo "  docker-compose logs -f machine-miner"
echo "  docker-compose logs -f alpha-generator"
echo "  docker-compose logs -f alpha-expression-miner"
