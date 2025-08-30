#!/bin/bash

# WorldQuant Alpha Mining System with Model Fleet Management
# This script starts the alpha mining system with automatic model downgrading on VRAM issues

set -e

echo "Starting WorldQuant Alpha Mining System with Model Fleet Management..."
echo "================================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if nvidia-docker is available
if ! command -v nvidia-smi > /dev/null 2>&1; then
    echo "Warning: nvidia-smi not found. VRAM monitoring will be limited."
fi

# Create necessary directories
mkdir -p results logs

# Function to cleanup on exit
cleanup() {
    echo "Shutting down alpha mining system..."
    docker-compose -f docker-compose.gpu.yml down
    pkill -f model_fleet_manager.py || true
    pkill -f vram_monitor.py || true
    echo "Cleanup complete"
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Start the model fleet manager in background
echo "Starting Model Fleet Manager..."
python model_fleet_manager.py --monitor &
MODEL_FLEET_PID=$!

# Wait a moment for the fleet manager to initialize
sleep 5

# Check fleet status
echo "Checking Model Fleet Status..."
python model_fleet_manager.py --status

# Start the main Docker services
echo "Starting Docker services..."
docker-compose -f docker-compose.gpu.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Check service status
echo "Service Status:"
docker-compose -f docker-compose.gpu.yml ps

# Show current model being used
echo ""
echo "Current Model Configuration:"
python model_fleet_manager.py --status

echo ""
echo "Alpha Mining System is running with Model Fleet Management!"
echo "================================================================"
echo "Web Dashboard: http://localhost:5000"
echo "Ollama WebUI: http://localhost:3000"
echo "Ollama API: http://localhost:11434"
echo ""
echo "Model Fleet Manager is monitoring for VRAM issues and will automatically"
echo "downgrade to smaller models if needed."
echo ""
echo "Press Ctrl+C to stop all services"

# Keep the script running and monitor the fleet manager
while kill -0 $MODEL_FLEET_PID 2>/dev/null; do
    sleep 10
    
    # Check if fleet manager is still running
    if ! kill -0 $MODEL_FLEET_PID 2>/dev/null; then
        echo "Model Fleet Manager stopped unexpectedly. Restarting..."
        python model_fleet_manager.py --monitor &
        MODEL_FLEET_PID=$!
    fi
done

echo "Model Fleet Manager stopped. Shutting down..."
