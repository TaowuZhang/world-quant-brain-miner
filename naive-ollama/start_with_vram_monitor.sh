#!/bin/bash

# WorldQuant Alpha Mining System with VRAM Monitoring
# This script starts the alpha mining system with automatic VRAM management

set -e

echo "Starting WorldQuant Alpha Mining System with VRAM Monitoring..."

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
    pkill -f vram_monitor.py || true
    echo "Cleanup complete."
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start VRAM monitor in background
echo "Starting VRAM monitor..."
python vram_monitor.py --threshold 0.85 --interval 30 &
VRAM_MONITOR_PID=$!

# Wait a moment for VRAM monitor to start
sleep 5

# Start the main alpha mining system
echo "Starting alpha mining system..."
docker-compose -f docker-compose.gpu.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Check if services are running
if ! docker-compose -f docker-compose.gpu.yml ps | grep -q "Up"; then
    echo "Error: Services failed to start properly."
    cleanup
    exit 1
fi

echo "Alpha mining system started successfully!"
echo "VRAM monitor PID: $VRAM_MONITOR_PID"
echo "Dashboard available at: http://localhost:5000"
echo "Ollama WebUI available at: http://localhost:3000"

# Monitor the system
echo "Monitoring system... (Press Ctrl+C to stop)"
while true; do
    # Check if VRAM monitor is still running
    if ! kill -0 $VRAM_MONITOR_PID 2>/dev/null; then
        echo "Warning: VRAM monitor stopped unexpectedly"
        # Restart VRAM monitor
        python vram_monitor.py --threshold 0.85 --interval 30 &
        VRAM_MONITOR_PID=$!
    fi
    
    # Check if main services are still running
    if ! docker-compose -f docker-compose.gpu.yml ps | grep -q "Up"; then
        echo "Error: Main services stopped unexpectedly"
        break
    fi
    
    sleep 60
done

# Cleanup on exit
cleanup
