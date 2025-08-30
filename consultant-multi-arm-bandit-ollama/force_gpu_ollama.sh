#!/bin/bash

echo "🔧 Forcing Ollama to use GPU..."

# Stop any running Ollama
pkill -f ollama || true
sleep 2

# Set GPU environment variables explicitly
export OLLAMA_GPU_LAYERS=20
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_GPU_MEMORY_UTILIZATION=0.8
export OLLAMA_GPU_MEMORY_FRACTION=0.8
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all

# Start Ollama with explicit GPU configuration
echo "Starting Ollama with GPU configuration..."
ollama serve > /app/logs/ollama_gpu.log 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
sleep 10

# Check if Ollama is running
if kill -0 $OLLAMA_PID 2>/dev/null; then
    echo "✅ Ollama started with PID: $OLLAMA_PID"
else
    echo "❌ Ollama failed to start"
    exit 1
fi

# Pull model with explicit GPU layers
echo "Pulling model with GPU layers..."
ollama pull deepseek-r1:8b --gpu-layers 20

# Test GPU inference
echo "Testing GPU inference..."
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1:8b",
    "prompt": "Test GPU inference",
    "stream": false,
    "options": {
      "num_gpu": 1,
      "gpu_layers": 20
    }
  }' | head -c 200

echo ""
echo "🎯 GPU configuration applied!"
echo "Check logs: tail -f /app/logs/ollama_gpu.log"
