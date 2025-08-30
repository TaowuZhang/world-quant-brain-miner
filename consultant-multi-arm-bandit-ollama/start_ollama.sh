#!/bin/bash

echo "🚀 Starting Ollama with GPU optimization..."

# Set CUDA library paths explicitly
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

# Set Ollama GPU environment variables
export OLLAMA_GPU_LAYERS=20
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_GPU_MEMORY_UTILIZATION=0.8
export OLLAMA_GPU_MEMORY_FRACTION=0.8
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all

# Check GPU availability
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -3
else
    echo "❌ nvidia-smi not available"
fi

# Check CUDA libraries
echo "📚 Checking CUDA libraries..."
ls -la /usr/lib/x86_64-linux-gnu/libcuda* 2>/dev/null | head -5 || echo "No CUDA libraries found"

# Stop any existing Ollama process
echo "🛑 Stopping any existing Ollama processes..."
pkill -f ollama || true
sleep 2

# Start Ollama in background
echo "🎯 Starting Ollama with GPU support..."
ollama serve > /app/logs/ollama_gpu.log 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to start
echo "⏳ Waiting for Ollama to start..."
sleep 10

# Check if Ollama is running
if ! kill -0 $OLLAMA_PID 2>/dev/null; then
    echo "❌ Failed to start Ollama"
    exit 1
fi

echo "✅ Ollama started with PID: $OLLAMA_PID"

# Pull the model with explicit GPU layers
echo "📥 Pulling deepseek-r1:8b model with GPU layers..."
if ollama pull deepseek-r1:8b --gpu-layers 20; then
    echo "✅ Model pulled successfully with GPU support"
elif ollama pull deepseek-r1:7b --gpu-layers 20; then
    echo "✅ Model pulled successfully with GPU support"
elif ollama pull deepseek-r1:1.5b --gpu-layers 20; then
    echo "✅ Model pulled successfully with GPU support"
elif ollama pull llama3:3b --gpu-layers 20; then
    echo "✅ Model pulled successfully with GPU support"
elif ollama pull phi3:mini --gpu-layers 20; then
    echo "✅ Model pulled successfully with GPU support"
else
    echo "❌ Failed to pull any model"
    exit 1
fi

# Verify GPU usage
echo "🔍 Verifying GPU configuration..."
sleep 5

# Test GPU inference
echo "🧪 Testing GPU inference..."
if ollama run deepseek-r1:8b "Hello, this is a GPU test" --verbose 2>&1 | grep -q "GPU"; then
    echo "✅ GPU inference working!"
else
    echo "⚠️  GPU inference may not be working - check logs"
fi

echo "🎯 Ollama GPU setup complete!"
echo "📊 Check GPU usage: nvidia-smi"
echo "📋 Check Ollama logs: tail -f /app/logs/ollama_gpu.log"

# Keep the script running
wait $OLLAMA_PID
