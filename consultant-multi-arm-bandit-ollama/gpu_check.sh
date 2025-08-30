#!/bin/bash

echo "🔍 GPU Access Diagnostic Script"
echo "================================"

# Check if we're in a container
echo "📦 Container Environment:"
if [ -f /.dockerenv ]; then
    echo "✅ Running inside Docker container"
else
    echo "⚠️  Not running inside Docker container"
fi

# Check NVIDIA runtime
echo -e "\n🖥️  NVIDIA Runtime Check:"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi available"
    nvidia-smi --version | head -1
else
    echo "❌ nvidia-smi not available"
fi

# Check GPU devices
echo -e "\n🎯 GPU Device Check:"
if nvidia-smi &> /dev/null; then
    echo "✅ GPU devices detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "❌ No GPU devices detected"
    echo "   This could mean:"
    echo "   - NVIDIA Docker runtime not configured"
    echo "   - GPU not passed to container"
    echo "   - NVIDIA drivers not installed"
fi

# Check CUDA
echo -e "\n🔧 CUDA Check:"
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA compiler available"
    nvcc --version | head -1
else
    echo "⚠️  CUDA compiler not available"
fi

# Check environment variables
echo -e "\n🌍 GPU Environment Variables:"
env | grep -E "(CUDA|NVIDIA|GPU)" | sort

# Check Ollama GPU support
echo -e "\n🤖 Ollama GPU Support:"
if command -v ollama &> /dev/null; then
    echo "✅ Ollama available"
    ollama --version
    
    # Check if Ollama can see GPU
    echo -e "\n🔍 Ollama GPU Detection:"
    if ollama list &> /dev/null; then
        echo "✅ Ollama can list models"
    else
        echo "❌ Ollama cannot list models"
    fi
else
    echo "❌ Ollama not available"
fi

# Check GPU libraries
echo -e "\n📚 GPU Libraries:"
if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so* ]; then
    echo "✅ CUDA libraries found"
    ls -la /usr/lib/x86_64-linux-gnu/libcuda.so* | head -3
else
    echo "❌ CUDA libraries not found"
fi

# Check if running with --gpus flag
echo -e "\n🚀 Docker GPU Access:"
if [ -n "$NVIDIA_VISIBLE_DEVICES" ]; then
    echo "✅ NVIDIA_VISIBLE_DEVICES set to: $NVIDIA_VISIBLE_DEVICES"
else
    echo "❌ NVIDIA_VISIBLE_DEVICES not set"
fi

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "✅ CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
else
    echo "❌ CUDA_VISIBLE_DEVICES not set"
fi

echo -e "\n🎯 Summary:"
if nvidia-smi &> /dev/null; then
    echo "✅ GPU access working - Ollama should be able to use GPU"
else
    echo "❌ GPU access not working - Ollama will use CPU only"
    echo ""
    echo "To fix this in your pods, ensure:"
    echo "1. Container is run with --gpus all or runtime: nvidia"
    echo "2. NVIDIA Docker runtime is installed on the host"
    echo "3. NVIDIA drivers are installed on the host"
    echo "4. GPU is available and not in use by other processes"
fi
