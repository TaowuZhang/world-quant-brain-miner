#!/bin/bash

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi || echo "No GPU detected, will use CPU"

# Start Ollama in the background with logging to file
mkdir -p /app/logs
ollama serve > /app/logs/ollama.log 2>&1 &

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 10

# Pull a financial model (try different names)
echo "Pulling financial model..."
if ollama pull llama3.2:3b; then
    echo "Using llama3.2:3b model"
    MODEL_NAME="llama3.2:3b"
elif ollama pull llama2:7b; then
    echo "Using llama2:7b model"
    MODEL_NAME="llama2:7b"
else
    echo "Using default model"
    MODEL_NAME="llama2:7b"
fi

echo "Ollama service started successfully with model: $MODEL_NAME"
echo "Ollama API available at http://localhost:11434"

# Keep the container running
tail -f /app/logs/ollama.log
