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

# Start the alpha submitter in the background (daily submission with hopeful alphas check)
echo "Starting improved alpha submitter in background (daily submission with 50+ hopeful alphas check)..."
python improved_alpha_submitter.py --use-hopeful-file --min-hopeful-count 50 --interval-hours 24 --batch-size 3 --log-level INFO &

# Start the main application with Ollama integration (concurrent mode)
echo "Starting alpha orchestrator with Ollama using $MODEL_NAME in concurrent mode..."
python alpha_orchestrator.py --ollama-url http://localhost:11434 --mode continuous --mining-interval 6 --batch-size 3 --max-concurrent 3
