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

# Create a script to start Ollama and the application
RUN echo '#!/bin/bash\n\
# Check GPU availability\n\
echo "Checking GPU availability..."\n\
nvidia-smi || echo "No GPU detected, will use CPU"\n\
\n\
# Start Ollama in the background\n\
ollama serve &\n\
\n\
# Wait for Ollama to be ready\n\
echo "Waiting for Ollama to start..."\n\
sleep 10\n\n\
\n\
# Pull a financial model (try different names)\n\
echo "Pulling financial model..."\n\
if ollama pull llama3.2:3b; then\n\
    echo "Using llama3.2:3b model"\n\
    MODEL_NAME="llama3.2:3b"\n\
elif ollama pull llama2:7b; then\n\
    echo "Using llama2:7b model"\n\
    MODEL_NAME="llama2:7b"\n\
else\n\
    echo "Using default model"\n\
    MODEL_NAME="llama2:7b"\n\
fi\n\
\n\
# Start the main application with Ollama integration\n\
echo "Starting alpha orchestrator with Ollama using $MODEL_NAME..."\n\
python alpha_orchestrator.py --ollama-url http://localhost:11434 --mode continuous --mining-interval 6 --batch-size 3\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose Ollama port
EXPOSE 11434

# Set the entrypoint
ENTRYPOINT ["/app/start.sh"]
