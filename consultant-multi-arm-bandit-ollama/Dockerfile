FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables for CUDA
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Set working directory
WORKDIR /app

# Install system dependencies including Python and CUDA tools
# Use Tsinghua mirror for better network access
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install additional dependencies for the integrated solution
RUN pip install --no-cache-dir \
    schedule \
    flask \
    flask-cors \
    psutil \
    GPUtil \
    numba \
    cupy-cuda12x

# Install latest Ollama v0.9.0 properly
RUN curl -fsSL https://ollama.ai/install.sh | sh && \
    ollama --version && \
    mkdir -p /root/.ollama && \
    chmod 755 /root/.ollama

# Create necessary directories
RUN mkdir -p /app/results /app/logs /app/state /app/orchestrator_state

# Copy application code
COPY . .

# Copy the start scripts and make them executable
COPY start.sh /app/start.sh
COPY start_ollama.sh /app/start_ollama.sh
RUN chmod +x /app/start.sh /app/start_ollama.sh

# Create a health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:11434/api/tags || exit 1' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Set GPU optimization environment variables
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV NUMBA_NUM_THREADS=8
ENV CUDA_LAUNCH_BLOCKING=0

# Expose ports
EXPOSE 11434 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Set the entrypoint (will be overridden by docker-compose)
ENTRYPOINT ["/app/start.sh"]
