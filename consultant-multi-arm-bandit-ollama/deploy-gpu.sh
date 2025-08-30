#!/bin/bash

# 🚀 Integrated Alpha Mining System - GPU Pod Deployment Script
# This script sets up and deploys the integrated alpha mining system optimized for GPU pods

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU prerequisites
check_gpu_prerequisites() {
    print_status "Checking GPU prerequisites..."
    
    # Check Docker
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    # Check NVIDIA Docker runtime
    if ! command_exists nvidia-smi; then
        print_error "NVIDIA drivers not found. GPU acceleration is required for this deployment."
        exit 1
    fi
    
    # Test NVIDIA Docker runtime
    if ! docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
        print_error "NVIDIA Docker runtime is not available. Please install nvidia-docker2."
        exit 1
    fi
    
    # Check GPU availability
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -eq 0 ]; then
        print_error "No GPUs detected. Please ensure GPUs are available."
        exit 1
    fi
    
    print_success "Found $GPU_COUNT GPU(s)"
    
    # Display GPU info
    print_status "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    
    print_success "GPU prerequisites check completed"
}

# Function to setup credentials
setup_credentials() {
    print_status "Setting up credentials..."
    
    if [ ! -f "credential.txt" ]; then
        print_warning "credential.txt not found. Creating template..."
        echo '["your-username@domain.com", "your-password"]' > credential.txt
        print_error "Please edit credential.txt with your WorldQuant Brain credentials before continuing."
        exit 1
    fi
    
    # Validate credentials format
    if ! python3 -c "import json; json.load(open('credential.txt'))" >/dev/null 2>&1; then
        print_error "credential.txt has invalid JSON format. Please check the file."
        exit 1
    fi
    
    print_success "Credentials setup completed"
}

# Function to create directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p results logs state
    
    print_success "Directories created"
}

# Function to optimize GPU settings
optimize_gpu_settings() {
    print_status "Optimizing GPU settings..."
    
    # Set GPU persistence mode
    if command_exists nvidia-smi; then
        sudo nvidia-smi -pm 1
        print_status "GPU persistence mode enabled"
    fi
    
    # Set GPU compute mode
    if command_exists nvidia-smi; then
        sudo nvidia-smi -c 0
        print_status "GPU compute mode set to default"
    fi
    
    print_success "GPU optimization completed"
}

# Function to build and start services
deploy_gpu_services() {
    local mode=${1:-production}
    
    print_status "Deploying GPU-optimized services in $mode mode..."
    
    # Stop any existing services
    docker-compose down 2>/dev/null || true
    
    # Remove old images to ensure fresh build
    docker system prune -f
    
    if [ "$mode" = "production" ]; then
        docker-compose -f docker-compose.prod.yml up -d --build
    else
        docker-compose up -d --build
    fi
    
    print_success "GPU services deployed successfully"
}

# Function to check GPU utilization
check_gpu_utilization() {
    print_status "Checking GPU utilization..."
    
    # Wait for services to start
    sleep 60
    
    # Check GPU utilization
    if command_exists nvidia-smi; then
        print_status "GPU Utilization:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    fi
    
    # Check if services are using GPU
    if docker-compose ps | grep -q "Up"; then
        print_success "All services are running"
    else
        print_error "Some services failed to start. Check logs with: docker-compose logs"
        exit 1
    fi
}

# Function to check service health
check_health() {
    print_status "Checking service health..."
    
    # Wait for services to start
    sleep 30
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        print_success "All services are running"
    else
        print_error "Some services failed to start. Check logs with: docker-compose logs"
        exit 1
    fi
    
    # Check Ollama health
    if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_success "Ollama service is healthy"
    else
        print_warning "Ollama service is not responding yet. It may still be starting up."
    fi
    
    # Check web dashboard
    if curl -f http://localhost:8080/api/status >/dev/null 2>&1; then
        print_success "Web dashboard is healthy"
    else
        print_warning "Web dashboard is not responding yet. It may still be starting up."
    fi
}

# Function to show status
show_status() {
    print_status "Service Status:"
    docker-compose ps
    
    echo ""
    print_status "GPU Status:"
    if command_exists nvidia-smi; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    fi
    
    echo ""
    print_status "Access URLs:"
    echo "  Web Dashboard: http://localhost:8080"
    echo "  Ollama API: http://localhost:11434"
    echo "  Ollama WebUI: http://localhost:3000 (if enabled)"
    
    echo ""
    print_status "Useful Commands:"
    echo "  View logs: docker-compose logs -f"
    echo "  Stop services: docker-compose down"
    echo "  Restart services: docker-compose restart"
    echo "  Check GPU: nvidia-smi"
    echo "  Monitor GPU: watch -n 1 nvidia-smi"
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    docker-compose down
    print_success "Services stopped"
}

# Function to show logs
show_logs() {
    local service=${1:-""}
    
    if [ -n "$service" ]; then
        print_status "Showing logs for $service..."
        docker-compose logs -f "$service"
    else
        print_status "Showing all logs..."
        docker-compose logs -f
    fi
}

# Function to monitor GPU
monitor_gpu() {
    print_status "Starting GPU monitoring..."
    watch -n 2 nvidia-smi
}

# Function to cleanup
cleanup() {
    print_status "Cleaning up..."
    
    # Stop and remove containers
    docker-compose down
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (optional)
    read -p "Do you want to remove unused volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
    fi
    
    print_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "🚀 Integrated Alpha Mining System - GPU Pod Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy [mode]     Deploy GPU-optimized services (production|development)"
    echo "  start            Start services"
    echo "  stop             Stop services"
    echo "  restart          Restart services"
    echo "  status           Show service and GPU status"
    echo "  logs [service]   Show logs (all or specific service)"
    echo "  health           Check service health"
    echo "  gpu              Monitor GPU utilization"
    echo "  optimize         Optimize GPU settings"
    echo "  cleanup          Clean up containers and images"
    echo "  help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy production"
    echo "  $0 logs integrated-miner"
    echo "  $0 gpu"
    echo "  $0 status"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        check_gpu_prerequisites
        setup_credentials
        create_directories
        optimize_gpu_settings
        deploy_gpu_services "${2:-production}"
        check_health
        check_gpu_utilization
        show_status
        ;;
    "start")
        docker-compose up -d
        print_success "Services started"
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        docker-compose restart
        print_success "Services restarted"
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "$2"
        ;;
    "health")
        check_health
        ;;
    "gpu")
        monitor_gpu
        ;;
    "optimize")
        optimize_gpu_settings
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
