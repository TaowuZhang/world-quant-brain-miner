#!/bin/bash

# 🚀 Integrated Alpha Mining System - Deployment Script
# This script sets up and deploys the integrated alpha mining system

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

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
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
    
    # Check NVIDIA Docker runtime (optional)
    if command_exists nvidia-smi; then
        if docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
            print_success "NVIDIA Docker runtime is available"
        else
            print_warning "NVIDIA Docker runtime is not available. GPU acceleration will not work."
        fi
    else
        print_warning "NVIDIA drivers not found. GPU acceleration will not work."
    fi
    
    print_success "Prerequisites check completed"
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

# Function to build and start services
deploy_services() {
    local mode=${1:-development}
    
    print_status "Deploying services in $mode mode..."
    
    if [ "$mode" = "production" ]; then
        docker-compose -f docker-compose.prod.yml up -d --build
    else
        docker-compose up -d --build
    fi
    
    print_success "Services deployed successfully"
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
    print_status "Access URLs:"
    echo "  Web Dashboard: http://localhost:8080"
    echo "  Ollama API: http://localhost:11434"
    echo "  Ollama WebUI: http://localhost:3000 (if enabled)"
    
    echo ""
    print_status "Useful Commands:"
    echo "  View logs: docker-compose logs -f"
    echo "  Stop services: docker-compose down"
    echo "  Restart services: docker-compose restart"
    echo "  Check health: docker-compose ps"
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
    echo "🚀 Integrated Alpha Mining System - Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy [mode]     Deploy services (development|production)"
    echo "  start            Start services"
    echo "  stop             Stop services"
    echo "  restart          Restart services"
    echo "  status           Show service status"
    echo "  logs [service]   Show logs (all or specific service)"
    echo "  health           Check service health"
    echo "  cleanup          Clean up containers and images"
    echo "  help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy production"
    echo "  $0 logs integrated-miner"
    echo "  $0 status"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        check_prerequisites
        setup_credentials
        create_directories
        deploy_services "${2:-development}"
        check_health
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
