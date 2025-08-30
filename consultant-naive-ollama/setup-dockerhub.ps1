# Docker Hub Setup Script for Naive-Ollama
# This script helps you build and push the image to Docker Hub

param(
    [Parameter(Mandatory=$true)]
    [string]$DockerHubUsername,
    
    [Parameter(Mandatory=$false)]
    [string]$Version = "latest"
)

Write-Host "🚀 Setting up Naive-Ollama for Docker Hub..." -ForegroundColor Green

# Check if Docker is running
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "Dockerfile.prod")) {
    Write-Host "❌ Dockerfile.prod not found. Please run this script from the naive-ollama directory." -ForegroundColor Red
    exit 1
}

# Build the image
Write-Host "🔨 Building Docker image..." -ForegroundColor Yellow
docker build -f Dockerfile.prod -t naive-ollama:$Version .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Image built successfully" -ForegroundColor Green

# Tag the image
Write-Host "🏷️  Tagging image..." -ForegroundColor Yellow
docker tag naive-ollama:$Version $DockerHubUsername/naive-ollama:$Version
docker tag naive-ollama:$Version $DockerHubUsername/naive-ollama:latest

Write-Host "✅ Image tagged successfully" -ForegroundColor Green

# Login to Docker Hub
Write-Host "🔐 Logging into Docker Hub..." -ForegroundColor Yellow
docker login

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker Hub login failed!" -ForegroundColor Red
    exit 1
}

# Push the image
Write-Host "📤 Pushing image to Docker Hub..." -ForegroundColor Yellow
docker push $DockerHubUsername/naive-ollama:$Version
docker push $DockerHubUsername/naive-ollama:latest

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Push failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Image pushed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "🎉 Your image is now available on Docker Hub!" -ForegroundColor Green
Write-Host "📦 Image: $DockerHubUsername/naive-ollama:$Version" -ForegroundColor Cyan
Write-Host "📦 Latest: $DockerHubUsername/naive-ollama:latest" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Users can now run your image with:" -ForegroundColor Yellow
Write-Host "docker run --gpus all -p 11434:11434 -p 5000:5000 -v `$(pwd)/credential.txt:/app/credential.txt $DockerHubUsername/naive-ollama:latest" -ForegroundColor White
