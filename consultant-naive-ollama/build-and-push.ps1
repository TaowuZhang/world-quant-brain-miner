# Build and Push Consultant-Naive-Ollama to Docker Hub
# This script builds and pushes the updated image with 100 expressions support

Write-Host "🚀 Building and pushing Consultant-Naive-Ollama to Docker Hub..." -ForegroundColor Green

# Get Docker Hub username
$DockerHubUsername = Read-Host "Enter your Docker Hub username"
$Version = "v1.4.3"

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
    Write-Host "❌ Dockerfile.prod not found. Please run this script from the consultant-naive-ollama directory." -ForegroundColor Red
    exit 1
}

# Build the production image
Write-Host "🔨 Building Docker production image..." -ForegroundColor Yellow
docker build -f Dockerfile.prod -t consultant-naive-ollama:$Version .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Image built successfully" -ForegroundColor Green

# Tag the image
Write-Host "🏷️  Tagging image..." -ForegroundColor Yellow
docker tag consultant-naive-ollama:$Version $DockerHubUsername/consultant-naive-ollama:$Version
docker tag consultant-naive-ollama:$Version $DockerHubUsername/consultant-naive-ollama:latest

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
docker push $DockerHubUsername/consultant-naive-ollama:$Version
docker push $DockerHubUsername/consultant-naive-ollama:latest

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Push failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Image pushed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "🎉 Your updated Consultant-Naive-Ollama image is now available on Docker Hub!" -ForegroundColor Green
Write-Host "📦 Image: $DockerHubUsername/consultant-naive-ollama:$Version" -ForegroundColor Cyan
Write-Host "📦 Latest: $DockerHubUsername/consultant-naive-ollama:latest" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 New features in this version:" -ForegroundColor Yellow
Write-Host "   • Generates 100 alpha ideas per cycle, simulates in batches of 10 (orchestrator updated)" -ForegroundColor White
Write-Host "   • Log monitoring with automatic reset on inactivity (5 min timeout)" -ForegroundColor White
Write-Host "   • Uses multi_simulate for efficient concurrent processing" -ForegroundColor White
Write-Host "   • Increased max concurrent simulations to 5" -ForegroundColor White
Write-Host "   • Better batch management and monitoring" -ForegroundColor White
Write-Host "   • Alpha expression miner now uses multi_simulate (10 at a time)" -ForegroundColor White
Write-Host "   • Enhanced error handling and rate limiting" -ForegroundColor White
Write-Host "   • Uses Ollama structured outputs for reliable JSON generation" -ForegroundColor White
Write-Host "   • No more text sanitization needed - guaranteed valid expressions" -ForegroundColor White
Write-Host "   • Comprehensive Ollama conversation logging" -ForegroundColor White
