# Ollama Setup Script for WorldQuant Template Generator
# This script fixes the common issues with Ollama and the Python client

Write-Host "🔧 Setting up Ollama for WorldQuant Template Generator..." -ForegroundColor Green

# Step 1: Set the correct models directory
Write-Host "📁 Setting Ollama models directory to E:\ollama-model..." -ForegroundColor Yellow
$env:OLLAMA_MODELS = "E:\ollama-model"

# Step 2: Clear proxy settings that interfere with Python client
Write-Host "🌐 Clearing proxy settings to fix Python client 503 errors..." -ForegroundColor Yellow
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:NO_PROXY = "localhost,127.0.0.1"

# Step 3: Kill any existing Ollama processes
Write-Host "🔄 Stopping existing Ollama processes..." -ForegroundColor Yellow
try {
    taskkill /f /im ollama.exe 2>$null
    taskkill /f /im "ollama app.exe" 2>$null
    Start-Sleep 2
}
catch {
    Write-Host "   No existing processes to kill" -ForegroundColor Gray
}

# Step 4: Start Ollama with correct settings
Write-Host "🚀 Starting Ollama server..." -ForegroundColor Yellow
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden

# Step 5: Wait for Ollama to start
Write-Host "⏳ Waiting for Ollama to start..." -ForegroundColor Yellow
Start-Sleep 5

# Step 6: Test if Ollama is working
Write-Host "🧪 Testing Ollama connection..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method GET
    if ($response.models.Count -gt 0) {
        Write-Host "✅ Ollama is running with $($response.models.Count) models available" -ForegroundColor Green
        foreach ($model in $response.models) {
            Write-Host "   📦 $($model.name) ($([math]::Round($model.size/1GB, 2)) GB)" -ForegroundColor Cyan
        }
    }
    else {
        Write-Host "⚠️  Ollama is running but no models found" -ForegroundColor Yellow
        Write-Host "   Run: ollama pull llama3.1" -ForegroundColor Gray
    }
}
catch {
    Write-Host "❌ Ollama is not responding" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Step 7: Test Python client
Write-Host "🐍 Testing Python ollama client..." -ForegroundColor Yellow
try {
    $pythonTest = python -c "import ollama; print('SUCCESS:', len(ollama.list().models))" 2>$null
    if ($pythonTest -match "SUCCESS:") {
        Write-Host "✅ Python ollama client is working" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Python ollama client test failed" -ForegroundColor Red
        Write-Host "   Make sure ollama package is installed: pip install ollama" -ForegroundColor Gray
    }
}
catch {
    Write-Host "❌ Python ollama client test failed" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 8: Final status
Write-Host "`n🎉 Setup Complete!" -ForegroundColor Green
Write-Host "📋 Environment Summary:" -ForegroundColor Cyan
Write-Host "   OLLAMA_MODELS: $env:OLLAMA_MODELS" -ForegroundColor White
Write-Host "   HTTP_PROXY: $env:HTTP_PROXY" -ForegroundColor White
Write-Host "   HTTPS_PROXY: $env:HTTPS_PROXY" -ForegroundColor White
Write-Host "   NO_PROXY: $env:NO_PROXY" -ForegroundColor White

Write-Host "`n🚀 Ready to run the template generator!" -ForegroundColor Green
Write-Host "   python .\run_enhanced_generator_v2.py --region USA --templates 1" -ForegroundColor Gray

Write-Host "`n💡 Note: This script sets environment variables for the current session only." -ForegroundColor Yellow
Write-Host "   To make changes permanent, add them to your PowerShell profile." -ForegroundColor Yellow
