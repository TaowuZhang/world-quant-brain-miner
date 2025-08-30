#!/bin/bash

echo "🔧 Fixing CUDA library version mismatches for Ollama..."

# Function to create symlinks for CUDA libraries
create_cuda_symlinks() {
    local lib_dir="/usr/lib/x86_64-linux-gnu"
    local cuda_dir="/usr/local/cuda/lib64"
    
    echo "📁 Checking for CUDA libraries in $lib_dir..."
    
    # List all available CUDA libraries
    if ls $lib_dir/libcuda* 1> /dev/null 2>&1; then
        echo "✅ Found CUDA libraries in $lib_dir:"
        ls -la $lib_dir/libcuda* | head -10
        
        # Find the latest libcuda.so version
        if ls $lib_dir/libcuda.so.* 1> /dev/null 2>&1; then
            LATEST_CUDA=$(ls $lib_dir/libcuda.so.* | sort -V | tail -1)
            echo "📦 Latest CUDA library: $LATEST_CUDA"
            
            # Create symlinks for common version patterns that Ollama might expect
            echo "🔗 Creating symlinks for Ollama compatibility..."
            ln -sf "$LATEST_CUDA" "$lib_dir/libcuda.so.550.127.08" || true
            ln -sf "$LATEST_CUDA" "$lib_dir/libcuda.so.550" || true
            ln -sf "$LATEST_CUDA" "$lib_dir/libcuda.so.1" || true
            ln -sf "$LATEST_CUDA" "$lib_dir/libcuda.so" || true
            
            echo "✅ Created symlinks for libcuda.so compatibility"
        fi
        
        # Do the same for other CUDA libraries
        for lib in libcudart libcublas libcublasLt libcurand libcusolver libcusparse; do
            if ls $lib_dir/${lib}.so.* 1> /dev/null 2>&1; then
                LATEST_LIB=$(ls $lib_dir/${lib}.so.* | sort -V | tail -1)
                ln -sf "$LATEST_LIB" "$lib_dir/${lib}.so" || true
                echo "✅ Created symlink for $lib.so"
            fi
        done
    else
        echo "⚠️  No CUDA libraries found in $lib_dir"
        
        # Try CUDA directory
        if [ -d "$cuda_dir" ]; then
            echo "📁 Checking CUDA directory: $cuda_dir"
            if ls $cuda_dir/libcuda* 1> /dev/null 2>&1; then
                echo "✅ Found CUDA libraries in $cuda_dir"
                ls -la $cuda_dir/libcuda* | head -5
                
                # Create symlinks from CUDA directory
                ln -sf $cuda_dir/libcuda.so $lib_dir/libcuda.so || true
                ln -sf $cuda_dir/libcudart.so $lib_dir/libcudart.so || true
                ln -sf $cuda_dir/libcublas.so $lib_dir/libcublas.so || true
                ln -sf $cuda_dir/libcublasLt.so $lib_dir/libcublasLt.so || true
                ln -sf $cuda_dir/libcurand.so $lib_dir/libcurand.so || true
                ln -sf $cuda_dir/libcusolver.so $lib_dir/libcusolver.so || true
                ln -sf $cuda_dir/libcusparse.so $lib_dir/libcusparse.so || true
                
                echo "✅ Created symlinks from CUDA directory"
            fi
        fi
    fi
}

# Function to check GPU availability
check_gpu() {
    echo "🔍 Checking GPU availability..."
    
    # Check nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ nvidia-smi available"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -3
    else
        echo "❌ nvidia-smi not available"
    fi
    
    # Check CUDA libraries
    echo "📚 Checking CUDA libraries..."
    ldconfig -p | grep -i cuda | head -5 || echo "No CUDA libraries found in cache"
}

# Function to test CUDA functionality
test_cuda() {
    echo "🧪 Testing CUDA functionality..."
    
    # Simple CUDA test
    if python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null; then
        echo "✅ PyTorch CUDA test passed"
    else
        echo "⚠️  PyTorch CUDA test failed"
    fi
}

# Main execution
echo "🚀 Starting CUDA library fix..."

# Create symlinks
create_cuda_symlinks

# Update library cache
echo "🔄 Updating library cache..."
ldconfig

# Check GPU
check_gpu

# Test CUDA
test_cuda

echo "✅ CUDA library fix complete!"
echo "📋 Current CUDA library status:"
ls -la /usr/lib/x86_64-linux-gnu/libcuda* 2>/dev/null | head -5 || echo "No CUDA libraries found"
