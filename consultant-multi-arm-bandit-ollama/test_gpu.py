#!/usr/bin/env python3
"""
GPU Test Script for Integrated Alpha Mining System
This script tests GPU utilization and Ollama GPU configuration
"""

import requests
import json
import subprocess
import time
import os

def check_nvidia_smi():
    """Check GPU status using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA-SMI working:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print("❌ NVIDIA-SMI failed")
            return False
    except Exception as e:
        print(f"❌ NVIDIA-SMI error: {e}")
        return False

def check_ollama_gpu():
    """Check if Ollama is using GPU"""
    try:
        # Check Ollama API
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✅ Ollama API responding")
            
            # Check if any models are loaded
            models = response.json().get('models', [])
            if models:
                print(f"✅ Found {len(models)} models:")
                for model in models:
                    print(f"   - {model['name']} ({model['size']})")
                    
                    # Check model details for GPU info
                    try:
                        model_response = requests.post('http://localhost:11434/api/show', 
                                                     json={'name': model['name']}, timeout=10)
                        if model_response.status_code == 200:
                            model_info = model_response.json()
                            if 'gpu_layers' in model_info.get('parameters', {}):
                                gpu_layers = model_info['parameters']['gpu_layers']
                                print(f"     GPU Layers: {gpu_layers}")
                            else:
                                print(f"     GPU Layers: Not specified")
                    except Exception as e:
                        print(f"     Error checking model details: {e}")
            else:
                print("⚠️  No models found")
        else:
            print(f"❌ Ollama API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection error: {e}")
        return False

def test_gpu_inference():
    """Test GPU inference with a simple prompt"""
    try:
        print("\n🧪 Testing GPU inference...")
        
        # Simple test prompt
        test_prompt = {
            "model": "deepseek-r1:8b",
            "prompt": "Generate a simple alpha expression for stock price prediction:",
            "stream": False
        }
        
        start_time = time.time()
        response = requests.post('http://localhost:11434/api/generate', 
                               json=test_prompt, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            duration = end_time - start_time
            print(f"✅ Inference completed in {duration:.2f} seconds")
            print(f"   Response: {result.get('response', '')[:100]}...")
            
            # Check if it was fast (GPU should be much faster than CPU)
            if duration < 5.0:
                print("✅ Fast response - likely using GPU")
            else:
                print("⚠️  Slow response - may be using CPU")
        else:
            print(f"❌ Inference failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Inference test error: {e}")

def check_environment():
    """Check GPU-related environment variables"""
    print("\n🔧 Checking environment variables:")
    
    gpu_vars = [
        'CUDA_VISIBLE_DEVICES',
        'NVIDIA_VISIBLE_DEVICES', 
        'NVIDIA_DRIVER_CAPABILITIES',
        'OLLAMA_GPU_LAYERS',
        'OLLAMA_NUM_PARALLEL',
        'OLLAMA_KEEP_ALIVE'
    ]
    
    for var in gpu_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")

def main():
    """Main test function"""
    print("🚀 GPU Test for Integrated Alpha Mining System")
    print("=" * 50)
    
    # Check environment
    check_environment()
    
    # Check GPU hardware
    print("\n🖥️  Checking GPU hardware:")
    check_nvidia_smi()
    
    # Check Ollama GPU usage
    print("\n🤖 Checking Ollama GPU usage:")
    check_ollama_gpu()
    
    # Test GPU inference
    test_gpu_inference()
    
    print("\n" + "=" * 50)
    print("🎯 GPU Test Complete!")
    print("\nIf you see:")
    print("✅ Fast inference (< 5 seconds) = GPU working")
    print("⚠️  Slow inference (> 10 seconds) = CPU only")
    print("❌ Errors = GPU not configured properly")

if __name__ == "__main__":
    main()
