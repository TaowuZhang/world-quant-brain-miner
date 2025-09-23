#!/usr/bin/env python3
"""
Test script for the Bruteforce Template Generator
"""

import json
import os
from bruteforce_template_generator import BruteforceTemplateGenerator

def test_template_generation():
    """Test template generation without API calls"""
    print("🧪 Testing template generation...")
    
    # Create a mock generator for testing
    generator = BruteforceTemplateGenerator(
        credentials_path="credential.example.json",
        ollama_model="llama3.1",
        max_concurrent=2
    )
    
    # Test region configurations
    print("📊 Testing region configurations...")
    for region, config in generator.regions.items():
        print(f"  {region}: {config.universe} - {len(config.neutralization_options)} neutralization options")
    
    print("✅ Template generation test completed")

def test_custom_template_loading():
    """Test custom template loading"""
    print("🧪 Testing custom template loading...")
    
    # Create example template
    example_template = {"template": "rank(close, 20)"}
    with open("test_template.json", "w") as f:
        json.dump(example_template, f)
    
    generator = BruteforceTemplateGenerator(
        credentials_path="credential.example.json",
        ollama_model="llama3.1",
        max_concurrent=2
    )
    
    # Test loading
    template = generator.load_custom_template("test_template.json")
    if template == "rank(close, 20)":
        print("✅ Custom template loading test passed")
    else:
        print("❌ Custom template loading test failed")
    
    # Cleanup
    os.remove("test_template.json")

def main():
    print("🚀 Starting Bruteforce Template Generator Tests")
    
    test_template_generation()
    test_custom_template_loading()
    
    print("🏆 All tests completed!")

if __name__ == "__main__":
    main()
