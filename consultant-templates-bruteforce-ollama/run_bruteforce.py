#!/usr/bin/env python3
"""
Simple runner script for the Bruteforce Template Generator
"""

import sys
import os
from bruteforce_template_generator import BruteforceTemplateGenerator

def main():
    # Default configuration
    credentials_file = "credential.json"  # Changed to match actual file
    ollama_model = "llama3.1"
    max_concurrent = 8
    max_batches = 3  # 3 batches of 4 templates each = 12 templates total
    progress_file = "bruteforce_progress.json"
    custom_template_file = "custom_templates.json"  # Set to None to use AI generation, or "custom_templates.json" for custom templates
    
    # Check if credentials file exists
    if not os.path.exists(credentials_file):
        print(f"❌ Credentials file '{credentials_file}' not found!")
        print("Please create a credentials.json file with your WorldQuant Brain credentials:")
        print('{"username": "your_username", "password": "your_password"}')
        return
    
    # Check for existing progress
    resume = False
    if os.path.exists(progress_file):
        print(f"📁 Found existing progress file: {progress_file}")
        response = input("Do you want to resume from previous progress? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            resume = True
            print("✅ Will resume from previous progress")
        else:
            print("🔄 Starting fresh")
    
    print("🚀 Starting Bruteforce Template Generator")
    if custom_template_file:
        print(f"🎯 Custom template mode: {custom_template_file}")
        print("🚫 Ollama NOT needed for custom templates")
    else:
        print(f"📊 Max concurrent: {max_concurrent}")
        print(f"📊 Max batches: {max_batches} (4 templates per batch, 2 subprocesses each)")
        print(f"📊 Total templates: {max_batches * 4}")
        print(f"🤖 Ollama model: {ollama_model}")
    print(f"📁 Resume mode: {'Yes' if resume else 'No'}")
    
    try:
        generator = BruteforceTemplateGenerator(
            credentials_path=credentials_file,
            ollama_model=ollama_model,
            max_concurrent=max_concurrent
        )
        
        generator.run_bruteforce(custom_template_file=custom_template_file, max_batches=max_batches, resume=resume)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        print("💾 Progress has been saved. Use --resume to continue later.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
