#!/usr/bin/env python3
"""
Run custom templates without Ollama - direct bruteforce testing
"""

import sys
import os
from bruteforce_template_generator import BruteforceTemplateGenerator

def main():
    """Run custom templates without Ollama"""
    # Configuration
    credentials_file = "credential.json"
    custom_template_file = "custom_templates.json"
    progress_file = "bruteforce_progress.json"
    
    # Check if credentials file exists
    if not os.path.exists(credentials_file):
        print(f"❌ Credentials file '{credentials_file}' not found!")
        print("Please create a credential.json file with your WorldQuant Brain credentials:")
        print('["username", "password"]')
        return
    
    # Check if custom template file exists
    if not os.path.exists(custom_template_file):
        print(f"❌ Custom template file '{custom_template_file}' not found!")
        print("Please create custom_templates.json with your template patterns")
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
    
    print("🚀 Starting Custom Template Bruteforce Testing")
    print(f"🎯 Custom template file: {custom_template_file}")
    print("🚫 Ollama NOT needed for custom templates")
    print(f"📁 Resume mode: {'Yes' if resume else 'No'}")
    
    try:
        generator = BruteforceTemplateGenerator(
            credentials_path=credentials_file,
            ollama_model="llama3.1",  # Not used for custom templates
            max_concurrent=8
        )
        
        # Run custom templates directly
        generator.run_bruteforce(custom_template_file=custom_template_file, max_batches=1, resume=resume)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        print("💾 Progress has been saved. Use --resume to continue later.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
