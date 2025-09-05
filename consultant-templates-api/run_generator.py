#!/usr/bin/env python3
"""
Simple script to run the template generator
"""

import os
import sys
from template_generator import TemplateGenerator
import json
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    # Check if credentials file exists
    if not os.path.exists('credential.txt'):
        print("Error: credential.txt file not found!")
        print("Please create a credential.txt file with your WorldQuant Brain credentials in JSON format:")
        print('["username", "password"]')
        return
    
    # Check if DeepSeek API key is provided
    deepseek_key = os.getenv('DEEPSEEK_API_KEY')
    if not deepseek_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set!")
        print("Please set your DeepSeek API key:")
        print("export DEEPSEEK_API_KEY='your-api-key-here'")
        return
    
    try:
        print("Starting template generation...")
        
        # Initialize generator
        generator = TemplateGenerator('credential.txt', deepseek_key)
        
        # Generate templates for all regions
        print("Generating templates for all regions...")
        result = generator.generate_all_templates()
        
        # Save results
        generator.save_templates(result, 'generatedTemplate.json')
        
        # Print summary
        total_templates = sum(len(templates) for templates in result['templates'].values())
        print(f"\n✅ Generation complete!")
        print(f"📊 Total templates generated: {total_templates}")
        print(f"📁 Saved to: generatedTemplate.json")
        
        print("\n📈 Templates per region:")
        for region, templates in result['templates'].items():
            print(f"  {region}: {len(templates)} templates")
        
        # Show sample templates
        print("\n🔍 Sample templates:")
        for region, templates in result['templates'].items():
            if templates:
                print(f"\n{region}:")
                for i, template in enumerate(templates[:2]):  # Show first 2 templates
                    print(f"  {i+1}. {template['template']}")
                break
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
