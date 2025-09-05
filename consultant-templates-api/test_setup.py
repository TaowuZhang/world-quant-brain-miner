#!/usr/bin/env python3
"""
Test script to verify the template generator setup
"""

import os
import json
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_credentials():
    """Test if credentials file exists and is valid"""
    print("🔐 Testing credentials...")
    
    if not os.path.exists('credential.txt'):
        print("❌ credential.txt file not found!")
        return False
    
    try:
        with open('credential.txt', 'r') as f:
            creds = json.load(f)
        
        if not isinstance(creds, list) or len(creds) != 2:
            print("❌ credential file format invalid! Should be: [\"username\", \"password\"]")
            return False
        
        print("✅ credentials file is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error reading credentials: {e}")
        return False

def test_deepseek_key():
    """Test if DeepSeek API key is set"""
    print("🔑 Testing DeepSeek API key...")
    
    key = os.getenv('DEEPSEEK_API_KEY')
    if not key:
        print("❌ DEEPSEEK_API_KEY environment variable not set!")
        print("   Set it with: export DEEPSEEK_API_KEY='your-api-key'")
        return False
    
    if len(key) < 20:
        print("❌ DeepSeek API key seems too short!")
        return False
    
    print("✅ DeepSeek API key is set")
    return True

def test_operators():
    """Test if operators file exists"""
    print("📊 Testing operators file...")
    
    if not os.path.exists('operatorRAW.json'):
        print("❌ operatorRAW.json not found!")
        return False
    
    try:
        with open('operatorRAW.json', 'r') as f:
            operators = json.load(f)
        
        if not isinstance(operators, list) or len(operators) == 0:
            print("❌ operatorRAW.json is empty or invalid!")
            return False
        
        print(f"✅ Found {len(operators)} operators")
        return True
        
    except Exception as e:
        print(f"❌ Error reading operators: {e}")
        return False

def test_imports():
    """Test if required modules can be imported"""
    print("📦 Testing imports...")
    
    try:
        import requests
        print("✅ requests module available")
    except ImportError:
        print("❌ requests module not found! Run: pip install requests")
        return False
    
    try:
        from template_generator import TemplateGenerator
        print("✅ template_generator module available")
    except ImportError as e:
        print(f"❌ template_generator import failed: {e}")
        return False
    
    return True

def main():
    print("🧪 Testing Template Generator Setup\n")
    
    tests = [
        test_credentials,
        test_deepseek_key,
        test_operators,
        test_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! You're ready to generate templates.")
        print("\nNext steps:")
        print("1. Run: python run_generator.py")
        print("2. Or: python template_generator.py --help")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
