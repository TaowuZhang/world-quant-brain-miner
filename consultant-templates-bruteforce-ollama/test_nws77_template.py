#!/usr/bin/env python3
"""
Test script for nws77 template functionality
"""

import json
import os
import sys
from bruteforce_template_generator import BruteforceTemplateGenerator

def test_nws77_template_loading():
    """Test loading and processing of nws77 template"""
    print("🧪 Testing nws77 template loading...")
    
    # Check if template file exists
    template_file = "nws77_template.json"
    if not os.path.exists(template_file):
        print(f"❌ Template file not found: {template_file}")
        return False
    
    # Load template data
    try:
        with open(template_file, 'r') as f:
            template_data = json.load(f)
        print(f"✅ Template file loaded successfully")
        print(f"📊 Template name: {template_data.get('template_name', 'Unknown')}")
        print(f"📊 Dataset: {template_data.get('dataset_specific', 'Unknown')}")
        print(f"📊 Search space: {template_data.get('search_space_size', 'Unknown')}")
    except Exception as e:
        print(f"❌ Failed to load template: {e}")
        return False
    
    # Test template generation
    try:
        generator = BruteforceTemplateGenerator(
            credentials_path="credential.json",  # Dummy path for testing
            target_dataset="nws77"
        )
        
        templates = generator.load_nws77_template(template_file)
        print(f"✅ Generated {len(templates)} template variations")
        
        # Show first few templates
        for i, template in enumerate(templates[:3]):
            print(f"📝 Template {i+1}: {template[:60]}...")
        
        if len(templates) > 3:
            print(f"📝 ... and {len(templates) - 3} more templates")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate templates: {e}")
        return False

def test_template_structure():
    """Test template structure and variables"""
    print("\n🧪 Testing template structure...")
    
    try:
        with open("nws77_template.json", 'r') as f:
            template_data = json.load(f)
        
        # Check required fields
        required_fields = ['template_name', 'template', 'variables', 'search_space_size']
        for field in required_fields:
            if field not in template_data:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check variables
        variables = template_data['variables']
        expected_vars = ['nws77', 'dataprocess_op', 'ts_Statistical_op']
        for var in expected_vars:
            if var not in variables:
                print(f"❌ Missing variable: {var}")
                return False
        
        # Check variable options
        dataprocess_options = variables['dataprocess_op']['options']
        ts_statistical_options = variables['ts_Statistical_op']['options']
        
        print(f"✅ Data processing options: {len(dataprocess_options)}")
        print(f"✅ Time series statistical options: {len(ts_statistical_options)}")
        
        # Calculate expected combinations
        expected_combinations = len(dataprocess_options) * len(ts_statistical_options)
        print(f"📊 Expected combinations: {expected_combinations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Template structure test failed: {e}")
        return False

def test_variable_substitution():
    """Test variable substitution in templates"""
    print("\n🧪 Testing variable substitution...")
    
    try:
        generator = BruteforceTemplateGenerator(
            credentials_path="credential.json",  # Dummy path for testing
            target_dataset="nws77"
        )
        
        templates = generator.load_nws77_template("nws77_template.json")
        
        # Check that variables are properly substituted
        for template in templates:
            if '<dataprocess_op/>' in template:
                print(f"❌ Variable not substituted: <dataprocess_op/>")
                return False
            if '<ts_Statistical_op/>' in template:
                print(f"❌ Variable not substituted: <ts_Statistical_op/>")
                return False
            if '<nws77/>' in template:
                print(f"❌ Variable not substituted: <nws77/>")
                return False
        
        print("✅ All variables properly substituted")
        return True
        
    except Exception as e:
        print(f"❌ Variable substitution test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting nws77 template tests...")
    
    tests = [
        test_nws77_template_loading,
        test_template_structure,
        test_variable_substitution
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ Test passed")
            else:
                print("❌ Test failed")
        except Exception as e:
            print(f"❌ Test error: {e}")
    
    print(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! nws77 template is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
