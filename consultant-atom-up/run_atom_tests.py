#!/usr/bin/env python3
"""
Simple runner script for atom testing
"""

import sys
import os
from atom_tester import AtomTester

def main():
    """Run atom tests with default settings"""
    print("🚀 Starting Atom Testing System...")
    print("="*60)
    
    # Check if credential file exists
    if not os.path.exists("credential.txt"):
        print("❌ Error: credential.txt not found!")
        print("Please create credential.txt with your WorldQuant Brain credentials:")
        print("Format 1 (JSON): [\"username\", \"password\"]")
        print("Format 2 (Two-line): username on line 1, password on line 2")
        sys.exit(1)
    
    # Check if operator file exists
    if not os.path.exists("operatorRAW.json"):
        print("❌ Error: operatorRAW.json not found!")
        print("Please copy operatorRAW.json from consultant-templates-api directory")
        sys.exit(1)
    
    # Check if data cache files exist
    cache_files = [f for f in os.listdir('.') if f.startswith('data_fields_cache_') and f.endswith('.json')]
    if not cache_files:
        print("❌ Error: No data_fields_cache_*.json files found!")
        print("Please ensure you have cached data fields files in this directory")
        sys.exit(1)
    
    print(f"✅ Found {len(cache_files)} cache files")
    print(f"✅ Found credential file")
    print(f"✅ Found operator file")
    print()
    
    # Create and run tester
    try:
        tester = AtomTester()
        
        # Run with moderate settings for initial testing
        print("🎯 Running atom tests with 50 test cases and 2 workers...")
        tester.run_atom_tests(max_tests=50, max_workers=2)
        
        print("\n🎉 Atom testing completed!")
        print("Check the following files for results:")
        print("  - atom_test_results.json (detailed results)")
        print("  - atom_statistics.json (statistical summary)")
        print("  - atom_tester.log (execution log)")
        
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
