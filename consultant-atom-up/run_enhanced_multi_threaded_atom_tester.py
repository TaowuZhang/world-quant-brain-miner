#!/usr/bin/env python3
"""
Runner script for Enhanced Multi-Threaded Atom Tester
- Easy execution with command line arguments
- Progress saving and resume functionality
- Region-based testing with operator combinations
"""

import os
import sys
import argparse
from enhanced_multi_threaded_atom_tester import EnhancedMultiThreadedAtomTester
import json
import time

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced Multi-Threaded Atom Tester with Operator Combinations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_enhanced_multi_threaded_atom_tester.py                    # Run with 8 workers
  python run_enhanced_multi_threaded_atom_tester.py --workers 4        # Run with 4 workers
  python run_enhanced_multi_threaded_atom_tester.py --resume           # Resume from previous progress
  python run_enhanced_multi_threaded_atom_tester.py --workers 8 --resume  # Resume with 8 workers
        """
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=8,
        help='Number of worker threads (default: 8)'
    )
    
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume from previous progress without prompting'
    )
    
    parser.add_argument(
        '--region', '-reg',
        type=str,
        choices=['ASI', 'CHN', 'EUR', 'GLB', 'USA'],
        help='Single region to test (if not specified, all regions will be tested)'
    )
    
    return parser.parse_args()

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'credential.txt',
        'operatorRAW.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Error: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all required files are present:")
        print("  - credential.txt: Your WorldQuant Brain credentials")
        print("  - operatorRAW.json: Operator definitions")
        return False
    
    # Check for data cache files
    cache_files = [f for f in os.listdir('.') if f.startswith('data_fields_cache_') and f.endswith('.json')]
    if not cache_files:
        print("❌ Error: No data_fields_cache_*.json files found!")
        print("Please ensure you have cached data fields files in this directory")
        return False
    
    print(f"✅ Found {len(cache_files)} cache files")
    return True

def check_ollama():
    """Check if Ollama is available"""
    try:
        import ollama
        # Test if Ollama is running
        ollama.list()
        print("✅ Ollama is available and running")
        return True
    except Exception as e:
        print("⚠️ Warning: Ollama is not available or not running!")
        print("The system will use fallback operator combinations instead of AI-generated ones.")
        print("To use Ollama:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Start Ollama service")
        print("3. Pull a model: ollama pull llama3.1")
        print()
        return False

def main():
    """Main function"""
    print("🚀 Enhanced Multi-Threaded Atom Tester")
    print("=" * 60)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check Ollama (optional)
    ollama_available = check_ollama()
    
    # Check for existing progress
    progress_file = "atom_test_progress.json"
    results_file = "enhanced_atom_results.json"
    resume = args.resume
    
    if os.path.exists(progress_file) and not args.resume:
        print(f"📁 Found existing progress file: {progress_file}")
        response = input("Do you want to resume from previous progress? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            resume = True
            print("✅ Will resume from previous progress")
        else:
            print("🔄 Starting fresh")
    
    try:
        print("🚀 Starting Enhanced Multi-Threaded Atom Testing System...")
        print(f"🔧 Using {args.workers} workers for parallel processing")
        if args.region:
            print(f"🌍 Testing region: {args.region}")
        else:
            print("🌍 Testing all regions: ASI, CHN, EUR, GLB, USA")
        print("💡 Use Ctrl+C to stop gracefully")
        print("=" * 60)
        
        # Create and run tester
        tester = EnhancedMultiThreadedAtomTester()
        
        # Override regions if specified
        if args.region:
            tester.regions = [args.region]
            print(f"🎯 Limited to region: {args.region}")
        
        # Run tests
        tester.run_multi_threaded_atom_tests(max_workers=args.workers, resume=resume)
        
        print("\n🎉 Enhanced Multi-Threaded Atom Testing completed!")
        print("Check the following files for results:")
        print(f"  - {results_file} (detailed results)")
        print(f"  - {progress_file} (progress state)")
        print("  - enhanced_multi_threaded_atom_tester.log (execution log)")
        
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
        print("💾 Progress has been saved - you can resume later with --resume")
    except Exception as e:
        print(f"\n💥 Error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
