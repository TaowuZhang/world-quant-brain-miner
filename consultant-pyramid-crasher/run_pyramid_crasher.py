#!/usr/bin/env python3
"""
Run Pyramid Crasher - 3 Concurrent Simulations for Pyramid Cracking
- Integrates with consultant-templates-api
- Runs alongside existing template generation
- Provides specialized pyramid-cracking algorithms
"""

import os
import sys
import argparse
import json
import time
from pyramid_crasher import PyramidCrasher

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run Pyramid Crasher - 3 Concurrent Simulations for Pyramid Cracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pyramid_crasher.py --iterations 50
  python run_pyramid_crasher.py --regions USA EUR --iterations 100
  python run_pyramid_crasher.py --resume --iterations 200
        """
    )
    
    parser.add_argument(
        '--credentials',
        default='credential.txt',
        help='Path to credentials file (default: credential.txt)'
    )
    
    parser.add_argument(
        '--output',
        default='pyramid_results.json',
        help='Output filename (default: pyramid_results.json)'
    )
    
    parser.add_argument(
        '--progress-file',
        default='pyramid_progress.json',
        help='Progress file (default: pyramid_progress.json)'
    )
    
    parser.add_argument(
        '--regions',
        nargs='+',
        choices=['USA', 'GLB', 'EUR', 'ASI', 'CHN'],
        help='Regions to process (default: all regions)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations to run (default: 100)'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=3,
        help='Maximum concurrent simulations (default: 3)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous progress'
    )
    
    parser.add_argument(
        '--breakthrough-threshold',
        type=float,
        default=2.0,
        help='Breakthrough threshold for scoring (default: 2.0)'
    )
    
    return parser.parse_args()

def main():
    """Main function to run pyramid crasher"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if credentials file exists
    if not os.path.exists(args.credentials):
        print("Error: credential.txt file not found!")
        print("Please create a credential.txt file with your WorldQuant Brain credentials in JSON format:")
        print('["username", "password"]')
        return 1
    
    try:
        print("🚀 Starting Pyramid Crasher - 3 Concurrent Simulations...")
        print("🎯 Strategies: Aggregate Breaker, Correlation Hunter, Volatility Exploiter")
        print("🔄 This will run pyramid cracking simulations concurrently")
        print("💡 Use Ctrl+C to stop gracefully")
        print("=" * 80)
        
        # Check for existing progress
        resume = args.resume
        
        if os.path.exists(args.progress_file) and not args.resume:
            print(f"📁 Found existing progress file: {args.progress_file}")
            response = input("Do you want to resume from previous progress? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                resume = True
                print("✅ Will resume from previous progress")
            else:
                print("🔄 Starting fresh")
        elif args.resume:
            print("✅ Resume mode enabled via command line")
        
        # Initialize pyramid crasher
        crasher = PyramidCrasher(
            args.credentials,
            max_concurrent=args.max_concurrent,
            progress_file=args.progress_file,
            results_file=args.output
        )
        
        # Set breakthrough threshold
        crasher.breakthrough_threshold = args.breakthrough_threshold
        
        # Configuration
        all_regions = ['USA', 'GLB', 'EUR', 'ASI', 'CHN']
        regions_to_process = args.regions if args.regions else all_regions
        
        print(f"🌍 Processing regions: {regions_to_process}")
        print(f"🔄 Running {args.iterations} iterations")
        print(f"⚡ Max concurrent: {args.max_concurrent}")
        print(f"💥 Breakthrough threshold: {args.breakthrough_threshold}")
        print("=" * 80)
        
        # Run concurrent pyramid cracking
        start_time = time.time()
        results = crasher.run_concurrent_pyramid_cracking(regions_to_process, args.iterations)
        end_time = time.time()
        
        # Save final results
        crasher.save_results(results, args.output)
        
        # Print final summary
        print(f"\n{'='*70}")
        print("🎉 PYRAMID CRACKING COMPLETE!")
        print(f"{'='*70}")
        
        total_simulations = crasher.completed_count
        successful_sims = crasher.successful_count
        breakthrough_sims = crasher.breakthrough_count
        runtime = end_time - start_time
        
        print(f"📊 Final Statistics:")
        print(f"   Total concurrent simulations: {total_simulations}")
        print(f"   Successful simulations: {successful_sims}")
        print(f"   Failed simulations: {total_simulations - successful_sims}")
        print(f"   Breakthrough simulations: {breakthrough_sims}")
        print(f"   Success rate: {successful_sims/total_simulations*100:.1f}%" if total_simulations > 0 else "   Success rate: N/A")
        print(f"   Breakthrough rate: {breakthrough_sims/successful_sims*100:.1f}%" if successful_sims > 0 else "   Breakthrough rate: N/A")
        print(f"   Best breakthrough score: {crasher.best_breakthrough:.3f}")
        print(f"   Runtime: {runtime:.1f} seconds")
        print(f"   Results saved to: {args.output}")
        print(f"   Progress saved to: {args.progress_file}")
        print(f"   Strategies used: {[s.value for s in crasher.strategies]}")
        print(f"   Max Concurrent: {crasher.max_concurrent}")
        
        # Display strategy performance
        if results.get('breakthrough_analysis', {}).get('strategy_performance'):
            print(f"\n🎯 Strategy Performance:")
            for strategy, perf in results['breakthrough_analysis']['strategy_performance'].items():
                print(f"   {strategy}: {perf['count']} breakthroughs, avg score: {perf['avg_breakthrough_score']:.3f}")
        
        # Display region performance
        if results.get('breakthrough_analysis', {}).get('region_performance'):
            print(f"\n🌍 Region Performance:")
            for region, perf in results['breakthrough_analysis']['region_performance'].items():
                print(f"   {region}: {perf['count']} breakthroughs, avg score: {perf['avg_breakthrough_score']:.3f}")
        
        print(f"\n💡 Tip: Check {args.output} for detailed results and breakthrough analysis!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting gracefully...")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
