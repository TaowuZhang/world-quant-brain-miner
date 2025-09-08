#!/usr/bin/env python3
"""
Enhanced runner script v2 for template generation with multi-simulation testing
- Progress saving and resume functionality
- Dynamic result display
- Improved template validation
"""

import os
import sys
from enhanced_template_generator_v2 import EnhancedTemplateGeneratorV2
import json
import time
from dotenv import load_dotenv

# Note: Unicode handling is now managed by the SafeStreamHandler in enhanced_template_generator_v2.py

# Load environment variables from .env file
load_dotenv()

def main():
    # C
    # eck if credentials file exists
    if not os.path.exists('credential.txt'):
        print("Error: credential.txt file not found!")
        print("Please create a credential.txt file with your WorldQuant Brain credentials in JSON format:")
        print('["username", "password"]')
        return 1
    
    # Check if DeepSeek API key is provided
    deepseek_key = os.getenv('DEEPSEEK_API_KEY')
    if not deepseek_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set!")
        print("Please set your DeepSeek API key in .env file:")
        print("DEEPSEEK_API_KEY=your-api-key-here")
        return 1
    
    try:
        print("🚀 Starting Enhanced Template Generator v2 with Continuous Multi-Arm Bandit...")
        print("🔄 This will run indefinitely using explore/exploit strategy")
        print("💡 Use Ctrl+C to stop gracefully")
        print("=" * 80)
        
        # Check for existing progress
        progress_file = "template_progress_v2.json"
        results_file = "enhanced_results_v2.json"
        resume = False
        
        if os.path.exists(progress_file):
            print(f"📁 Found existing progress file: {progress_file}")
            response = input("Do you want to resume from previous progress? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                resume = True
                print("✅ Will resume from previous progress")
            else:
                print("🔄 Starting fresh")
        
        # Initialize generator
        generator = EnhancedTemplateGeneratorV2(
            'credential.txt', 
            deepseek_key, 
            max_concurrent=5,
            progress_file=progress_file,
            results_file=results_file
        )
        
        # Configuration
        regions = ['USA', 'GLB', 'EUR', 'ASI', 'CHN']  # All regions
        templates_per_region = 8  # Increased for better results
        
        print(f"📊 Configuration:")
        print(f"   Regions: {', '.join(regions)}")
        print(f"   Templates per region: {templates_per_region}")
        print(f"   Max concurrent simulations: 5")
        print(f"   Resume mode: {'Yes' if resume else 'No'}")
        print()
        
        # Generate templates and test them
        print("🎯 Starting Template Generation and Multi-Simulation...")
        print("📈 Progress will be displayed dynamically below:")
        print()
        
        results = generator.generate_and_test_templates(regions, templates_per_region, resume, max_iterations=None)
        
        # Save final results
        generator.save_results(results, results_file)
        
        # Print detailed summary
        print(f"\n{'='*80}")
        print("📈 FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        
        total_templates = sum(len(templates) for templates in results['templates'].values())
        total_simulations = sum(len(sims) for sims in results['simulation_results'].values())
        successful_sims = sum(len([s for s in sims if s.get('success', False)]) for sims in results['simulation_results'].values())
        
        print(f"📊 Overall Statistics:")
        print(f"   Total templates generated: {total_templates}")
        print(f"   Total simulations: {total_simulations}")
        print(f"   Successful simulations: {successful_sims}")
        print(f"   Success rate: {successful_sims/total_simulations*100:.1f}%" if total_simulations > 0 else "   Success rate: N/A")
        print(f"   Best Sharpe ratio: {generator.progress_tracker.best_sharpe:.3f}")
        print()
        
        # Per-region breakdown
        print("🌍 Per-Region Results:")
        for region in regions:
            templates = results['templates'].get(region, [])
            simulations = results['simulation_results'].get(region, [])
            successful = len([s for s in simulations if s.get('success', False)])
            
            print(f"   {region}:")
            print(f"     Templates: {len(templates)}")
            print(f"     Simulations: {len(simulations)}")
            print(f"     Successful: {successful}")
            print(f"     Success rate: {successful/len(simulations)*100:.1f}%" if simulations else "     Success rate: N/A")
        print()
        
        # Performance analysis
        analysis = results.get('analysis', {})
        if analysis and analysis.get('performance_metrics'):
            print("📊 Performance Metrics (Successful Simulations):")
            metrics = analysis['performance_metrics']
            
            print(f"   Sharpe Ratio:")
            print(f"     Mean: {metrics['sharpe']['mean']:.3f}")
            print(f"     Std:  {metrics['sharpe']['std']:.3f}")
            print(f"     Range: {metrics['sharpe']['min']:.3f} to {metrics['sharpe']['max']:.3f}")
            
            print(f"   Fitness:")
            print(f"     Mean: {metrics['fitness']['mean']:.3f}")
            print(f"     Std:  {metrics['fitness']['std']:.3f}")
            print(f"     Range: {metrics['fitness']['min']:.3f} to {metrics['fitness']['max']:.3f}")
            
            print(f"   Turnover:")
            print(f"     Mean: {metrics['turnover']['mean']:.3f}")
            print(f"     Std:  {metrics['turnover']['std']:.3f}")
            print(f"     Range: {metrics['turnover']['min']:.3f} to {metrics['turnover']['max']:.3f}")
            print()
        
        # Show best performing templates
        all_simulations = []
        for region_sims in results['simulation_results'].values():
            all_simulations.extend(region_sims)
        
        successful_simulations = [s for s in all_simulations if s.get('success', False)]
        if successful_simulations:
            print("🏆 Top Performing Templates:")
            
            # Sort by Sharpe ratio
            top_templates = sorted(successful_simulations, key=lambda x: x.get('sharpe', 0), reverse=True)[:5]
            
            for i, template in enumerate(top_templates, 1):
                print(f"   {i}. Sharpe: {template.get('sharpe', 0):.3f} | Region: {template.get('region', 'N/A')}")
                print(f"      {template.get('template', 'N/A')}")
                print(f"      Fitness: {template.get('fitness', 0):.3f} | Turnover: {template.get('turnover', 0):.3f}")
                print()
        
        # Show common error patterns
        failed_simulations = [s for s in all_simulations if not s.get('success', False)]
        if failed_simulations:
            print("❌ Common Error Patterns:")
            error_counts = {}
            for sim in failed_simulations:
                error = sim.get('error_message', 'Unknown error')
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"   {error}: {count} occurrences")
            print()
        
        print("✅ Enhanced template generation and testing completed!")
        print(f"📁 Results saved to: {results_file}")
        print(f"📁 Progress saved to: {progress_file}")
        print("\n💡 Next steps:")
        print("   - Review the generated templates in the JSON file")
        print("   - Use the best performing templates as starting points")
        print("   - Run example_usage.py to explore the results")
        print("   - Use --resume flag to continue from where you left off")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Process interrupted by user")
        print("💾 Progress has been saved. You can resume later with --resume flag")
        return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
