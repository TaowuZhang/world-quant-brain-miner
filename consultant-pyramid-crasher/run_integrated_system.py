#!/usr/bin/env python3
"""
Run Integrated System - Coordinates Pyramid Crasher with Templates API
- Runs 3 concurrent pyramid cracking simulations alongside consultant-templates-api
- Provides unified orchestration of both systems
- Combines results for comprehensive analysis
"""

import os
import sys
import argparse
import json
import time
import subprocess
import threading
import signal
from datetime import datetime
from typing import Dict, List, Optional

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run Integrated System - Coordinates Pyramid Crasher with Templates API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_integrated_system.py --deepseek-key YOUR_API_KEY
  python run_integrated_system.py --deepseek-key YOUR_API_KEY --iterations 200
  python run_integrated_system.py --deepseek-key YOUR_API_KEY --regions USA EUR --iterations 100
        """
    )
    
    parser.add_argument(
        '--credentials',
        default='credential.txt',
        help='Path to credentials file (default: credential.txt)'
    )
    
    parser.add_argument(
        '--deepseek-key',
        required=True,
        help='DeepSeek API key for template generation'
    )
    
    parser.add_argument(
        '--output',
        default='integrated_system_results.json',
        help='Output filename (default: integrated_system_results.json)'
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
        '--templates-per-region',
        type=int,
        default=10,
        help='Number of templates per region (default: 10)'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=8,
        help='Maximum concurrent simulations (default: 8)'
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
        help='Breakthrough threshold for pyramid cracking (default: 2.0)'
    )
    
    return parser.parse_args()

class IntegratedSystemOrchestrator:
    """Orchestrates both template generation and pyramid cracking systems"""
    
    def __init__(self, args):
        self.args = args
        self.template_process = None
        self.pyramid_process = None
        self.results = {
            'metadata': {
                'started_at': datetime.now().isoformat(),
                'regions': args.regions or ['USA', 'GLB', 'EUR', 'ASI', 'CHN'],
                'iterations': args.iterations,
                'templates_per_region': args.templates_per_region,
                'max_concurrent': args.max_concurrent,
                'breakthrough_threshold': args.breakthrough_threshold
            },
            'template_results': {},
            'pyramid_results': {},
            'integrated_analysis': {}
        }
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def run_integrated_system(self):
        """Run the integrated system with both template generation and pyramid cracking"""
        print("🚀 Starting Integrated System...")
        print("🎯 Template Generation: 5 concurrent slots")
        print("💥 Pyramid Cracking: 3 concurrent slots")
        print("🌍 Regions:", self.args.regions or "All")
        print("🔄 Iterations:", self.args.iterations)
        print("=" * 80)
        
        try:
            # Start template generation process
            self._start_template_generation()
            
            # Start pyramid cracking process
            self._start_pyramid_cracking()
            
            # Monitor both processes
            self._monitor_processes()
            
            # Collect results
            self._collect_results()
            
            # Generate integrated analysis
            self._generate_integrated_analysis()
            
            # Save final results
            self._save_results()
            
            # Print final summary
            self._print_final_summary()
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user. Stopping gracefully...")
            self._stop_processes()
        except Exception as e:
            print(f"❌ Error: {e}")
            self._stop_processes()
            raise
    
    def _start_template_generation(self):
        """Start the template generation process"""
        print("🚀 Starting Template Generation Process...")
        
        # Build command for template generation
        cmd = [
            'python', 'enhanced_template_generator_v2.py',
            '--credentials', self.args.credentials,
            '--deepseek-key', self.args.deepseek_key,
            '--output', 'template_results.json',
            '--progress-file', 'template_progress.json',
            '--templates-per-region', str(self.args.templates_per_region),
            '--max-concurrent', '5'  # 5 slots for template generation
        ]
        
        if self.args.regions:
            cmd.extend(['--regions'] + self.args.regions)
        
        if self.args.resume:
            cmd.append('--resume')
        
        # Start process
        self.template_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("✅ Template generation process started")
    
    def _start_pyramid_cracking(self):
        """Start the pyramid cracking process"""
        print("💥 Starting Pyramid Cracking Process...")
        
        # Build command for pyramid cracking
        cmd = [
            'python', 'run_pyramid_crasher.py',
            '--credentials', self.args.credentials,
            '--output', 'pyramid_results.json',
            '--progress-file', 'pyramid_progress.json',
            '--iterations', str(self.args.iterations),
            '--max-concurrent', '3',  # 3 slots for pyramid cracking
            '--breakthrough-threshold', str(self.args.breakthrough_threshold)
        ]
        
        if self.args.regions:
            cmd.extend(['--regions'] + self.args.regions)
        
        if self.args.resume:
            cmd.append('--resume')
        
        # Start process
        self.pyramid_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("✅ Pyramid cracking process started")
    
    def _monitor_processes(self):
        """Monitor both processes and display progress"""
        print("\n🔄 Monitoring processes...")
        print("=" * 80)
        
        start_time = time.time()
        
        while True:
            # Check if processes are still running
            template_running = self.template_process.poll() is None
            pyramid_running = self.pyramid_process.poll() is None
            
            if not template_running and not pyramid_running:
                break
            
            # Display status
            elapsed = time.time() - start_time
            status = []
            
            if template_running:
                status.append("📊 Template Generation: Running")
            else:
                status.append("📊 Template Generation: Completed")
            
            if pyramid_running:
                status.append("💥 Pyramid Cracking: Running")
            else:
                status.append("💥 Pyramid Cracking: Completed")
            
            print(f"\r⏱️  Elapsed: {elapsed:.1f}s | {' | '.join(status)}", end="")
            sys.stdout.flush()
            
            time.sleep(1)
        
        print("\n✅ All processes completed")
    
    def _collect_results(self):
        """Collect results from both processes"""
        print("\n📊 Collecting results...")
        
        # Collect template results
        if os.path.exists('template_results.json'):
            try:
                with open('template_results.json', 'r') as f:
                    self.results['template_results'] = json.load(f)
                print("✅ Template results collected")
            except Exception as e:
                print(f"⚠️  Error loading template results: {e}")
        
        # Collect pyramid results
        if os.path.exists('pyramid_results.json'):
            try:
                with open('pyramid_results.json', 'r') as f:
                    self.results['pyramid_results'] = json.load(f)
                print("✅ Pyramid results collected")
            except Exception as e:
                print(f"⚠️  Error loading pyramid results: {e}")
    
    def _generate_integrated_analysis(self):
        """Generate integrated analysis combining both systems"""
        print("\n🔍 Generating integrated analysis...")
        
        analysis = {
            'total_simulations': 0,
            'successful_simulations': 0,
            'breakthrough_simulations': 0,
            'template_performance': {},
            'pyramid_performance': {},
            'region_performance': {},
            'strategy_performance': {},
            'best_results': []
        }
        
        # Analyze template results
        if 'template_results' in self.results:
            template_results = self.results['template_results']
            if 'simulation_results' in template_results:
                for region, simulations in template_results['simulation_results'].items():
                    successful = [s for s in simulations if s.get('success', False)]
                    analysis['template_performance'][region] = {
                        'total': len(simulations),
                        'successful': len(successful),
                        'success_rate': len(successful) / len(simulations) if simulations else 0,
                        'avg_sharpe': sum(s.get('sharpe', 0) for s in successful) / len(successful) if successful else 0
                    }
                    analysis['total_simulations'] += len(simulations)
                    analysis['successful_simulations'] += len(successful)
        
        # Analyze pyramid results
        if 'pyramid_results' in self.results:
            pyramid_results = self.results['pyramid_results']
            if 'pyramid_results' in pyramid_results:
                for region, simulations in pyramid_results['pyramid_results'].items():
                    successful = [s for s in simulations if s.get('success', False)]
                    breakthroughs = [s for s in successful if s.get('breakthrough_score', 0) >= self.args.breakthrough_threshold]
                    
                    analysis['pyramid_performance'][region] = {
                        'total': len(simulations),
                        'successful': len(successful),
                        'breakthroughs': len(breakthroughs),
                        'success_rate': len(successful) / len(simulations) if simulations else 0,
                        'breakthrough_rate': len(breakthroughs) / len(successful) if successful else 0,
                        'avg_sharpe': sum(s.get('sharpe', 0) for s in successful) / len(successful) if successful else 0,
                        'avg_breakthrough_score': sum(s.get('breakthrough_score', 0) for s in breakthroughs) / len(breakthroughs) if breakthroughs else 0
                    }
                    analysis['total_simulations'] += len(simulations)
                    analysis['successful_simulations'] += len(successful)
                    analysis['breakthrough_simulations'] += len(breakthroughs)
        
        # Calculate overall performance
        analysis['overall_success_rate'] = analysis['successful_simulations'] / analysis['total_simulations'] if analysis['total_simulations'] > 0 else 0
        analysis['overall_breakthrough_rate'] = analysis['breakthrough_simulations'] / analysis['successful_simulations'] if analysis['successful_simulations'] > 0 else 0
        
        self.results['integrated_analysis'] = analysis
        print("✅ Integrated analysis generated")
    
    def _save_results(self):
        """Save final results"""
        print("\n💾 Saving results...")
        
        # Add completion metadata
        self.results['metadata']['completed_at'] = datetime.now().isoformat()
        self.results['metadata']['total_runtime'] = time.time() - time.mktime(datetime.now().timetuple())
        
        # Save integrated results
        with open(self.args.output, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Results saved to {self.args.output}")
    
    def _print_final_summary(self):
        """Print final summary of results"""
        print(f"\n{'='*70}")
        print("🎉 INTEGRATED SYSTEM COMPLETE!")
        print(f"{'='*70}")
        
        analysis = self.results.get('integrated_analysis', {})
        
        print(f"📊 Final Statistics:")
        print(f"   Total simulations: {analysis.get('total_simulations', 0)}")
        print(f"   Successful simulations: {analysis.get('successful_simulations', 0)}")
        print(f"   Breakthrough simulations: {analysis.get('breakthrough_simulations', 0)}")
        print(f"   Overall success rate: {analysis.get('overall_success_rate', 0)*100:.1f}%")
        print(f"   Overall breakthrough rate: {analysis.get('overall_breakthrough_rate', 0)*100:.1f}%")
        
        # Template performance
        if analysis.get('template_performance'):
            print(f"\n📊 Template Performance:")
            for region, perf in analysis['template_performance'].items():
                print(f"   {region}: {perf['successful']}/{perf['total']} successful ({perf['success_rate']*100:.1f}%)")
        
        # Pyramid performance
        if analysis.get('pyramid_performance'):
            print(f"\n💥 Pyramid Performance:")
            for region, perf in analysis['pyramid_performance'].items():
                print(f"   {region}: {perf['successful']}/{perf['total']} successful, {perf['breakthroughs']} breakthroughs ({perf['breakthrough_rate']*100:.1f}%)")
        
        print(f"\n💡 Detailed results saved to: {self.args.output}")
        print(f"📁 Template results: template_results.json")
        print(f"📁 Pyramid results: pyramid_results.json")
    
    def _stop_processes(self):
        """Stop both processes gracefully"""
        if self.template_process and self.template_process.poll() is None:
            print("\n🛑 Stopping template generation process...")
            self.template_process.terminate()
            self.template_process.wait()
        
        if self.pyramid_process and self.pyramid_process.poll() is None:
            print("🛑 Stopping pyramid cracking process...")
            self.pyramid_process.terminate()
            self.pyramid_process.wait()
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print(f"\n🛑 Received signal {signum}. Stopping gracefully...")
        self._stop_processes()
        sys.exit(0)

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Check if credentials file exists
    if not os.path.exists(args.credentials):
        print("Error: credential.txt file not found!")
        print("Please create a credential.txt file with your WorldQuant Brain credentials in JSON format:")
        print('["username", "password"]')
        return 1
    
    # Check if we're in the right directory
    if not os.path.exists('enhanced_template_generator_v2.py'):
        print("Error: enhanced_template_generator_v2.py not found!")
        print("Please run this script from the consultant-templates-api directory")
        return 1
    
    if not os.path.exists('run_pyramid_crasher.py'):
        print("Error: run_pyramid_crasher.py not found!")
        print("Please run this script from the consultant-pyramid-crasher directory")
        return 1
    
    try:
        # Create orchestrator
        orchestrator = IntegratedSystemOrchestrator(args)
        
        # Run integrated system
        orchestrator.run_integrated_system()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
