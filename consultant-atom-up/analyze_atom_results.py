#!/usr/bin/env python3
"""
Analyze atom testing results and provide comprehensive insights
"""

import json
import os
import statistics
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

class AtomResultsAnalyzer:
    """Analyze atom testing results"""
    
    def __init__(self, results_file: str = "atom_test_results.json", 
                 stats_file: str = "atom_statistics.json"):
        self.results_file = results_file
        self.stats_file = stats_file
        self.results = []
        self.statistics = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load results and statistics data"""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            print(f"✅ Loaded {len(self.results)} test results")
        else:
            print(f"❌ Results file {self.results_file} not found")
            return
        
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                self.statistics = json.load(f)
            print(f"✅ Loaded statistics for {len(self.statistics)} datasets")
        else:
            print(f"⚠️ Statistics file {self.stats_file} not found")
    
    def analyze_overall_performance(self):
        """Analyze overall performance metrics"""
        print("\n" + "="*80)
        print("📊 OVERALL PERFORMANCE ANALYSIS")
        print("="*80)
        
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r['status'] == 'success'])
        failed_tests = len([r for r in self.results if r['status'] == 'failed'])
        error_tests = len([r for r in self.results if r['status'] == 'error'])
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"Errors: {error_tests} ({error_tests/total_tests*100:.1f}%)")
        
        if successful_tests > 0:
            # Performance metrics
            sharpe_ratios = [r['sharpe_ratio'] for r in self.results 
                           if r['status'] == 'success' and r['sharpe_ratio'] is not None]
            returns = [r['returns'] for r in self.results 
                      if r['status'] == 'success' and r['returns'] is not None]
            max_drawdowns = [r['max_drawdown'] for r in self.results 
                           if r['status'] == 'success' and r['max_drawdown'] is not None]
            hit_ratios = [r['hit_ratio'] for r in self.results 
                        if r['status'] == 'success' and r['hit_ratio'] is not None]
            
            if sharpe_ratios:
                print(f"\nSharpe Ratio Statistics:")
                print(f"  Average: {statistics.mean(sharpe_ratios):.3f}")
                print(f"  Median: {statistics.median(sharpe_ratios):.3f}")
                print(f"  Maximum: {max(sharpe_ratios):.3f}")
                print(f"  Minimum: {min(sharpe_ratios):.3f}")
                print(f"  Std Dev: {statistics.stdev(sharpe_ratios):.3f}")
                
                # Performance categories
                excellent = len([s for s in sharpe_ratios if s > 2.0])
                good = len([s for s in sharpe_ratios if 1.0 < s <= 2.0])
                average = len([s for s in sharpe_ratios if 0.0 < s <= 1.0])
                poor = len([s for s in sharpe_ratios if s <= 0.0])
                
                print(f"\nPerformance Categories:")
                print(f"  Excellent (Sharpe > 2.0): {excellent} ({excellent/len(sharpe_ratios)*100:.1f}%)")
                print(f"  Good (1.0 < Sharpe ≤ 2.0): {good} ({good/len(sharpe_ratios)*100:.1f}%)")
                print(f"  Average (0.0 < Sharpe ≤ 1.0): {average} ({average/len(sharpe_ratios)*100:.1f}%)")
                print(f"  Poor (Sharpe ≤ 0.0): {poor} ({poor/len(sharpe_ratios)*100:.1f}%)")
            
            if returns:
                print(f"\nReturns Statistics:")
                print(f"  Average: {statistics.mean(returns):.3f}")
                print(f"  Maximum: {max(returns):.3f}")
                print(f"  Minimum: {min(returns):.3f}")
            
            if max_drawdowns:
                print(f"\nMax Drawdown Statistics:")
                print(f"  Average: {statistics.mean(max_drawdowns):.3f}")
                print(f"  Maximum: {max(max_drawdowns):.3f}")
                print(f"  Minimum: {min(max_drawdowns):.3f}")
    
    def analyze_top_performers(self, top_n: int = 10):
        """Analyze top performing atoms"""
        print(f"\n" + "="*80)
        print(f"🏆 TOP {top_n} PERFORMING ATOMS")
        print("="*80)
        
        successful_results = [r for r in self.results 
                            if r['status'] == 'success' and r['sharpe_ratio'] is not None]
        
        if not successful_results:
            print("No successful results found")
            return
        
        # Sort by Sharpe ratio
        top_results = sorted(successful_results, key=lambda x: x['sharpe_ratio'], reverse=True)[:top_n]
        
        for i, result in enumerate(top_results, 1):
            print(f"\n{i}. {result['expression']}")
            print(f"   Dataset: {result['dataset_name']} ({result['dataset_id']})")
            print(f"   Region: {result['region']}, Universe: {result['universe']}")
            print(f"   Neutralization: {result['neutralization']}, Delay: {result['delay']}")
            print(f"   Sharpe: {result['sharpe_ratio']:.3f}, Returns: {result['returns']:.3f}")
            print(f"   Max DD: {result['max_drawdown']:.3f}, Hit Ratio: {result['hit_ratio']:.3f}")
            print(f"   Alpha ID: {result['atom_id']}")
    
    def analyze_dataset_performance(self):
        """Analyze performance by dataset"""
        print(f"\n" + "="*80)
        print("📈 DATASET PERFORMANCE ANALYSIS")
        print("="*80)
        
        if not self.statistics:
            print("No statistics data available")
            return
        
        # Sort datasets by average Sharpe ratio
        sorted_datasets = sorted(self.statistics.items(), 
                               key=lambda x: x[1]['avg_sharpe'] or -999, reverse=True)
        
        print(f"{'Dataset':<20} {'Tests':<8} {'Success%':<10} {'Avg Sharpe':<12} {'Max Sharpe':<12} {'Best Atom'}")
        print("-" * 100)
        
        for dataset_id, stats in sorted_datasets:
            if stats['avg_sharpe'] is not None:
                best_atom = stats['best_atom'][:30] + "..." if len(stats['best_atom']) > 30 else stats['best_atom']
                print(f"{dataset_id:<20} {stats['total_tests']:<8} {stats['success_rate']:<10.1%} "
                      f"{stats['avg_sharpe']:<12.3f} {stats['max_sharpe']:<12.3f} {best_atom}")
    
    def analyze_operator_effectiveness(self):
        """Analyze effectiveness of different operators"""
        print(f"\n" + "="*80)
        print("🔧 OPERATOR EFFECTIVENESS ANALYSIS")
        print("="*80)
        
        operator_stats = defaultdict(list)
        
        for result in self.results:
            if result['status'] == 'success' and result['sharpe_ratio'] is not None:
                expression = result['expression']
                # Extract operator from expression
                if '(' in expression:
                    operator = expression.split('(')[0]
                    operator_stats[operator].append(result['sharpe_ratio'])
        
        if not operator_stats:
            print("No operator data available")
            return
        
        # Calculate statistics for each operator
        operator_summary = []
        for operator, sharpe_ratios in operator_stats.items():
            if sharpe_ratios:
                operator_summary.append({
                    'operator': operator,
                    'count': len(sharpe_ratios),
                    'avg_sharpe': statistics.mean(sharpe_ratios),
                    'max_sharpe': max(sharpe_ratios),
                    'min_sharpe': min(sharpe_ratios),
                    'std_sharpe': statistics.stdev(sharpe_ratios) if len(sharpe_ratios) > 1 else 0
                })
        
        # Sort by average Sharpe ratio
        operator_summary.sort(key=lambda x: x['avg_sharpe'], reverse=True)
        
        print(f"{'Operator':<15} {'Count':<8} {'Avg Sharpe':<12} {'Max Sharpe':<12} {'Min Sharpe':<12} {'Std Dev'}")
        print("-" * 80)
        
        for op in operator_summary:
            print(f"{op['operator']:<15} {op['count']:<8} {op['avg_sharpe']:<12.3f} "
                  f"{op['max_sharpe']:<12.3f} {op['min_sharpe']:<12.3f} {op['std_sharpe']:<12.3f}")
    
    def analyze_regional_performance(self):
        """Analyze performance by region"""
        print(f"\n" + "="*80)
        print("🌍 REGIONAL PERFORMANCE ANALYSIS")
        print("="*80)
        
        regional_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'sharpe_ratios': []})
        
        for result in self.results:
            region = result['region']
            regional_stats[region]['total'] += 1
            
            if result['status'] == 'success':
                regional_stats[region]['successful'] += 1
                if result['sharpe_ratio'] is not None:
                    regional_stats[region]['sharpe_ratios'].append(result['sharpe_ratio'])
        
        print(f"{'Region':<10} {'Total':<8} {'Success':<8} {'Success%':<10} {'Avg Sharpe':<12} {'Max Sharpe'}")
        print("-" * 70)
        
        for region, stats in regional_stats.items():
            success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            avg_sharpe = statistics.mean(stats['sharpe_ratios']) if stats['sharpe_ratios'] else 0
            max_sharpe = max(stats['sharpe_ratios']) if stats['sharpe_ratios'] else 0
            
            print(f"{region:<10} {stats['total']:<8} {stats['successful']:<8} "
                  f"{success_rate:<10.1%} {avg_sharpe:<12.3f} {max_sharpe:<12.3f}")
    
    def analyze_universe_performance(self):
        """Analyze performance by universe"""
        print(f"\n" + "="*80)
        print("🎯 UNIVERSE PERFORMANCE ANALYSIS")
        print("="*80)
        
        universe_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'sharpe_ratios': []})
        
        for result in self.results:
            universe = result['universe']
            universe_stats[universe]['total'] += 1
            
            if result['status'] == 'success':
                universe_stats[universe]['successful'] += 1
                if result['sharpe_ratio'] is not None:
                    universe_stats[universe]['sharpe_ratios'].append(result['sharpe_ratio'])
        
        print(f"{'Universe':<12} {'Total':<8} {'Success':<8} {'Success%':<10} {'Avg Sharpe':<12} {'Max Sharpe'}")
        print("-" * 75)
        
        for universe, stats in universe_stats.items():
            success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            avg_sharpe = statistics.mean(stats['sharpe_ratios']) if stats['sharpe_ratios'] else 0
            max_sharpe = max(stats['sharpe_ratios']) if stats['sharpe_ratios'] else 0
            
            print(f"{universe:<12} {stats['total']:<8} {stats['successful']:<8} "
                  f"{success_rate:<10.1%} {avg_sharpe:<12.3f} {max_sharpe:<12.3f}")
    
    def analyze_neutralization_impact(self):
        """Analyze impact of different neutralization options"""
        print(f"\n" + "="*80)
        print("⚖️ NEUTRALIZATION IMPACT ANALYSIS")
        print("="*80)
        
        neutralization_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'sharpe_ratios': []})
        
        for result in self.results:
            neutralization = result['neutralization']
            neutralization_stats[neutralization]['total'] += 1
            
            if result['status'] == 'success':
                neutralization_stats[neutralization]['successful'] += 1
                if result['sharpe_ratio'] is not None:
                    neutralization_stats[neutralization]['sharpe_ratios'].append(result['sharpe_ratio'])
        
        print(f"{'Neutralization':<15} {'Total':<8} {'Success':<8} {'Success%':<10} {'Avg Sharpe':<12} {'Max Sharpe'}")
        print("-" * 80)
        
        for neutralization, stats in neutralization_stats.items():
            success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            avg_sharpe = statistics.mean(stats['sharpe_ratios']) if stats['sharpe_ratios'] else 0
            max_sharpe = max(stats['sharpe_ratios']) if stats['sharpe_ratios'] else 0
            
            print(f"{neutralization:<15} {stats['total']:<8} {stats['successful']:<8} "
                  f"{success_rate:<10.1%} {avg_sharpe:<12.3f} {max_sharpe:<12.3f}")
    
    def generate_insights(self):
        """Generate key insights and recommendations"""
        print(f"\n" + "="*80)
        print("💡 KEY INSIGHTS AND RECOMMENDATIONS")
        print("="*80)
        
        if not self.results:
            print("No data available for insights")
            return
        
        # Dataset insights
        if self.statistics:
            best_dataset = max(self.statistics.items(), 
                             key=lambda x: x[1]['avg_sharpe'] or -999)
            print(f"🏆 Best Performing Dataset: {best_dataset[0]} ({best_dataset[1]['dataset_name']})")
            print(f"   Average Sharpe: {best_dataset[1]['avg_sharpe']:.3f}")
            print(f"   Success Rate: {best_dataset[1]['success_rate']:.1%}")
        
        # Operator insights
        operator_stats = defaultdict(list)
        for result in self.results:
            if result['status'] == 'success' and result['sharpe_ratio'] is not None:
                expression = result['expression']
                if '(' in expression:
                    operator = expression.split('(')[0]
                    operator_stats[operator].append(result['sharpe_ratio'])
        
        if operator_stats:
            best_operator = max(operator_stats.items(), key=lambda x: statistics.mean(x[1]))
            print(f"🔧 Most Effective Operator: {best_operator[0]}")
            print(f"   Average Sharpe: {statistics.mean(best_operator[1]):.3f}")
            print(f"   Usage Count: {len(best_operator[1])}")
        
        # Regional insights
        regional_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'sharpe_ratios': []})
        for result in self.results:
            region = result['region']
            regional_stats[region]['total'] += 1
            if result['status'] == 'success' and result['sharpe_ratio'] is not None:
                regional_stats[region]['successful'] += 1
                regional_stats[region]['sharpe_ratios'].append(result['sharpe_ratio'])
        
        if regional_stats:
            best_region = max(regional_stats.items(), 
                            key=lambda x: statistics.mean(x[1]['sharpe_ratios']) if x[1]['sharpe_ratios'] else -999)
            print(f"🌍 Best Performing Region: {best_region[0]}")
            if best_region[1]['sharpe_ratios']:
                print(f"   Average Sharpe: {statistics.mean(best_region[1]['sharpe_ratios']):.3f}")
                print(f"   Success Rate: {best_region[1]['successful']/best_region[1]['total']:.1%}")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        print(f"1. Focus on high-performing datasets identified in the analysis")
        print(f"2. Use the most effective operators for your target datasets")
        print(f"3. Consider regional differences when selecting test parameters")
        print(f"4. Monitor success rates to identify problematic configurations")
        print(f"5. Expand testing on promising combinations for deeper insights")
    
    def run_full_analysis(self):
        """Run complete analysis"""
        print("🔍 Starting comprehensive atom results analysis...")
        
        self.analyze_overall_performance()
        self.analyze_top_performers()
        self.analyze_dataset_performance()
        self.analyze_operator_effectiveness()
        self.analyze_regional_performance()
        self.analyze_universe_performance()
        self.analyze_neutralization_impact()
        self.generate_insights()
        
        print(f"\n✅ Analysis complete!")

def main():
    """Main function"""
    analyzer = AtomResultsAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
