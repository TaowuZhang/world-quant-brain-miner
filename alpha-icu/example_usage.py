"""
Example usage of Alpha ICU system
Demonstrates how to use the system programmatically.
"""

from main import AlphaICU
import json

def example_basic_analysis():
    """Example of basic alpha analysis"""
    print("=" * 60)
    print("EXAMPLE: Basic Alpha Analysis")
    print("=" * 60)
    
    try:
        # Initialize Alpha ICU
        alpha_icu = AlphaICU()
        
        # Run analysis for last 3 days, limit to 20 alphas for demo
        results = alpha_icu.run_full_analysis(
            days_back=3,
            max_alphas=20,
            check_correlations=True,
            save_results=False  # Don't save for demo
        )
        
        # Print summary
        print(f"Total alphas processed: {results['summary']['total_alphas_fetched']}")
        print(f"Successful alphas: {results['summary']['successful_alphas']}")
        print(f"Success rate: {results['summary']['success_rate']:.1%}")
        
        # Show top 5 performers
        if results.get("successful_alphas"):
            print("\nTop 5 performers:")
            top_performers = alpha_icu.get_top_performers(days_back=3, top_n=5, sort_by='sharpe')
            for i, alpha in enumerate(top_performers, 1):
                print(f"{i}. {alpha['alpha_id']} - Sharpe: {alpha['sharpe']:.2f}, "
                      f"Fitness: {alpha['fitness']:.2f}, Returns: {alpha['returns']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"Error in basic analysis: {e}")
        return None

def example_correlation_analysis():
    """Example of correlation analysis only"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Correlation Analysis")
    print("=" * 60)
    
    try:
        # Initialize Alpha ICU
        alpha_icu = AlphaICU()
        
        # Get successful alphas first
        alphas = alpha_icu._fetch_alphas(days_back=3, max_alphas=10, status_filter="UNSUBMITTED,IS_FAIL")
        successful_alphas, _ = alpha_icu.analyzer.filter_successful_alphas(alphas)
        
        if successful_alphas:
            print(f"Found {len(successful_alphas)} successful alphas")
            
            # Check correlations for first 3 successful alphas
            correlation_results = alpha_icu._check_correlations(successful_alphas[:3])
            
            if correlation_results.get("report"):
                report = correlation_results["report"]
                print(f"\nCorrelation Analysis Summary:")
                print(f"  Risk distribution: {report['summary']['risk_distribution']}")
                print(f"  High risk alphas: {report['risk_analysis']['high_risk_count']}")
                print(f"  Medium risk alphas: {report['risk_analysis']['medium_risk_count']}")
                
                # Show recommendations
                if report.get("recommendations", {}).get("action_items"):
                    print(f"\nRecommendations:")
                    for action in report["recommendations"]["action_items"]:
                        print(f"  - {action}")
        else:
            print("No successful alphas found for correlation analysis")
            
    except Exception as e:
        print(f"Error in correlation analysis: {e}")

def example_custom_criteria():
    """Example with custom success criteria"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Custom Success Criteria")
    print("=" * 60)
    
    try:
        # Initialize with custom criteria
        from alpha_analyzer import AlphaAnalyzer
        
        # Create analyzer with stricter criteria
        strict_analyzer = AlphaAnalyzer(
            min_sharpe=2.0,      # Higher Sharpe requirement
            min_fitness=1.5,     # Higher fitness requirement
            max_drawdown=0.3,    # Lower drawdown tolerance
            min_turnover=0.01,
            max_turnover=0.5,    # Lower turnover tolerance
            min_returns=0.08     # Higher returns requirement
        )
        
        # Initialize Alpha ICU
        alpha_icu = AlphaICU()
        
        # Fetch alphas
        alphas = alpha_icu._fetch_alphas(days_back=3, max_alphas=50, status_filter="UNSUBMITTED,IS_FAIL")
        
        # Analyze with strict criteria
        successful_alphas, unsuccessful_alphas = strict_analyzer.filter_successful_alphas(alphas)
        
        print(f"Total alphas: {len(alphas)}")
        print(f"Successful with strict criteria: {len(successful_alphas)}")
        print(f"Success rate: {len(successful_alphas) / len(alphas):.1%}")
        
        if successful_alphas:
            print("\nAlphas meeting strict criteria:")
            for alpha in successful_alphas[:5]:  # Show first 5
                print(f"  {alpha.alpha_id} - Sharpe: {alpha.sharpe:.2f}, "
                      f"Fitness: {alpha.fitness:.2f}, Returns: {alpha.returns:.3f}")
        
    except Exception as e:
        print(f"Error in custom criteria example: {e}")

def example_individual_alpha_analysis():
    """Example of analyzing individual alpha"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Individual Alpha Analysis")
    print("=" * 60)
    
    try:
        # Initialize Alpha ICU
        alpha_icu = AlphaICU()
        
        # Fetch a few alphas
        alphas = alpha_icu._fetch_alphas(days_back=3, max_alphas=5, status_filter="UNSUBMITTED,IS_FAIL")
        
        if alphas:
            # Analyze first alpha in detail
            first_alpha = alphas[0]
            metrics = alpha_icu.analyzer.extract_alpha_metrics(first_alpha)
            is_successful, reasons = alpha_icu.analyzer.is_successful_alpha(metrics)
            
            print(f"Analyzing alpha: {metrics.alpha_id}")
            print(f"Code: {metrics.code}")
            print(f"Region: {metrics.region}, Universe: {metrics.universe}")
            print(f"Sharpe: {metrics.sharpe:.2f}")
            print(f"Fitness: {metrics.fitness:.2f}")
            print(f"Returns: {metrics.returns:.3f}")
            print(f"Turnover: {metrics.turnover:.3f}")
            print(f"Drawdown: {metrics.drawdown:.3f}")
            print(f"Is successful: {is_successful}")
            
            if reasons:
                print(f"Reasons: {', '.join(reasons)}")
            
            # Check correlations if successful
            if is_successful:
                try:
                    correlation_data = alpha_icu.fetcher.get_correlation_data(metrics.alpha_id)
                    analysis = alpha_icu.correlation_checker.analyze_correlation_data(metrics.alpha_id, correlation_data)
                    
                    print(f"\nCorrelation Analysis:")
                    print(f"  Risk level: {analysis.risk_level}")
                    print(f"  Max correlation: {analysis.max_correlation:.3f}")
                    print(f"  High correlations: {analysis.high_correlation_count}")
                    print(f"  Recommendations:")
                    for rec in analysis.recommendations[:3]:  # Show first 3
                        print(f"    - {rec}")
                        
                except Exception as e:
                    print(f"Could not get correlation data: {e}")
        else:
            print("No alphas found for individual analysis")
            
    except Exception as e:
        print(f"Error in individual alpha analysis: {e}")

def main():
    """Run all examples"""
    print("Alpha ICU - Example Usage")
    print("This script demonstrates various ways to use the Alpha ICU system.")
    print("Make sure you have set up your credentials in credential.txt")
    print()
    
    # Run examples
    example_basic_analysis()
    example_correlation_analysis()
    example_custom_criteria()
    example_individual_alpha_analysis()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("Check the generated files and logs for more details.")
    print("=" * 60)

if __name__ == "__main__":
    main()
