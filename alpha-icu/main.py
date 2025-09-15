"""
Alpha ICU - Main Orchestrator
Coordinates alpha fetching, analysis, and correlation checking from WorldQuant Brain API.
"""

import json
import logging
import argparse
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

from alpha_fetcher import AlphaFetcher
from alpha_analyzer import AlphaAnalyzer, AlphaMetrics
from correlation_checker import CorrelationChecker, CorrelationAnalysis

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpha_icu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlphaICU:
    """Main orchestrator for Alpha ICU system"""
    
    def __init__(self, credential_file: str = "credential.txt"):
        """
        Initialize Alpha ICU system
        
        Args:
            credential_file: Path to credential file
        """
        self.credential_file = credential_file
        self.fetcher = None
        self.analyzer = None
        self.correlation_checker = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing Alpha ICU components...")
            
            # Initialize fetcher
            self.fetcher = AlphaFetcher(self.credential_file)
            logger.info("✓ Alpha Fetcher initialized")
            
            # Initialize analyzer with success criteria
            self.analyzer = AlphaAnalyzer(
                min_sharpe=1.2,  # New requirement: Sharpe > 1.2
                min_margin=0.0005,  # New requirement: Margin > 8 bps
                max_prod_correlation=0.7  # Hard constraint: cannot submit if > 0.7
            )
            logger.info("✓ Alpha Analyzer initialized")
            
            # Initialize correlation checker
            self.correlation_checker = CorrelationChecker(
                high_correlation_threshold=0.5,
                medium_correlation_threshold=0.2,
                negative_correlation_threshold=-0.2,
                max_high_correlations=1000  # More realistic limit for WorldQuant Brain
            )
            logger.info("✓ Correlation Checker initialized")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run_full_analysis(self, 
                         days_back: int = 3,
                         max_alphas: Optional[int] = None,
                         status_filter: str = "UNSUBMITTED,IS_FAIL",
                         check_correlations: bool = True,
                         save_results: bool = True) -> Dict:
        """
        Run full alpha analysis pipeline
        
        Args:
            days_back: Number of days back to fetch alphas
            max_alphas: Maximum number of alphas to process (None for all)
            status_filter: Status filter for alphas
            check_correlations: Whether to check correlations
            save_results: Whether to save results to files
            
        Returns:
            Dictionary containing complete analysis results
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING ALPHA ICU FULL ANALYSIS")
            logger.info("=" * 60)
            
            # Step 1: Fetch alphas
            logger.info("Step 1: Fetching alphas from WorldQuant Brain API...")
            alphas = self._fetch_alphas(days_back, max_alphas, status_filter)
            
            if not alphas:
                logger.warning("No alphas found matching criteria")
                return {"error": "No alphas found"}
            
            # Step 2: Pre-filter alphas for basic criteria (avoid unnecessary correlation checks)
            logger.info("Step 2: Pre-filtering alphas for basic success criteria...")
            candidate_alphas = []
            for alpha_data in alphas:
                try:
                    metrics = self.analyzer.extract_alpha_metrics(alpha_data)
                    # Check basic criteria without correlation
                    is_successful, reasons = self.analyzer.is_successful_alpha(metrics, None)
                    if is_successful:
                        candidate_alphas.append((alpha_data, metrics))
                        logger.info(f"Alpha {metrics.alpha_id} passed basic criteria")
                    else:
                        logger.info(f"Alpha {metrics.alpha_id} failed basic criteria: {', '.join(reasons)}")
                except Exception as e:
                    logger.error(f"Error processing alpha {alpha_data.get('id', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Pre-filtered to {len(candidate_alphas)} candidate alphas out of {len(alphas)} total")
            
            # Step 3: Check correlations only for candidate alphas (if requested)
            correlation_results = {}
            if check_correlations and candidate_alphas:
                logger.info("Step 3: Checking correlations for candidate alphas...")
                candidate_metrics = [metrics for _, metrics in candidate_alphas]
                correlation_results = self._check_correlations(candidate_metrics)
            elif check_correlations:
                logger.info("Step 3: No candidate alphas to check correlations for")
            else:
                logger.info("Step 3: Skipping correlation checks")
            
            # Step 4: Final filtering with correlation constraints
            logger.info("Step 4: Final filtering with correlation constraints...")
            if candidate_alphas:
                candidate_alpha_data = [alpha_data for alpha_data, _ in candidate_alphas]
                successful_alphas, unsuccessful_alphas = self._filter_alphas_with_correlations(candidate_alpha_data, correlation_results)
            else:
                successful_alphas, unsuccessful_alphas = [], []
            
            # Step 5: Generate reports
            logger.info("Step 5: Generating analysis reports...")
            analysis_report = self.analyzer.generate_summary_report(successful_alphas, unsuccessful_alphas)
            
            # Compile final results
            results = {
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "days_back": days_back,
                    "max_alphas": max_alphas,
                    "status_filter": status_filter,
                    "check_correlations": check_correlations
                },
                "summary": {
                    "total_alphas_fetched": len(alphas),
                    "successful_alphas": len(successful_alphas),
                    "unsuccessful_alphas": len(unsuccessful_alphas),
                    "success_rate": len(successful_alphas) / len(alphas) if alphas else 0
                },
                "analysis_report": analysis_report,
                "successful_alphas": [self._alpha_metrics_to_dict(alpha) for alpha in successful_alphas],
                "correlation_results": correlation_results
            }
            
            # Save results if requested
            if save_results:
                self._save_results(results)
            
            logger.info("=" * 60)
            logger.info("ALPHA ICU ANALYSIS COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Total alphas processed: {len(alphas)}")
            logger.info(f"Successful alphas: {len(successful_alphas)}")
            logger.info(f"Success rate: {results['summary']['success_rate']:.1%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in full analysis: {e}")
            raise
    
    def _fetch_alphas(self, days_back: int, max_alphas: Optional[int], status_filter: str) -> List[Dict]:
        """Fetch alphas from API"""
        try:
            # For now, fetch without date filters due to API parameter issues
            # TODO: Fix date parameter format for WorldQuant Brain API
            logger.info(f"Fetching alphas with status filter: {status_filter}")
            
            # Fetch alphas with performance filters
            alphas = self.fetcher.fetch_all_alphas(
                status=status_filter,
                date_from=None,  # Disable date filters for now
                date_to=None,    # Disable date filters for now
                max_alphas=max_alphas,
                min_sharpe=1.2,      # Filter for Sharpe > 1.2
                min_fitness=1.0,     # Filter for fitness > 1
                min_margin=0.0005    # Filter for margin > 0.0005 (5 bps)
            )
            
            logger.info(f"Fetched {len(alphas)} alphas")
            return alphas
            
        except Exception as e:
            logger.error(f"Error fetching alphas: {e}")
            raise
    
    def _filter_alphas_with_correlations(self, alphas: List[Dict], correlation_results: Dict) -> Tuple[List[AlphaMetrics], List[AlphaMetrics]]:
        """
        Filter alphas with correlation constraints applied
        
        Args:
            alphas: List of raw alpha data
            correlation_results: Correlation analysis results
            
        Returns:
            Tuple of (successful_alphas, unsuccessful_alphas)
        """
        successful = []
        unsuccessful = []
        
        # Get correlation analyses if available
        correlation_analyses = correlation_results.get("analyses", {})
        
        for alpha_data in alphas:
            try:
                metrics = self.analyzer.extract_alpha_metrics(alpha_data)
                
                # Get max correlation for this alpha if available
                max_correlation = None
                if metrics.alpha_id in correlation_analyses:
                    max_correlation = correlation_analyses[metrics.alpha_id].get("max_correlation")
                
                # Check success criteria including correlation constraint
                is_successful, reasons = self.analyzer.is_successful_alpha(metrics, max_correlation)
                
                if is_successful:
                    successful.append(metrics)
                    logger.info(f"Alpha {metrics.alpha_id} is SUCCESSFUL")
                else:
                    unsuccessful.append(metrics)
                    logger.info(f"Alpha {metrics.alpha_id} is UNSUCCESSFUL: {', '.join(reasons)}")
                    
            except Exception as e:
                logger.error(f"Error processing alpha {alpha_data.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Filtered {len(successful)} successful alphas out of {len(alphas)} total")
        return successful, unsuccessful
    
    def _check_correlations(self, alpha_metrics: List[AlphaMetrics]) -> Dict:
        """Check correlations for alphas"""
        try:
            correlation_data = {}
            
            # Fetch correlation data for each alpha with rate limiting
            for i, alpha in enumerate(alpha_metrics):
                try:
                    logger.info(f"Fetching correlation data for alpha {alpha.alpha_id}...")
                    data = self.fetcher.get_correlation_data(alpha.alpha_id)
                    correlation_data[alpha.alpha_id] = data
                    
                    # Add delay between requests to avoid rate limiting (except for last request)
                    if i < len(alpha_metrics) - 1:
                        time.sleep(2.0)  # 2 second delay between requests to respect 60/min limit
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch correlation data for alpha {alpha.alpha_id}: {e}")
                    continue
            
            # Analyze correlations
            if correlation_data:
                analyses = self.correlation_checker.check_multiple_alphas(correlation_data)
                report = self.correlation_checker.generate_correlation_report(analyses)
                
                return {
                    "analyses": {alpha_id: self._correlation_analysis_to_dict(analysis) 
                               for alpha_id, analysis in analyses.items()},
                    "report": report
                }
            else:
                return {"error": "No correlation data available"}
                
        except Exception as e:
            logger.error(f"Error checking correlations: {e}")
            return {"error": str(e)}
    
    def _alpha_metrics_to_dict(self, metrics: AlphaMetrics) -> Dict:
        """Convert AlphaMetrics to dictionary for JSON serialization"""
        return {
            "alpha_id": metrics.alpha_id,
            "code": metrics.code,
            "region": metrics.region,
            "universe": metrics.universe,
            "delay": metrics.delay,
            "neutralization": metrics.neutralization,
            "sharpe": metrics.sharpe,
            "fitness": metrics.fitness,
            "returns": metrics.returns,
            "turnover": metrics.turnover,
            "drawdown": metrics.drawdown,
            "pnl": metrics.pnl,
            "long_count": metrics.long_count,
            "short_count": metrics.short_count,
            "checks_passed": metrics.checks_passed,
            "checks_failed": metrics.checks_failed,
            "checks_warning": metrics.checks_warning,
            "checks_pending": metrics.checks_pending,
            "pyramid_matches": metrics.pyramid_matches,
            "theme_matches": metrics.theme_matches,
            "competition_matches": metrics.competition_matches,
            "date_created": metrics.date_created,
            "status": metrics.status
        }
    
    def _correlation_analysis_to_dict(self, analysis: CorrelationAnalysis) -> Dict:
        """Convert CorrelationAnalysis to dictionary for JSON serialization"""
        return {
            "alpha_id": analysis.alpha_id,
            "max_correlation": analysis.max_correlation,
            "min_correlation": analysis.min_correlation,
            "high_correlation_count": analysis.high_correlation_count,
            "medium_correlation_count": analysis.medium_correlation_count,
            "low_correlation_count": analysis.low_correlation_count,
            "negative_correlation_count": analysis.negative_correlation_count,
            "total_production_alphas": analysis.total_production_alphas,
            "correlation_buckets": [
                {
                    "min_correlation": bucket.min_correlation,
                    "max_correlation": bucket.max_correlation,
                    "alpha_count": bucket.alpha_count
                }
                for bucket in analysis.correlation_buckets
            ],
            "risk_level": analysis.risk_level,
            "recommendations": analysis.recommendations
        }
    
    def _save_results(self, results: Dict):
        """Save results to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save complete results
            results_file = f"alpha_icu_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_file}")
            
            # Save successful alphas summary
            if results.get("successful_alphas"):
                summary_file = f"successful_alphas_{timestamp}.json"
                with open(summary_file, 'w') as f:
                    json.dump(results["successful_alphas"], f, indent=2)
                logger.info(f"Successful alphas summary saved to {summary_file}")
            
            # Save correlation report
            if results.get("correlation_results", {}).get("report"):
                correlation_file = f"correlation_report_{timestamp}.json"
                with open(correlation_file, 'w') as f:
                    json.dump(results["correlation_results"]["report"], f, indent=2)
                logger.info(f"Correlation report saved to {correlation_file}")
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def get_top_performers(self, days_back: int = 3, top_n: int = 10, 
                          sort_by: str = 'sharpe', max_alphas: Optional[int] = None) -> List[Dict]:
        """
        Get top performing alphas
        
        Args:
            days_back: Number of days back to fetch alphas
            top_n: Number of top performers to return
            sort_by: Metric to sort by ('sharpe', 'fitness', 'returns', 'pnl')
            max_alphas: Maximum number of alphas to fetch for analysis
            
        Returns:
            List of top performing alpha dictionaries
        """
        try:
            # Fetch and analyze alphas with a reasonable limit
            # If no max_alphas specified, use a reasonable default to avoid fetching too many
            fetch_limit = max_alphas if max_alphas else min(1000, top_n * 20)  # Fetch up to 20x the top_n needed
            
            alphas = self._fetch_alphas(days_back, fetch_limit, "UNSUBMITTED,IS_FAIL")
            successful_alphas, _ = self.analyzer.filter_successful_alphas(alphas)
            
            # Get top performers
            top_performers = self.analyzer.get_top_performers(successful_alphas, top_n, sort_by)
            
            return [self._alpha_metrics_to_dict(alpha) for alpha in top_performers]
            
        except Exception as e:
            logger.error(f"Error getting top performers: {e}")
            raise

def main():
    """Main entry point for Alpha ICU"""
    parser = argparse.ArgumentParser(description="Alpha ICU - WorldQuant Brain Alpha Analysis")
    parser.add_argument("--days", type=int, default=3, help="Days back to fetch alphas (default: 3)")
    parser.add_argument("--max-alphas", type=int, help="Maximum number of alphas to process")
    parser.add_argument("--status", default="UNSUBMITTED,IS_FAIL", help="Status filter for alphas")
    parser.add_argument("--no-correlations", action="store_true", help="Skip correlation checks")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top performers to show")
    parser.add_argument("--sort-by", choices=['sharpe', 'fitness', 'returns', 'pnl'], 
                       default='sharpe', help="Sort metric for top performers")
    parser.add_argument("--credential-file", default="credential.txt", help="Credential file path")
    
    args = parser.parse_args()
    
    try:
        # Initialize Alpha ICU
        alpha_icu = AlphaICU(args.credential_file)
        
        # Run full analysis
        results = alpha_icu.run_full_analysis(
            days_back=args.days,
            max_alphas=args.max_alphas,
            status_filter=args.status,
            check_correlations=not args.no_correlations,
            save_results=not args.no_save
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("ALPHA ICU ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total alphas processed: {results['summary']['total_alphas_fetched']}")
        print(f"Successful alphas: {results['summary']['successful_alphas']}")
        print(f"Success rate: {results['summary']['success_rate']:.1%}")
        
        if results.get("successful_alphas"):
            print(f"\nTop {args.top_n} performers (sorted by {args.sort_by}):")
            top_performers = alpha_icu.get_top_performers(args.days, args.top_n, args.sort_by, args.max_alphas)
            for i, alpha in enumerate(top_performers, 1):
                print(f"{i:2d}. {alpha['alpha_id']} - Sharpe: {alpha['sharpe']:.2f}, "
                      f"Fitness: {alpha['fitness']:.2f}, Returns: {alpha['returns']:.3f}")
        
        if results.get("correlation_results", {}).get("report"):
            report = results["correlation_results"]["report"]
            print(f"\nCorrelation Analysis:")
            print(f"  Risk distribution: {report['summary']['risk_distribution']}")
            print(f"  High risk alphas: {report['risk_analysis']['high_risk_count']}")
            print(f"  Medium risk alphas: {report['risk_analysis']['medium_risk_count']}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Alpha ICU analysis failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
