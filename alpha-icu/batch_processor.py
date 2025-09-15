#!/usr/bin/env python3
"""
Batch Processor for Alpha ICU - Handles large batches with proper rate limiting
"""

import time
import logging
from typing import List, Dict
from alpha_fetcher import AlphaFetcher
from alpha_analyzer import AlphaAnalyzer, AlphaMetrics
from correlation_checker import CorrelationChecker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Processes alphas in batches with proper rate limiting"""
    
    def __init__(self, credential_file: str = "credential.txt"):
        """Initialize the batch processor"""
        self.fetcher = AlphaFetcher(credential_file)
        self.analyzer = AlphaAnalyzer(
            min_sharpe=1.2,
            min_margin=0.0008,
            max_prod_correlation=0.7
        )
        self.correlation_checker = CorrelationChecker(
            high_correlation_threshold=0.5,
            medium_correlation_threshold=0.2,
            max_high_correlations=1000
        )
    
    def process_batch(self, max_alphas: int = 50, batch_size: int = 5, 
                     max_corr_threshold: float = 0.3) -> Dict:
        """
        Process alphas in smaller batches to avoid rate limiting
        
        Args:
            max_alphas: Maximum total alphas to process
            batch_size: Number of alphas to process in each batch
            max_corr_threshold: Maximum correlation threshold for filtering
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Starting batch processing: {max_alphas} alphas in batches of {batch_size}")
        
        # Fetch all alphas first
        logger.info("Fetching alphas...")
        all_alphas = self.fetcher.fetch_all_alphas(
            max_alphas=max_alphas,
            status="UNSUBMITTED,IS_FAIL"
        )
        
        if not all_alphas:
            logger.warning("No alphas found")
            return {"successful_alphas": [], "total_processed": 0}
        
        logger.info(f"Found {len(all_alphas)} alphas, processing in batches...")
        
        successful_alphas = []
        total_processed = 0
        
        # Process in batches
        for i in range(0, len(all_alphas), batch_size):
            batch = all_alphas[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(all_alphas) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} alphas)...")
            
            # Process this batch
            batch_results = self._process_single_batch(batch, max_corr_threshold)
            successful_alphas.extend(batch_results)
            total_processed += len(batch)
            
            # Add delay between batches (except for last batch)
            if i + batch_size < len(all_alphas):
                logger.info("Waiting 5 seconds before next batch...")
                time.sleep(5.0)
        
        logger.info(f"Batch processing complete: {len(successful_alphas)} successful alphas out of {total_processed}")
        
        return {
            "successful_alphas": successful_alphas,
            "total_processed": total_processed,
            "success_rate": len(successful_alphas) / total_processed if total_processed > 0 else 0
        }
    
    def _process_single_batch(self, batch: List[Dict], max_corr_threshold: float) -> List[Dict]:
        """Process a single batch of alphas"""
        batch_results = []
        
        # Check correlations for this batch
        correlation_data = {}
        for i, alpha_data in enumerate(batch):
            alpha_id = alpha_data.get('id', '')
            logger.info(f"  Checking correlations for {alpha_id}...")
            
            try:
                data = self.fetcher.get_correlation_data(alpha_id)
                correlation_data[alpha_id] = data
                
                # Add delay between requests (except for last in batch)
                if i < len(batch) - 1:
                    time.sleep(2.0)
                    
            except Exception as e:
                logger.warning(f"  Failed to fetch correlation data for {alpha_id}: {e}")
                continue
        
        # Analyze correlations
        if correlation_data:
            analyses = self.correlation_checker.check_multiple_alphas(correlation_data)
            
            # Filter alphas
            for alpha_data in batch:
                try:
                    metrics = self.analyzer.extract_alpha_metrics(alpha_data)
                    alpha_id = metrics.alpha_id
                    
                    # Get correlation analysis
                    max_correlation = None
                    if alpha_id in analyses:
                        max_correlation = analyses[alpha_id].max_correlation
                    
                    # Check success criteria
                    is_successful, reasons = self.analyzer.is_successful_alpha(metrics, max_correlation)
                    
                    # Additional filter: low correlation threshold
                    if is_successful and (max_correlation is None or max_correlation <= max_corr_threshold):
                        batch_results.append({
                            'metrics': metrics,
                            'correlation_analysis': analyses.get(alpha_id),
                            'raw_data': alpha_data
                        })
                        logger.info(f"  ✓ {alpha_id} - SUCCESSFUL (Sharpe: {metrics.sharpe:.3f}, Margin: {metrics.margin:.6f}, Corr: {max_correlation:.3f if max_correlation else 'N/A'})")
                    else:
                        logger.info(f"  ✗ {alpha_id} - UNSUCCESSFUL: {', '.join(reasons) if reasons else 'High correlation'}")
                        
                except Exception as e:
                    logger.error(f"  Error processing alpha {alpha_data.get('id', 'unknown')}: {e}")
                    continue
        
        return batch_results

def main():
    """Main function for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process alphas with rate limiting")
    parser.add_argument("--max-alphas", type=int, default=50, help="Maximum alphas to process")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing")
    parser.add_argument("--max-corr", type=float, default=0.3, help="Maximum correlation threshold")
    
    args = parser.parse_args()
    
    processor = BatchProcessor()
    results = processor.process_batch(
        max_alphas=args.max_alphas,
        batch_size=args.batch_size,
        max_corr_threshold=args.max_corr
    )
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING RESULTS")
    print(f"{'='*60}")
    print(f"Total alphas processed: {results['total_processed']}")
    print(f"Successful alphas: {len(results['successful_alphas'])}")
    print(f"Success rate: {results['success_rate']:.1%}")
    
    if results['successful_alphas']:
        print(f"\nTop 5 successful alphas:")
        sorted_alphas = sorted(results['successful_alphas'], 
                             key=lambda x: x['metrics'].sharpe, reverse=True)
        for i, item in enumerate(sorted_alphas[:5], 1):
            metrics = item['metrics']
            corr_analysis = item['correlation_analysis']
            max_corr = corr_analysis.max_correlation if corr_analysis else 0.0
            print(f"{i}. {metrics.alpha_id}: Sharpe={metrics.sharpe:.3f}, "
                  f"Margin={metrics.margin:.6f}, Corr={max_corr:.3f}")

if __name__ == "__main__":
    main()
