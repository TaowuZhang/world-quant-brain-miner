"""
Alpha Analyzer for filtering and analyzing successful alphas
Extracts successful alphas based on performance metrics and checks.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CheckResult(Enum):
    """Enum for check results"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    PENDING = "PENDING"

@dataclass
class AlphaMetrics:
    """Data class for alpha performance metrics"""
    alpha_id: str
    code: str
    region: str
    universe: str
    delay: int
    neutralization: str
    sharpe: float
    fitness: float
    returns: float
    turnover: float
    drawdown: float
    margin: float
    pnl: int
    long_count: int
    short_count: int
    checks_passed: int
    checks_failed: int
    checks_warning: int
    checks_pending: int
    pyramid_matches: List[str]
    theme_matches: List[str]
    competition_matches: List[str]
    date_created: str
    status: str

class AlphaAnalyzer:
    """Analyzes and filters alphas based on success criteria"""
    
    def __init__(self, 
                 min_sharpe: float = 1.2,
                 min_margin: float = 0.0008,  # 8 bps = 0.0008
                 max_prod_correlation: float = 0.7):
        """
        Initialize the AlphaAnalyzer with success criteria based on API response status
        
        Args:
            min_sharpe: Minimum Sharpe ratio required (1.2)
            min_margin: Minimum margin required (8 bps = 0.0008)
            max_prod_correlation: Maximum production correlation allowed (hard constraint)
        """
        self.min_sharpe = min_sharpe
        self.min_margin = min_margin
        self.max_prod_correlation = max_prod_correlation
    
    def extract_alpha_metrics(self, alpha_data: Dict) -> AlphaMetrics:
        """
        Extract metrics from alpha data
        
        Args:
            alpha_data: Raw alpha data from API
            
        Returns:
            AlphaMetrics object with extracted data
        """
        try:
            # Basic info
            alpha_id = alpha_data.get('id', '')
            code = alpha_data.get('regular', {}).get('code', '')
            date_created = alpha_data.get('dateCreated', '')
            status = alpha_data.get('status', '')
            
            # Settings
            settings = alpha_data.get('settings', {})
            region = settings.get('region', '')
            universe = settings.get('universe', '')
            delay = settings.get('delay', 0)
            neutralization = settings.get('neutralization', '')
            
            # IS performance metrics
            is_data = alpha_data.get('is', {})
            sharpe = is_data.get('sharpe', 0.0)
            fitness = is_data.get('fitness', 0.0)
            returns = is_data.get('returns', 0.0)
            turnover = is_data.get('turnover', 0.0)
            drawdown = is_data.get('drawdown', 0.0)
            margin = is_data.get('margin', 0.0)
            pnl = is_data.get('pnl', 0)
            long_count = is_data.get('longCount', 0)
            short_count = is_data.get('shortCount', 0)
            
            # Analyze checks
            checks = is_data.get('checks', [])
            checks_passed = 0
            checks_failed = 0
            checks_warning = 0
            checks_pending = 0
            
            for check in checks:
                result = check.get('result', '')
                if result == CheckResult.PASS.value:
                    checks_passed += 1
                elif result == CheckResult.FAIL.value:
                    checks_failed += 1
                elif result == CheckResult.WARNING.value:
                    checks_warning += 1
                elif result == CheckResult.PENDING.value:
                    checks_pending += 1
            
            # Extract pyramid matches
            pyramid_matches = []
            for check in checks:
                if check.get('name') == 'MATCHES_PYRAMID' and check.get('result') == 'PASS':
                    pyramids = check.get('pyramids', [])
                    pyramid_matches = [p.get('name', '') for p in pyramids]
                    break
            
            # Extract theme matches
            theme_matches = []
            for check in checks:
                if check.get('name') == 'MATCHES_THEMES':
                    themes = check.get('themes', [])
                    theme_matches = [t.get('name', '') for t in themes]
                    break
            
            # Extract competition matches
            competition_matches = []
            for check in checks:
                if check.get('name') == 'MATCHES_COMPETITION':
                    competitions = check.get('competitions', [])
                    competition_matches = [c.get('name', '') for c in competitions]
                    break
            
            return AlphaMetrics(
                alpha_id=alpha_id,
                code=code,
                region=region,
                universe=universe,
                delay=delay,
                neutralization=neutralization,
                sharpe=sharpe,
                fitness=fitness,
                returns=returns,
                turnover=turnover,
                drawdown=drawdown,
                margin=margin,
                pnl=pnl,
                long_count=long_count,
                short_count=short_count,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                checks_warning=checks_warning,
                checks_pending=checks_pending,
                pyramid_matches=pyramid_matches,
                theme_matches=theme_matches,
                competition_matches=competition_matches,
                date_created=date_created,
                status=status
            )
            
        except Exception as e:
            logger.error(f"Error extracting metrics for alpha {alpha_data.get('id', 'unknown')}: {e}")
            raise
    
    def is_successful_alpha(self, metrics: AlphaMetrics, max_correlation: Optional[float] = None) -> Tuple[bool, List[str]]:
        """
        Determine if an alpha meets success criteria based on API response status
        
        New criteria:
        - FAIL = Reject the alpha
        - WARNING = Accept the alpha (fine)
        - Sharpe > 1.2 and Margin > 8bps as additional filters
        
        Args:
            metrics: AlphaMetrics object to evaluate
            max_correlation: Maximum production correlation (if available)
            
        Returns:
            Tuple of (is_successful, list_of_reasons)
        """
        reasons = []
        
        # Check production correlation constraint (HARD CONSTRAINT - cannot be submitted if > 0.7)
        if max_correlation is not None and max_correlation > self.max_prod_correlation:
            reasons.append(f"Production correlation {max_correlation:.3f} exceeds maximum {self.max_prod_correlation} (CANNOT BE SUBMITTED)")
        
        # Check if any checks FAILED (reject if any FAIL)
        if metrics.checks_failed > 0:
            reasons.append(f"Has {metrics.checks_failed} FAILED checks (rejecting)")
        
        # Check Sharpe ratio (must be > 1.2)
        if metrics.sharpe < self.min_sharpe:
            reasons.append(f"Sharpe ratio {metrics.sharpe:.2f} below minimum {self.min_sharpe}")
        
        # Check margin (must be > 8 bps = 0.0008)
        if metrics.margin < self.min_margin:
            reasons.append(f"Margin {metrics.margin:.6f} below minimum {self.min_margin} (8 bps)")
        
        # WARNING checks are fine - we don't reject for warnings
        
        is_successful = len(reasons) == 0
        return is_successful, reasons
    
    def filter_successful_alphas(self, alphas: List[Dict]) -> Tuple[List[AlphaMetrics], List[AlphaMetrics]]:
        """
        Filter alphas into successful and unsuccessful categories
        
        Args:
            alphas: List of raw alpha data from API
            
        Returns:
            Tuple of (successful_alphas, unsuccessful_alphas)
        """
        successful = []
        unsuccessful = []
        
        for alpha_data in alphas:
            try:
                metrics = self.extract_alpha_metrics(alpha_data)
                is_successful, reasons = self.is_successful_alpha(metrics)
                
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
    
    def get_top_performers(self, successful_alphas: List[AlphaMetrics], 
                          top_n: int = 10, 
                          sort_by: str = 'sharpe') -> List[AlphaMetrics]:
        """
        Get top performing alphas
        
        Args:
            successful_alphas: List of successful alpha metrics
            top_n: Number of top performers to return
            sort_by: Metric to sort by ('sharpe', 'fitness', 'returns', 'pnl')
            
        Returns:
            List of top performing alphas
        """
        if sort_by == 'sharpe':
            sorted_alphas = sorted(successful_alphas, key=lambda x: x.sharpe, reverse=True)
        elif sort_by == 'fitness':
            sorted_alphas = sorted(successful_alphas, key=lambda x: x.fitness, reverse=True)
        elif sort_by == 'returns':
            sorted_alphas = sorted(successful_alphas, key=lambda x: x.returns, reverse=True)
        elif sort_by == 'pnl':
            sorted_alphas = sorted(successful_alphas, key=lambda x: x.pnl, reverse=True)
        else:
            raise ValueError(f"Invalid sort_by parameter: {sort_by}")
        
        return sorted_alphas[:top_n]
    
    def generate_summary_report(self, successful_alphas: List[AlphaMetrics], 
                               unsuccessful_alphas: List[AlphaMetrics]) -> Dict:
        """
        Generate a summary report of alpha analysis
        
        Args:
            successful_alphas: List of successful alpha metrics
            unsuccessful_alphas: List of unsuccessful alpha metrics
            
        Returns:
            Dictionary containing summary statistics
        """
        total_alphas = len(successful_alphas) + len(unsuccessful_alphas)
        success_rate = len(successful_alphas) / total_alphas if total_alphas > 0 else 0
        
        # Calculate averages for successful alphas
        if successful_alphas:
            avg_sharpe = sum(a.sharpe for a in successful_alphas) / len(successful_alphas)
            avg_fitness = sum(a.fitness for a in successful_alphas) / len(successful_alphas)
            avg_returns = sum(a.returns for a in successful_alphas) / len(successful_alphas)
            avg_turnover = sum(a.turnover for a in successful_alphas) / len(successful_alphas)
            avg_drawdown = sum(a.drawdown for a in successful_alphas) / len(successful_alphas)
            avg_margin = sum(a.margin for a in successful_alphas) / len(successful_alphas)
            total_pnl = sum(a.pnl for a in successful_alphas)
        else:
            avg_sharpe = avg_fitness = avg_returns = avg_turnover = avg_drawdown = avg_margin = 0
            total_pnl = 0
        
        # Region distribution
        region_dist = {}
        for alpha in successful_alphas:
            region_dist[alpha.region] = region_dist.get(alpha.region, 0) + 1
        
        # Universe distribution
        universe_dist = {}
        for alpha in successful_alphas:
            universe_dist[alpha.universe] = universe_dist.get(alpha.universe, 0) + 1
        
        return {
            "total_alphas": total_alphas,
            "successful_alphas": len(successful_alphas),
            "unsuccessful_alphas": len(unsuccessful_alphas),
            "success_rate": success_rate,
            "average_metrics": {
                "sharpe": avg_sharpe,
                "fitness": avg_fitness,
                "returns": avg_returns,
                "turnover": avg_turnover,
                "drawdown": avg_drawdown,
                "margin": avg_margin,
                "total_pnl": total_pnl
            },
            "region_distribution": region_dist,
            "universe_distribution": universe_dist,
            "criteria_used": {
                "min_sharpe": self.min_sharpe,
                "min_margin": self.min_margin,
                "max_prod_correlation": self.max_prod_correlation,
                "api_response_based": "FAIL=reject, WARNING=accept"
            }
        }

def main():
    """Example usage of AlphaAnalyzer"""
    # Sample alpha data (from the API response)
    sample_alpha = {
        "id": "NZ9g967",
        "type": "REGULAR",
        "author": "OS40510",
        "settings": {
            "instrumentType": "EQUITY",
            "region": "EUR",
            "universe": "TOP2500",
            "delay": 0,
            "decay": 0,
            "neutralization": "NONE",
            "truncation": 0.08,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "OFF",
            "maxTrade": "OFF",
            "language": "FASTEXPR",
            "visualization": False,
            "startDate": "2013-01-20",
            "endDate": "2023-01-20",
            "testPeriod": "P5Y0M0D"
        },
        "regular": {
            "code": "quantile(ts_delta(anl94_find, 20), driver = gaussian)",
            "description": None,
            "operatorCount": 2
        },
        "dateCreated": "2025-09-13T21:32:25-04:00",
        "dateSubmitted": None,
        "dateModified": "2025-09-13T21:32:26-04:00",
        "name": None,
        "favorite": False,
        "hidden": False,
        "color": None,
        "category": None,
        "tags": [],
        "classifications": [
            {
                "id": "DATA_USAGE:SINGLE_DATA_SET",
                "name": "Single Data Set Alpha"
            }
        ],
        "grade": None,
        "stage": "IS",
        "status": "UNSUBMITTED",
        "is": {
            "pnl": 9818739,
            "bookSize": 20000000,
            "longCount": 300,
            "shortCount": 300,
            "turnover": 0.3973,
            "returns": 0.0951,
            "drawdown": 0.0411,
            "margin": 0.000479,
            "sharpe": 2.35,
            "fitness": 1.15,
            "startDate": "2013-01-20",
            "checks": [
                {
                    "name": "LOW_SHARPE",
                    "result": "WARNING",
                    "limit": 2.69,
                    "value": 2.35
                },
                {
                    "name": "LOW_FITNESS",
                    "result": "WARNING",
                    "limit": 1.5,
                    "value": 1.15
                }
            ]
        }
    }
    
    # Initialize analyzer
    analyzer = AlphaAnalyzer()
    
    # Extract metrics
    metrics = analyzer.extract_alpha_metrics(sample_alpha)
    print(f"Extracted metrics for alpha {metrics.alpha_id}:")
    print(f"  Sharpe: {metrics.sharpe}")
    print(f"  Fitness: {metrics.fitness}")
    print(f"  Returns: {metrics.returns}")
    print(f"  Turnover: {metrics.turnover}")
    print(f"  Drawdown: {metrics.drawdown}")
    
    # Check if successful
    is_successful, reasons = analyzer.is_successful_alpha(metrics)
    print(f"  Is successful: {is_successful}")
    if reasons:
        print(f"  Reasons: {', '.join(reasons)}")

if __name__ == "__main__":
    main()
