"""
Correlation Checker for analyzing alpha correlations with production alphas
Checks correlation patterns and identifies potential issues.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CorrelationBucket:
    """Data class for correlation bucket information"""
    min_correlation: float
    max_correlation: float
    alpha_count: int

@dataclass
class CorrelationAnalysis:
    """Data class for correlation analysis results"""
    alpha_id: str
    max_correlation: float
    min_correlation: float
    high_correlation_count: int  # > 0.5
    medium_correlation_count: int  # 0.2 to 0.5
    low_correlation_count: int  # -0.2 to 0.2
    negative_correlation_count: int  # < -0.2
    total_production_alphas: int
    correlation_buckets: List[CorrelationBucket]
    risk_level: str  # LOW, MEDIUM, HIGH
    recommendations: List[str]

class CorrelationChecker:
    """Checks and analyzes alpha correlations with production alphas"""
    
    def __init__(self, 
                 high_correlation_threshold: float = 0.5,
                 medium_correlation_threshold: float = 0.2,
                 negative_correlation_threshold: float = -0.2,
                 max_high_correlations: int = 10):
        """
        Initialize the CorrelationChecker with thresholds
        
        Args:
            high_correlation_threshold: Threshold for high correlation (default 0.5)
            medium_correlation_threshold: Threshold for medium correlation (default 0.2)
            negative_correlation_threshold: Threshold for negative correlation (default -0.2)
            max_high_correlations: Maximum acceptable high correlations (default 10)
        """
        self.high_correlation_threshold = high_correlation_threshold
        self.medium_correlation_threshold = medium_correlation_threshold
        self.negative_correlation_threshold = negative_correlation_threshold
        self.max_high_correlations = max_high_correlations
    
    def analyze_correlation_data(self, alpha_id: str, correlation_data: Dict) -> CorrelationAnalysis:
        """
        Analyze correlation data for an alpha
        
        Args:
            alpha_id: The alpha ID being analyzed
            correlation_data: Raw correlation data from API
            
        Returns:
            CorrelationAnalysis object with analysis results
        """
        try:
            # Check if correlation data contains an error
            if correlation_data.get('error'):
                logger.warning(f"Correlation data error for alpha {alpha_id}: {correlation_data['error']}")
                return CorrelationAnalysis(
                    alpha_id=alpha_id,
                    max_correlation=0.0,
                    min_correlation=0.0,
                    high_correlation_count=0,
                    medium_correlation_count=0,
                    low_correlation_count=0,
                    negative_correlation_count=0,
                    total_production_alphas=0,
                    correlation_buckets=[],
                    risk_level="UNKNOWN",
                    recommendations=[f"Could not analyze correlations: {correlation_data['error']}"]
                )
            
            records = correlation_data.get('records', [])
            max_correlation = correlation_data.get('max', 0.0)
            min_correlation = correlation_data.get('min', 0.0)
            
            # Count correlations in different ranges
            high_correlation_count = 0
            medium_correlation_count = 0
            low_correlation_count = 0
            negative_correlation_count = 0
            total_production_alphas = 0
            
            correlation_buckets = []
            
            for record in records:
                min_corr, max_corr, count = record
                total_production_alphas += count
                
                bucket = CorrelationBucket(
                    min_correlation=min_corr,
                    max_correlation=max_corr,
                    alpha_count=count
                )
                correlation_buckets.append(bucket)
                
                # Categorize correlations
                if max_corr >= self.high_correlation_threshold:
                    high_correlation_count += count
                elif max_corr >= self.medium_correlation_threshold:
                    medium_correlation_count += count
                elif max_corr >= self.negative_correlation_threshold:
                    low_correlation_count += count
                else:
                    negative_correlation_count += count
            
            # Determine risk level
            risk_level = self._determine_risk_level(
                high_correlation_count,
                medium_correlation_count,
                total_production_alphas
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                high_correlation_count,
                medium_correlation_count,
                low_correlation_count,
                negative_correlation_count,
                total_production_alphas,
                max_correlation,
                min_correlation
            )
            
            return CorrelationAnalysis(
                alpha_id=alpha_id,
                max_correlation=max_correlation,
                min_correlation=min_correlation,
                high_correlation_count=high_correlation_count,
                medium_correlation_count=medium_correlation_count,
                low_correlation_count=low_correlation_count,
                negative_correlation_count=negative_correlation_count,
                total_production_alphas=total_production_alphas,
                correlation_buckets=correlation_buckets,
                risk_level=risk_level,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing correlation data for alpha {alpha_id}: {e}")
            raise
    
    def _determine_risk_level(self, high_correlations: int, 
                             medium_correlations: int, 
                             total_alphas: int) -> str:
        """
        Determine risk level based on correlation patterns
        
        Args:
            high_correlations: Number of high correlations
            medium_correlations: Number of medium correlations
            total_alphas: Total number of production alphas
            
        Returns:
            Risk level: LOW, MEDIUM, or HIGH
        """
        if total_alphas == 0:
            return "UNKNOWN"
        
        high_correlation_rate = high_correlations / total_alphas
        medium_correlation_rate = medium_correlations / total_alphas
        
        if high_correlations > self.max_high_correlations or high_correlation_rate > 0.1:
            return "HIGH"
        elif high_correlations > 5 or high_correlation_rate > 0.05 or medium_correlation_rate > 0.2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(self, high_correlations: int,
                                 medium_correlations: int,
                                 low_correlations: int,
                                 negative_correlations: int,
                                 total_alphas: int,
                                 max_correlation: float,
                                 min_correlation: float) -> List[str]:
        """
        Generate recommendations based on correlation analysis
        
        Args:
            high_correlations: Number of high correlations
            medium_correlations: Number of medium correlations
            low_correlations: Number of low correlations
            negative_correlations: Number of negative correlations
            total_alphas: Total number of production alphas
            max_correlation: Maximum correlation value
            min_correlation: Minimum correlation value
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if total_alphas == 0:
            recommendations.append("No production alphas found for correlation analysis")
            return recommendations
        
        # High correlation recommendations
        if high_correlations > self.max_high_correlations:
            recommendations.append(f"WARNING: {high_correlations} high correlations (>0.5) found, exceeds limit of {self.max_high_correlations}")
            recommendations.append("Consider modifying the alpha to reduce correlation with existing production alphas")
        
        elif high_correlations > 0:
            recommendations.append(f"CAUTION: {high_correlations} high correlations (>0.5) found")
            recommendations.append("Monitor for potential overfitting or similarity to existing alphas")
        
        # Medium correlation recommendations
        if medium_correlations > 50:
            recommendations.append(f"NOTE: {medium_correlations} medium correlations (0.2-0.5) found")
            recommendations.append("Consider diversifying the alpha strategy")
        
        # Correlation range recommendations
        if max_correlation > 0.8:
            recommendations.append(f"CRITICAL: Maximum correlation of {max_correlation:.3f} is very high")
            recommendations.append("Strongly consider alpha modification or rejection")
        
        elif max_correlation > 0.6:
            recommendations.append(f"WARNING: Maximum correlation of {max_correlation:.3f} is high")
            recommendations.append("Review alpha for uniqueness and originality")
        
        # Distribution recommendations
        if negative_correlations > high_correlations:
            recommendations.append("POSITIVE: More negative correlations than high positive correlations")
            recommendations.append("Alpha may provide good diversification benefits")
        
        # Overall assessment
        if high_correlations == 0 and medium_correlations < 20:
            recommendations.append("GOOD: Low correlation with production alphas")
            recommendations.append("Alpha appears to be unique and potentially valuable")
        
        return recommendations
    
    def check_multiple_alphas(self, alpha_correlations: Dict[str, Dict]) -> Dict[str, CorrelationAnalysis]:
        """
        Check correlations for multiple alphas
        
        Args:
            alpha_correlations: Dictionary mapping alpha_id to correlation data
            
        Returns:
            Dictionary mapping alpha_id to CorrelationAnalysis
        """
        results = {}
        
        for alpha_id, correlation_data in alpha_correlations.items():
            try:
                analysis = self.analyze_correlation_data(alpha_id, correlation_data)
                results[alpha_id] = analysis
                logger.info(f"Analyzed correlations for alpha {alpha_id}: {analysis.risk_level} risk")
            except Exception as e:
                logger.error(f"Error analyzing correlations for alpha {alpha_id}: {e}")
                continue
        
        return results
    
    def generate_correlation_report(self, analyses: Dict[str, CorrelationAnalysis]) -> Dict:
        """
        Generate a comprehensive correlation report
        
        Args:
            analyses: Dictionary of correlation analyses
            
        Returns:
            Dictionary containing summary report
        """
        if not analyses:
            return {"error": "No correlation analyses provided"}
        
        total_alphas = len(analyses)
        risk_distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "UNKNOWN": 0}
        
        total_high_correlations = 0
        total_medium_correlations = 0
        total_production_alphas = 0
        max_correlations = []
        min_correlations = []
        
        for analysis in analyses.values():
            risk_distribution[analysis.risk_level] += 1
            total_high_correlations += analysis.high_correlation_count
            total_medium_correlations += analysis.medium_correlation_count
            total_production_alphas += analysis.total_production_alphas
            max_correlations.append(analysis.max_correlation)
            min_correlations.append(analysis.min_correlation)
        
        # Calculate statistics
        avg_max_correlation = sum(max_correlations) / len(max_correlations) if max_correlations else 0
        avg_min_correlation = sum(min_correlations) / len(min_correlations) if min_correlations else 0
        overall_max_correlation = max(max_correlations) if max_correlations else 0
        overall_min_correlation = min(min_correlations) if min_correlations else 0
        
        # Identify problematic alphas
        high_risk_alphas = [alpha_id for alpha_id, analysis in analyses.items() 
                           if analysis.risk_level == "HIGH"]
        medium_risk_alphas = [alpha_id for alpha_id, analysis in analyses.items() 
                             if analysis.risk_level == "MEDIUM"]
        
        return {
            "summary": {
                "total_alphas_analyzed": total_alphas,
                "total_production_alphas": total_production_alphas,
                "total_high_correlations": total_high_correlations,
                "total_medium_correlations": total_medium_correlations,
                "risk_distribution": risk_distribution
            },
            "correlation_statistics": {
                "average_max_correlation": avg_max_correlation,
                "average_min_correlation": avg_min_correlation,
                "overall_max_correlation": overall_max_correlation,
                "overall_min_correlation": overall_min_correlation
            },
            "risk_analysis": {
                "high_risk_alphas": high_risk_alphas,
                "medium_risk_alphas": medium_risk_alphas,
                "high_risk_count": len(high_risk_alphas),
                "medium_risk_count": len(medium_risk_alphas),
                "low_risk_count": risk_distribution["LOW"]
            },
            "recommendations": {
                "overall_assessment": self._get_overall_assessment(risk_distribution, total_alphas),
                "action_items": self._get_action_items(high_risk_alphas, medium_risk_alphas)
            }
        }
    
    def _get_overall_assessment(self, risk_distribution: Dict, total_alphas: int) -> str:
        """Get overall assessment based on risk distribution"""
        high_risk_rate = risk_distribution["HIGH"] / total_alphas if total_alphas > 0 else 0
        medium_risk_rate = risk_distribution["MEDIUM"] / total_alphas if total_alphas > 0 else 0
        
        if high_risk_rate > 0.3:
            return "CRITICAL: High proportion of alphas with high correlation risk"
        elif high_risk_rate > 0.1 or medium_risk_rate > 0.5:
            return "CAUTION: Significant correlation risks detected"
        elif medium_risk_rate > 0.3:
            return "MODERATE: Some correlation concerns present"
        else:
            return "GOOD: Low correlation risk across alphas"
    
    def _get_action_items(self, high_risk_alphas: List[str], medium_risk_alphas: List[str]) -> List[str]:
        """Get action items based on risk analysis"""
        actions = []
        
        if high_risk_alphas:
            actions.append(f"Review and potentially modify {len(high_risk_alphas)} high-risk alphas")
            actions.append("Consider rejection of alphas with excessive correlations")
        
        if medium_risk_alphas:
            actions.append(f"Monitor {len(medium_risk_alphas)} medium-risk alphas")
            actions.append("Consider diversification strategies for medium-risk alphas")
        
        if not high_risk_alphas and not medium_risk_alphas:
            actions.append("Continue monitoring correlation patterns")
            actions.append("Maintain current alpha development strategies")
        
        return actions

def main():
    """Example usage of CorrelationChecker"""
    # Sample correlation data (from the API response)
    sample_correlation_data = {
        "schema": {
            "name": "prodCorrelation",
            "title": "Prod Correlated",
            "properties": [
                {
                    "name": "min",
                    "title": "Min",
                    "type": "decimal"
                },
                {
                    "name": "max",
                    "title": "Max",
                    "type": "decimal"
                },
                {
                    "name": "alphas",
                    "title": "№ Production Alphas",
                    "type": "integer"
                }
            ]
        },
        "records": [
            [-1.0, -0.9, 0],
            [-0.9, -0.8, 0],
            [-0.8, -0.7, 0],
            [-0.7, -0.6, 0],
            [-0.6, -0.5, 1],
            [-0.5, -0.4, 23],
            [-0.4, -0.3, 599],
            [-0.3, -0.2, 13412],
            [-0.2, -0.1, 108621],
            [-0.1, 0.0, 504917],
            [0.0, 0.1, 717660],
            [0.1, 0.2, 228611],
            [0.2, 0.3, 24438],
            [0.3, 0.4, 2051],
            [0.4, 0.5, 164],
            [0.5, 0.6, 37],
            [0.6, 0.7, 39],
            [0.7, 0.8, 11],
            [0.8, 0.9, 2],
            [0.9, 1, 0]
        ],
        "max": 0.8516,
        "min": -0.5184
    }
    
    # Initialize checker
    checker = CorrelationChecker()
    
    # Analyze correlation data
    analysis = checker.analyze_correlation_data("NZ9g967", sample_correlation_data)
    
    print(f"Correlation Analysis for Alpha {analysis.alpha_id}:")
    print(f"  Risk Level: {analysis.risk_level}")
    print(f"  Max Correlation: {analysis.max_correlation:.3f}")
    print(f"  Min Correlation: {analysis.min_correlation:.3f}")
    print(f"  High Correlations (>0.5): {analysis.high_correlation_count}")
    print(f"  Medium Correlations (0.2-0.5): {analysis.medium_correlation_count}")
    print(f"  Total Production Alphas: {analysis.total_production_alphas}")
    print(f"  Recommendations:")
    for rec in analysis.recommendations:
        print(f"    - {rec}")

if __name__ == "__main__":
    main()
