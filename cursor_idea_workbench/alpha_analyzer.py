#!/usr/bin/env python3
"""
Alpha Expression Analyzer and Recommender
A tool to analyze WorldQuant alpha expressions and suggest improvements.
"""

import json
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import ast
import sys

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass

# Import vector database query tool
try:
    from query_vector_database import WorldQuantMinerQuery
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    print("Warning: Vector database query tool not available. Field suggestions will be limited.")

class OperatorCategory(Enum):
    ARITHMETIC = "Arithmetic"
    LOGICAL = "Logical"
    TIME_SERIES = "Time Series"
    CROSS_SECTIONAL = "Cross Sectional"
    VECTOR = "Vector"
    TRANSFORMATIONAL = "Transformational"
    GROUP = "Group"

@dataclass
class Operator:
    name: str
    category: str
    definition: str
    description: str
    level: str

@dataclass
class AlphaExpression:
    original: str
    operators: List[str]
    fields: List[str]
    complexity: int
    categories: List[str]
    suggestions: List[str]

@dataclass
class FieldSuggestion:
    field_name: str
    description: str
    category: str
    relevance_score: float
    suggested_combination: str

class AlphaAnalyzer:
    """
    Analyzes alpha expressions and provides improvement suggestions.
    """
    
    def __init__(self, operators_file: str = "__operator__.json"):
        """Initialize the analyzer with operator definitions."""
        self.operators = self._load_operators(operators_file)
        self.operator_names = {op.name for op in self.operators}
        self.operator_dict = {op.name: op for op in self.operators}
        
        # Initialize vector database client if available
        self.vector_db = None
        if VECTOR_DB_AVAILABLE:
            try:
                self.vector_db = WorldQuantMinerQuery()
                print("Vector database connected for field suggestions")
            except Exception as e:
                print(f"Warning: Could not connect to vector database: {e}")
    
    def _load_operators(self, filename: str) -> List[Operator]:
        """Load operator definitions from JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            operators = []
            for op_data in data:
                operator = Operator(
                    name=op_data['name'],
                    category=op_data['category'],
                    definition=op_data['definition'],
                    description=op_data['description'],
                    level=op_data['level']
                )
                operators.append(operator)
            
            return operators
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Using built-in operator list.")
            return self._get_builtin_operators()
        except Exception as e:
            print(f"Error loading operators: {e}")
            return self._get_builtin_operators()
    
    def _get_builtin_operators(self) -> List[Operator]:
        """Fallback built-in operator list."""
        basic_operators = [
            Operator("abs", "Arithmetic", "abs(x)", "Absolute value of x", "ALL"),
            Operator("add", "Arithmetic", "add(x, y)", "x + y", "ALL"),
            Operator("subtract", "Arithmetic", "subtract(x, y)", "x - y", "ALL"),
            Operator("multiply", "Arithmetic", "multiply(x, y)", "x * y", "ALL"),
            Operator("divide", "Arithmetic", "divide(x, y)", "x / y", "ALL"),
            Operator("ts_mean", "Time Series", "ts_mean(x, d)", "Returns average value of x for the past d days", "ALL"),
            Operator("ts_std_dev", "Time Series", "ts_std_dev(x, d)", "Returns standard deviation of x for the past d days", "ALL"),
            Operator("ts_zscore", "Time Series", "ts_zscore(x, d)", "Z-score over past d days", "ALL"),
            Operator("rank", "Cross Sectional", "rank(x)", "Ranks the input among all instruments", "ALL"),
            Operator("normalize", "Cross Sectional", "normalize(x)", "Normalizes values by subtracting mean", "ALL"),
            Operator("winsorize", "Cross Sectional", "winsorize(x, std=4)", "Winsorizes x to limit outliers", "ALL"),
            Operator("hump", "Time Series", "hump(x, hump=0.01)", "Limits changes in input to reduce turnover", "ALL"),
            Operator("if_else", "Logical", "if_else(condition, true_value, false_value)", "Conditional operator", "ALL"),
        ]
        return basic_operators
    
    def parse_expression(self, expression: str) -> AlphaExpression:
        """Parse an alpha expression and extract components."""
        # Clean the expression
        clean_expr = expression.strip()
        
        # Extract operators
        operators = self._extract_operators(clean_expr)
        
        # Extract fields (variables)
        fields = self._extract_fields(clean_expr)
        
        # Calculate complexity
        complexity = self._calculate_complexity(clean_expr, operators)
        
        # Get operator categories
        categories = self._get_operator_categories(operators)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(clean_expr, operators, fields, complexity)
        
        return AlphaExpression(
            original=clean_expr,
            operators=operators,
            fields=fields,
            complexity=complexity,
            categories=categories,
            suggestions=suggestions
        )
    
    def _extract_operators(self, expression: str) -> List[str]:
        """Extract operator names from the expression."""
        operators = []
        
        # Find all function calls (operators)
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, expression)
        
        for match in matches:
            if match in self.operator_names:
                operators.append(match)
        
        return list(set(operators))  # Remove duplicates
    
    def _extract_fields(self, expression: str) -> List[str]:
        """Extract field names (variables) from the expression."""
        fields = []
        
        # Find all identifiers that are not operators
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(pattern, expression)
        
        for match in matches:
            if (match not in self.operator_names and 
                match not in ['true', 'false', 'nan', 'inf'] and
                not match.isdigit()):
                fields.append(match)
        
        return list(set(fields))  # Remove duplicates
    
    def _calculate_complexity(self, expression: str, operators: List[str]) -> int:
        """Calculate complexity score of the expression."""
        complexity = 0
        
        # Base complexity
        complexity += len(operators) * 2
        
        # Nested function calls
        nested_calls = expression.count('(') - expression.count(')')
        complexity += abs(nested_calls) * 3
        
        # Length factor
        complexity += len(expression) // 10
        
        # Category complexity
        category_weights = {
            "Time Series": 3,
            "Cross Sectional": 2,
            "Group": 4,
            "Arithmetic": 1,
            "Logical": 1,
            "Vector": 2,
            "Transformational": 3
        }
        
        for op in operators:
            if op in self.operator_dict:
                category = self.operator_dict[op].category
                complexity += category_weights.get(category, 1)
        
        return complexity
    
    def _get_operator_categories(self, operators: List[str]) -> List[str]:
        """Get categories of operators used in the expression."""
        categories = set()
        for op in operators:
            if op in self.operator_dict:
                categories.add(self.operator_dict[op].category)
        return list(categories)
    
    def _find_related_fields(self, fields: List[str], improvement_request: str = None) -> List[FieldSuggestion]:
        """Find related fields using vector database."""
        if not self.vector_db:
            return []
        
        suggestions = []
        
        for field in fields:
            try:
                # Search for related fields
                results = self.vector_db.search_by_text(field, top_k=5)
                
                for result in results:
                    # Handle different result structures from hosted embeddings vs traditional search
                    if hasattr(result, 'fields'):
                        # Hosted embeddings response format
                        metadata = result.fields
                        field_name = metadata.get('name', '')
                        relevance_score = result._score
                    elif hasattr(result, 'metadata'):
                        # Traditional search response format
                        metadata = result.metadata
                        field_name = metadata.get('name', '')
                        relevance_score = result.score
                    else:
                        continue
                    
                    # Skip if it's the same field
                    if field_name.lower() == field.lower():
                        continue
                    
                    # Calculate relevance based on improvement request
                    if improvement_request:
                        if "turnover" in improvement_request.lower() and "volatility" in field_name.lower():
                            relevance_score += 0.2
                        if "robust" in improvement_request.lower() and "stability" in field_name.lower():
                            relevance_score += 0.2
                        if "risk" in improvement_request.lower() and "risk" in field_name.lower():
                            relevance_score += 0.2
                    
                    # Generate suggested combination
                    suggested_combination = self._generate_field_combination(field, field_name, improvement_request)
                    
                    suggestion = FieldSuggestion(
                        field_name=field_name,
                        description=metadata.get('description', ''),
                        category=metadata.get('category', ''),
                        relevance_score=relevance_score,
                        suggested_combination=suggested_combination
                    )
                    suggestions.append(suggestion)
                
            except Exception as e:
                print(f"Error searching for related fields for {field}: {e}")
        
        # Sort by relevance score
        suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
        return suggestions[:10]  # Return top 10 suggestions
    
    def _generate_field_combination(self, original_field: str, new_field: str, improvement_request: str = None) -> str:
        """Generate a suggested combination of fields."""
        if improvement_request:
            request_lower = improvement_request.lower()
            
            if "turnover" in request_lower:
                return f"subtract({original_field}, {new_field})"
            elif "robust" in request_lower:
                return f"ts_corr({original_field}, {new_field}, 20)"
            elif "risk" in request_lower:
                return f"if_else(greater({original_field}, ts_mean({original_field}, 20)), {new_field}, 0)"
            elif "condition" in request_lower:
                return f"if_else(greater({original_field}, 0), {new_field}, 0)"
            else:
                return f"add({original_field}, {new_field})"
        else:
            return f"add({original_field}, {new_field})"
    
    def _generate_suggestions(self, expression: str, operators: List[str], 
                            fields: List[str], complexity: int) -> List[str]:
        """Generate improvement suggestions for the alpha expression."""
        suggestions = []
        
        # Complexity-based suggestions
        if complexity > 20:
            suggestions.append("Consider breaking down the expression into smaller components for better maintainability")
        
        if complexity > 30:
            suggestions.append("Expression is very complex - consider using intermediate variables or helper functions")
        
        # Operator-specific suggestions
        if "abs" in operators and "subtract" in operators:
            suggestions.append("Consider using 'ts_std_dev' or 'winsorize' to handle outliers instead of just taking absolute differences")
        
        if "ts_mean" in operators or "ts_std_dev" in operators:
            suggestions.append("Consider adding 'hump' operator to reduce turnover and transaction costs")
        
        if "rank" in operators:
            suggestions.append("Consider using 'quantile' for more robust ranking with distribution transformation")
        
        if "normalize" in operators:
            suggestions.append("Consider using 'group_neutralize' if you want to neutralize against specific groups (sector, industry, etc.)")
        
        # Missing optimization suggestions
        if not any(op in operators for op in ["winsorize", "ts_zscore", "zscore"]):
            suggestions.append("Consider adding outlier handling with 'winsorize' or 'zscore' to improve robustness")
        
        if not any(op in operators for op in ["hump", "ts_decay_linear"]):
            suggestions.append("Consider adding 'hump' to reduce turnover or 'ts_decay_linear' for time-weighted calculations")
        
        # Field-specific suggestions
        if any("news" in field.lower() for field in fields):
            suggestions.append("For news-based signals, consider adding time decay with 'ts_decay_linear' to give more weight to recent news")
        
        if any("ret" in field.lower() for field in fields):
            suggestions.append("For return-based signals, consider using 'ts_zscore' for normalization or 'winsorize' for outlier handling")
        
        # Performance suggestions
        if len(operators) > 5:
            suggestions.append("Consider using vectorized operations where possible to improve performance")
        
        # Risk management suggestions
        if not any(op in operators for op in ["scale", "group_scale"]):
            suggestions.append("Consider adding 'scale' operator to control position sizing and risk")
        
        return suggestions
    
    def suggest_improvements(self, expression: str, improvement_request: str = None) -> Dict[str, Any]:
        """Suggest specific improvements for an alpha expression based on custom request."""
        parsed = self.parse_expression(expression)
        
        suggestions = {
            "original_expression": expression,
            "improvement_request": improvement_request,
            "analysis": {
                "operators_used": parsed.operators,
                "fields_used": parsed.fields,
                "complexity_score": parsed.complexity,
                "categories": parsed.categories
            },
            "general_suggestions": parsed.suggestions,
            "specific_improvements": [],
            "field_suggestions": []
        }
        
        # Generate specific improvements based on improvement request
        if improvement_request:
            suggestions["specific_improvements"] = self._suggest_custom_improvements(parsed, improvement_request)
        
        # Find related fields using vector database
        if parsed.fields:
            field_suggestions = self._find_related_fields(parsed.fields, improvement_request)
            suggestions["field_suggestions"] = [
                {
                    "field_name": fs.field_name,
                    "description": fs.description,
                    "category": fs.category,
                    "relevance_score": fs.relevance_score,
                    "suggested_combination": fs.suggested_combination
                }
                for fs in field_suggestions
            ]
        
        return suggestions
    
    def _suggest_custom_improvements(self, parsed: AlphaExpression, improvement_request: str) -> List[str]:
        """Generate custom improvements based on the improvement request."""
        improvements = []
        request_lower = improvement_request.lower()
        
        # Analyze the improvement request and suggest relevant improvements
        if "turnover" in request_lower or "transaction" in request_lower:
            if "hump" not in parsed.operators:
                improvements.append(f"Add 'hump' operator: hump({parsed.original}, 0.01)")
            
            if "ts_decay_linear" not in parsed.operators:
                improvements.append(f"Add time decay for smoother signals: ts_decay_linear({parsed.original}, 20)")
            
            if "ts_delay" not in parsed.operators:
                improvements.append(f"Add delay to reduce noise: ts_delay({parsed.original}, 1)")
        
        if "condition" in request_lower or "filter" in request_lower:
            improvements.append(f"Add conditional logic with 'if_else': if_else(greater({parsed.original}, 0), {parsed.original}, 0)")
            improvements.append(f"Add threshold filtering with 'trade_when': trade_when({parsed.original}, greater({parsed.original}, 0), 0)")
        
        if "robust" in request_lower or "outlier" in request_lower or "stable" in request_lower:
            if "winsorize" not in parsed.operators:
                improvements.append(f"Add outlier handling: winsorize({parsed.original}, 4)")
            
            if "ts_zscore" not in parsed.operators and "zscore" not in parsed.operators:
                improvements.append(f"Add normalization: ts_zscore({parsed.original}, 20)")
        
        if "performance" in request_lower or "efficient" in request_lower:
            if parsed.complexity > 15:
                improvements.append("Break down into intermediate steps for better performance and readability")
            
            if len(parsed.operators) > 3:
                improvements.append("Consider using vectorized operations where possible")
        
        if "risk" in request_lower or "position" in request_lower:
            improvements.append(f"Add position scaling: scale({parsed.original}, 1)")
            improvements.append(f"Add group neutralization: group_neutralize({parsed.original}, sector)")
        
        if "time" in request_lower or "decay" in request_lower:
            if "ts_decay_linear" not in parsed.operators:
                improvements.append(f"Add time-weighted calculations: ts_decay_linear({parsed.original}, 20)")
            
            if "ts_mean" not in parsed.operators:
                improvements.append(f"Add time series smoothing: ts_mean({parsed.original}, 10)")
        
        # If no specific improvements found, provide general ones
        if not improvements:
            improvements = self._suggest_all_improvements(parsed)
        
        return improvements
    
    def _suggest_all_improvements(self, parsed: AlphaExpression) -> List[str]:
        """Suggest all types of improvements."""
        improvements = []
        
        # Turnover reduction
        if "hump" not in parsed.operators:
            improvements.append(f"Add 'hump' operator to reduce turnover: hump({parsed.original}, 0.01)")
        
        if "ts_decay_linear" not in parsed.operators:
            improvements.append(f"Add time decay: ts_decay_linear({parsed.original}, 20)")
        
        # Robustness improvements
        if "winsorize" not in parsed.operators:
            improvements.append(f"Add outlier handling: winsorize({parsed.original}, 4)")
        
        if "ts_zscore" not in parsed.operators and "zscore" not in parsed.operators:
            improvements.append(f"Add normalization: ts_zscore({parsed.original}, 20)")
        
        # Performance optimizations
        if parsed.complexity > 15:
            improvements.append("Break down into intermediate steps for better performance and readability")
        
        return improvements
    
    def generate_improved_expression(self, expression: str, improvement_request: str = None) -> str:
        """Generate an improved version of the alpha expression based on custom request."""
        parsed = self.parse_expression(expression)
        
        # Start with the original expression
        improved = parsed.original
        
        if improvement_request:
            request_lower = improvement_request.lower()
            
            # Apply improvements based on the request
            if "turnover" in request_lower or "transaction" in request_lower:
                if "hump" not in parsed.operators:
                    improved = f"hump({improved}, 0.01)"
                if "ts_decay_linear" not in parsed.operators:
                    improved = f"ts_decay_linear({improved}, 20)"
            
            elif "robust" in request_lower or "outlier" in request_lower:
                if "winsorize" not in parsed.operators:
                    improved = f"winsorize({improved}, 4)"
                if "ts_zscore" not in parsed.operators and "zscore" not in parsed.operators:
                    improved = f"ts_zscore({improved}, 20)"
            
            elif "condition" in request_lower or "filter" in request_lower:
                improved = f"if_else(greater({improved}, 0), {improved}, 0)"
            
            else:
                # Default improvements
                if "winsorize" not in parsed.operators:
                    improved = f"winsorize({improved}, 4)"
                if "hump" not in parsed.operators:
                    improved = f"hump({improved}, 0.01)"
        else:
            # Default improvements
            if "winsorize" not in parsed.operators:
                improved = f"winsorize({improved}, 4)"
            if "hump" not in parsed.operators:
                improved = f"hump({improved}, 0.01)"
        
        return improved
    
    def suggest_multiple_expressions(self, expressions: List[str], improvement_request: str = None) -> Dict[str, Any]:
        """Analyze multiple expressions and suggest combinations."""
        results = {
            "expressions": [],
            "combinations": [],
            "improvement_request": improvement_request
        }
        
        # Analyze each expression
        for expr in expressions:
            expr_result = self.suggest_improvements(expr.strip(), improvement_request)
            results["expressions"].append(expr_result)
        
        # Generate combination suggestions
        if len(expressions) >= 2:
            combinations = self._generate_expression_combinations(expressions, improvement_request)
            results["combinations"] = combinations
        
        return results
    
    def _generate_expression_combinations(self, expressions: List[str], improvement_request: str = None) -> List[str]:
        """Generate combinations of multiple expressions."""
        combinations = []
        
        if len(expressions) == 2:
            expr1, expr2 = expressions[0], expressions[1]
            
            if improvement_request and "condition" in improvement_request.lower():
                # Conditional combination
                combinations.append(f"if_else(greater({expr1}, 0), {expr2}, 0)")
                combinations.append(f"if_else(greater({expr1}, ts_mean({expr1}, 20)), {expr2}, 0)")
            else:
                # Arithmetic combinations
                combinations.append(f"add({expr1}, {expr2})")
                combinations.append(f"subtract({expr1}, {expr2})")
                combinations.append(f"multiply({expr1}, {expr2})")
        
        elif len(expressions) >= 3:
            # For multiple expressions, create a weighted combination
            weighted_expr = f"add({expressions[0]}, add({expressions[1]}, {expressions[2]}))"
            combinations.append(weighted_expr)
            
            # Conditional combination with first expression as condition
            combinations.append(f"if_else(greater({expressions[0]}, 0), add({expressions[1]}, {expressions[2]}), 0)")
        
        return combinations
    
    def print_analysis(self, expression: str, improvement_request: str = None):
        """Print a formatted analysis of the alpha expression."""
        suggestions = self.suggest_improvements(expression, improvement_request)
        
        print("=" * 60)
        print("ALPHA EXPRESSION ANALYSIS")
        print("=" * 60)
        print(f"Original Expression: {suggestions['original_expression']}")
        if suggestions['improvement_request']:
            print(f"Improvement Request: {suggestions['improvement_request']}")
        print()
        
        print("ANALYSIS:")
        print(f"  Operators Used: {', '.join(suggestions['analysis']['operators_used'])}")
        print(f"  Fields Used: {', '.join(suggestions['analysis']['fields_used'])}")
        print(f"  Complexity Score: {suggestions['analysis']['complexity_score']}")
        print(f"  Categories: {', '.join(suggestions['analysis']['categories'])}")
        print()
        
        if suggestions['general_suggestions']:
            print("GENERAL SUGGESTIONS:")
            for i, suggestion in enumerate(suggestions['general_suggestions'], 1):
                print(f"  {i}. {suggestion}")
            print()
        
        if suggestions['specific_improvements']:
            print("SPECIFIC IMPROVEMENTS:")
            for i, improvement in enumerate(suggestions['specific_improvements'], 1):
                print(f"  {i}. {improvement}")
            print()
        
        if suggestions['field_suggestions']:
            print("RELATED FIELD SUGGESTIONS:")
            for i, field_suggestion in enumerate(suggestions['field_suggestions'][:5], 1):
                print(f"  {i}. {field_suggestion['field_name']} (Score: {field_suggestion['relevance_score']:.3f})")
                print(f"     Category: {field_suggestion['category']}")
                print(f"     Description: {field_suggestion['description']}")
                print(f"     Suggested Combination: {field_suggestion['suggested_combination']}")
                print()
        
        # Generate improved expression
        improved = self.generate_improved_expression(expression, improvement_request)
        if improved != expression:
            print("IMPROVED EXPRESSION:")
            print(f"  {improved}")
            print()


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Alpha Expression Analyzer and Recommender")
    parser.add_argument("expression", help="Alpha expression to analyze (use semicolon to separate multiple expressions)")
    parser.add_argument("--improvement-request", help="Custom improvement request (e.g., 'Reduce turnover by adding conditions')")
    parser.add_argument("--operators-file", default="__operator__.json", 
                       help="Path to operators JSON file")
    parser.add_argument("--output-format", choices=["text", "json"], default="text",
                       help="Output format")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AlphaAnalyzer(args.operators_file)
    
    # Check if multiple expressions (separated by semicolon)
    if ";" in args.expression:
        expressions = [expr.strip() for expr in args.expression.split(";")]
        if args.output_format == "json":
            # JSON output for multiple expressions
            suggestions = analyzer.suggest_multiple_expressions(expressions, args.improvement_request)
            print(json.dumps(suggestions, indent=2))
        else:
            # Text output for multiple expressions
            print("=" * 60)
            print("MULTIPLE EXPRESSION ANALYSIS")
            print("=" * 60)
            print(f"Expressions: {expressions}")
            if args.improvement_request:
                print(f"Improvement Request: {args.improvement_request}")
            print()
            
            for i, expr in enumerate(expressions, 1):
                print(f"Expression {i}: {expr}")
                analyzer.print_analysis(expr, args.improvement_request)
            
            # Show combinations
            suggestions = analyzer.suggest_multiple_expressions(expressions, args.improvement_request)
            if suggestions['combinations']:
                print("EXPRESSION COMBINATIONS:")
                for i, combination in enumerate(suggestions['combinations'], 1):
                    print(f"  {i}. {combination}")
                print()
    else:
        # Single expression
        if args.output_format == "json":
            # JSON output
            suggestions = analyzer.suggest_improvements(args.expression, args.improvement_request)
            print(json.dumps(suggestions, indent=2))
        else:
            # Text output
            analyzer.print_analysis(args.expression, args.improvement_request)


if __name__ == "__main__":
    main()
