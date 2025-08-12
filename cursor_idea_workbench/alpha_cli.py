#!/usr/bin/env python3
"""
Interactive Alpha Expression CLI Tool
A command-line interface for analyzing and improving WorldQuant alpha expressions.
"""

import sys
import json
from alpha_analyzer import AlphaAnalyzer
import argparse
from typing import List

class AlphaCLI:
    """
    Interactive CLI for alpha expression analysis and improvement.
    """
    
    def __init__(self, operators_file: str = "__operator__.json"):
        """Initialize the CLI with the analyzer."""
        self.analyzer = AlphaAnalyzer(operators_file)
    
    def run_interactive(self):
        """Run the interactive CLI mode."""
        print("=" * 60)
        print("ALPHA EXPRESSION ANALYZER & RECOMMENDER")
        print("=" * 60)
        print("Enter 'quit' to exit, 'help' for commands")
        print("Use semicolon (;) to separate multiple expressions")
        print()
        
        while True:
            try:
                # Get alpha expression(s)
                expression = input("Enter alpha expression(s): ").strip()
                
                if expression.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if expression.lower() == 'help':
                    self._show_help()
                    continue
                
                if not expression:
                    continue
                
                # Get improvement request
                print("\nEnter your improvement request (or press Enter for general suggestions):")
                print("Examples:")
                print("  - 'Reduce turnover by adding conditions'")
                print("  - 'Make it more robust against outliers'")
                print("  - 'Add time decay for news signals'")
                print("  - 'Improve performance and efficiency'")
                print("  - 'Add risk management controls'")
                print("  - 'Find related fields for combination'")
                
                improvement_request = input("Improvement request: ").strip()
                
                # Analyze and display results
                print("\n" + "="*60)
                
                # Check if multiple expressions
                if ";" in expression:
                    self._analyze_multiple_expressions(expression, improvement_request)
                else:
                    self._analyze_single_expression(expression, improvement_request)
                
                print("\n" + "-"*60)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")
    
    def _analyze_single_expression(self, expression: str, improvement_request: str):
        """Analyze a single expression."""
        self.analyzer.print_analysis(expression, improvement_request if improvement_request else None)
        
        # Ask if user wants to see improved expression
        show_improved = input("\nShow improved expression? (y/n, default=y): ").strip().lower()
        if show_improved in ['', 'y', 'yes']:
            improved = self.analyzer.generate_improved_expression(expression, improvement_request if improvement_request else None)
            if improved != expression:
                print(f"\nIMPROVED EXPRESSION:")
                print(f"  {improved}")
            else:
                print("\nNo improvements applied (expression already optimal for selected criteria)")
    
    def _analyze_multiple_expressions(self, expressions_str: str, improvement_request: str):
        """Analyze multiple expressions."""
        expressions = [expr.strip() for expr in expressions_str.split(";")]
        
        print("MULTIPLE EXPRESSION ANALYSIS")
        print("=" * 60)
        print(f"Expressions: {expressions}")
        if improvement_request:
            print(f"Improvement Request: {improvement_request}")
        print()
        
        # Analyze each expression
        for i, expr in enumerate(expressions, 1):
            print(f"Expression {i}: {expr}")
            self.analyzer.print_analysis(expr, improvement_request if improvement_request else None)
        
        # Show combinations
        suggestions = self.analyzer.suggest_multiple_expressions(expressions, improvement_request if improvement_request else None)
        if suggestions['combinations']:
            print("EXPRESSION COMBINATIONS:")
            for i, combination in enumerate(suggestions['combinations'], 1):
                print(f"  {i}. {combination}")
            print()
    
    def _show_help(self):
        """Show help information."""
        print("\nHELP:")
        print("  Enter alpha expressions in WorldQuant format, e.g.:")
        print("    -abs(subtract(news_max_up_ret, news_max_dn_ret))")
        print("    ts_mean(close, 20)")
        print("    rank(subtract(high, low))")
        print()
        print("  Multiple expressions (separated by semicolon):")
        print("    ts_mean(close, 20); rank(volume); subtract(high, low)")
        print("    if_else(greater(close, ts_mean(close, 20)), 1, -1); ts_std_dev(volume, 10)")
        print()
        print("  Improvement request examples:")
        print("    - 'Reduce turnover by adding conditions'")
        print("    - 'Make it more robust against outliers'")
        print("    - 'Add time decay for news signals'")
        print("    - 'Improve performance and efficiency'")
        print("    - 'Add risk management controls'")
        print("    - 'Add filtering conditions'")
        print("    - 'Make it more stable over time'")
        print("    - 'Find related fields for combination'")
        print()
    
    def analyze_single(self, expression: str, improvement_request: str = None, output_format: str = "text"):
        """Analyze a single expression and return results."""
        if output_format == "json":
            return self.analyzer.suggest_improvements(expression, improvement_request)
        else:
            self.analyzer.print_analysis(expression, improvement_request)
            return None
    
    def analyze_multiple(self, expressions: List[str], improvement_request: str = None, output_format: str = "text"):
        """Analyze multiple expressions and return results."""
        if output_format == "json":
            return self.analyzer.suggest_multiple_expressions(expressions, improvement_request)
        else:
            for i, expr in enumerate(expressions, 1):
                print(f"\nExpression {i}: {expr}")
                self.analyzer.print_analysis(expr, improvement_request)
            
            # Show combinations
            suggestions = self.analyzer.suggest_multiple_expressions(expressions, improvement_request)
            if suggestions['combinations']:
                print("\nEXPRESSION COMBINATIONS:")
                for i, combination in enumerate(suggestions['combinations'], 1):
                    print(f"  {i}. {combination}")
            return None
    
    def batch_analyze(self, expressions: List[str], improvement_request: str = None):
        """Analyze multiple expressions."""
        results = []
        
        for i, expression in enumerate(expressions, 1):
            print(f"\n{'='*20} Expression {i} {'='*20}")
            result = self.analyzer.suggest_improvements(expression, improvement_request)
            results.append(result)
            
            # Print summary
            print(f"Expression: {expression}")
            if improvement_request:
                print(f"Improvement Request: {improvement_request}")
            print(f"Complexity: {result['analysis']['complexity_score']}")
            print(f"Operators: {', '.join(result['analysis']['operators_used'])}")
            print(f"Suggestions: {len(result['general_suggestions'])} general, {len(result['specific_improvements'])} specific")
            
            # Show field suggestions
            if result['field_suggestions']:
                print(f"Field Suggestions: {len(result['field_suggestions'])} related fields found")
                for j, field_suggestion in enumerate(result['field_suggestions'][:3], 1):
                    print(f"  {j}. {field_suggestion['field_name']} (Score: {field_suggestion['relevance_score']:.3f})")
                    print(f"     Suggested: {field_suggestion['suggested_combination']}")
            
            # Show improved expression
            improved = self.analyzer.generate_improved_expression(expression, improvement_request)
            if improved != expression:
                print(f"Improved: {improved}")
        
        return results


def main():
    """Main function for CLI tool."""
    parser = argparse.ArgumentParser(description="Alpha Expression CLI Tool")
    parser.add_argument("--expression", "-e", help="Alpha expression(s) to analyze (use semicolon to separate multiple)")
    parser.add_argument("--improvement-request", "-i", 
                       help="Custom improvement request (e.g., 'Reduce turnover by adding conditions')")
    parser.add_argument("--operators-file", "-f", default="__operator__.json",
                       help="Path to operators JSON file")
    parser.add_argument("--output-format", "-o", choices=["text", "json"], default="text",
                       help="Output format")
    parser.add_argument("--batch", "-b", nargs="+", help="Analyze multiple expressions")
    parser.add_argument("--interactive", "-t", action="store_true", 
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = AlphaCLI(args.operators_file)
    
    try:
        if args.interactive:
            # Interactive mode
            cli.run_interactive()
        
        elif args.batch:
            # Batch analysis mode
            cli.batch_analyze(args.batch, args.improvement_request)
        
        elif args.expression:
            # Single or multiple expression analysis
            if ";" in args.expression:
                # Multiple expressions
                expressions = [expr.strip() for expr in args.expression.split(";")]
                result = cli.analyze_multiple(expressions, args.improvement_request, args.output_format)
                
                if args.output_format == "json" and result:
                    print(json.dumps(result, indent=2))
            else:
                # Single expression
                result = cli.analyze_single(args.expression, args.improvement_request, args.output_format)
                
                if args.output_format == "json" and result:
                    print(json.dumps(result, indent=2))
        
        else:
            # Default to interactive mode
            cli.run_interactive()
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
