#!/usr/bin/env python3
"""
Example usage of the Alpha Expression Analyzer
Demonstrates how to analyze and improve alpha expressions with vector database integration.
"""

from alpha_analyzer import AlphaAnalyzer
from alpha_cli import AlphaCLI

def example_analysis():
    """Example analysis of the provided alpha expression."""
    
    # Initialize analyzer
    analyzer = AlphaAnalyzer("__operator__.json")
    
    # The example expression from the user
    original_expression = "-abs(subtract(news_max_up_ret, news_max_dn_ret))"
    improvement_request = "Reduce turnover by adding conditions over some relevant fields"
    
    print("=" * 70)
    print("ALPHA EXPRESSION ANALYSIS EXAMPLE")
    print("=" * 70)
    print(f"Original Expression: {original_expression}")
    print(f"Improvement Request: {improvement_request}")
    print()
    
    # Analyze the expression
    print("ANALYSIS:")
    suggestions = analyzer.suggest_improvements(original_expression, improvement_request)
    
    print(f"  Operators Used: {', '.join(suggestions['analysis']['operators_used'])}")
    print(f"  Fields Used: {', '.join(suggestions['analysis']['fields_used'])}")
    print(f"  Complexity Score: {suggestions['analysis']['complexity_score']}")
    print(f"  Categories: {', '.join(suggestions['analysis']['categories'])}")
    print()
    
    # Show general suggestions
    if suggestions['general_suggestions']:
        print("GENERAL SUGGESTIONS:")
        for i, suggestion in enumerate(suggestions['general_suggestions'], 1):
            print(f"  {i}. {suggestion}")
        print()
    
    # Show specific improvements for the custom request
    if suggestions['specific_improvements']:
        print("SPECIFIC IMPROVEMENTS:")
        for i, improvement in enumerate(suggestions['specific_improvements'], 1):
            print(f"  {i}. {improvement}")
        print()
    
    # Show field suggestions from vector database
    if suggestions['field_suggestions']:
        print("RELATED FIELD SUGGESTIONS:")
        for i, field_suggestion in enumerate(suggestions['field_suggestions'][:3], 1):
            print(f"  {i}. {field_suggestion['field_name']} (Score: {field_suggestion['relevance_score']:.3f})")
            print(f"     Category: {field_suggestion['category']}")
            print(f"     Description: {field_suggestion['description']}")
            print(f"     Suggested Combination: {field_suggestion['suggested_combination']}")
            print()
    
    # Generate improved expression
    improved_expression = analyzer.generate_improved_expression(original_expression, improvement_request)
    
    print("IMPROVED EXPRESSION:")
    print(f"  Original: {original_expression}")
    print(f"  Improved: {improved_expression}")
    print()
    
    # Explain the improvements
    print("EXPLANATION OF IMPROVEMENTS:")
    print("  1. Added 'hump' operator to limit changes and reduce turnover")
    print("  2. Added 'ts_decay_linear' for time-weighted calculations")
    print("  3. The improved expression will be more stable and have lower transaction costs")
    print("  4. Vector database integration provides related field suggestions for combinations")
    print()
    
    return improved_expression

def example_multiple_expressions():
    """Example showing multiple expression analysis and combinations."""
    
    analyzer = AlphaAnalyzer("__operator__.json")
    expressions = [
        "ts_mean(close, 20)",
        "rank(volume)",
        "subtract(high, low)"
    ]
    improvement_request = "Add filtering conditions"
    
    print("=" * 70)
    print("MULTIPLE EXPRESSION ANALYSIS EXAMPLE")
    print("=" * 70)
    print(f"Expressions: {expressions}")
    print(f"Improvement Request: {improvement_request}")
    print()
    
    # Analyze multiple expressions
    suggestions = analyzer.suggest_multiple_expressions(expressions, improvement_request)
    
    # Show individual expression analysis
    for i, expr_result in enumerate(suggestions['expressions'], 1):
        print(f"Expression {i}: {expressions[i-1]}")
        print(f"  Complexity: {expr_result['analysis']['complexity_score']}")
        print(f"  Operators: {', '.join(expr_result['analysis']['operators_used'])}")
        print(f"  Fields: {', '.join(expr_result['analysis']['fields_used'])}")
        print()
    
    # Show combinations
    if suggestions['combinations']:
        print("EXPRESSION COMBINATIONS:")
        for i, combination in enumerate(suggestions['combinations'], 1):
            print(f"  {i}. {combination}")
        print()

def example_field_suggestions():
    """Example showing field suggestions from vector database."""
    
    analyzer = AlphaAnalyzer("__operator__.json")
    expression = "ts_mean(close, 20)"
    improvement_request = "Find related fields for combination"
    
    print("=" * 70)
    print("FIELD SUGGESTIONS EXAMPLE")
    print("=" * 70)
    print(f"Expression: {expression}")
    print(f"Improvement Request: {improvement_request}")
    print()
    
    # Get suggestions
    suggestions = analyzer.suggest_improvements(expression, improvement_request)
    
    if suggestions['field_suggestions']:
        print("RELATED FIELD SUGGESTIONS FROM VECTOR DATABASE:")
        for i, field_suggestion in enumerate(suggestions['field_suggestions'][:5], 1):
            print(f"  {i}. {field_suggestion['field_name']} (Score: {field_suggestion['relevance_score']:.3f})")
            print(f"     Category: {field_suggestion['category']}")
            print(f"     Description: {field_suggestion['description']}")
            print(f"     Suggested Combination: {field_suggestion['suggested_combination']}")
            print()
    else:
        print("No field suggestions available (vector database not connected)")
        print()

def example_conditional_expressions():
    """Example showing conditional expressions with if_else operators."""
    
    analyzer = AlphaAnalyzer("__operator__.json")
    expressions = [
        "if_else(greater(close, ts_mean(close, 20)), 1, -1)",
        "ts_std_dev(volume, 10)"
    ]
    improvement_request = "Add risk management controls"
    
    print("=" * 70)
    print("CONDITIONAL EXPRESSIONS EXAMPLE")
    print("=" * 70)
    print(f"Expressions: {expressions}")
    print(f"Improvement Request: {improvement_request}")
    print()
    
    # Analyze multiple expressions
    suggestions = analyzer.suggest_multiple_expressions(expressions, improvement_request)
    
    # Show individual expression analysis
    for i, expr_result in enumerate(suggestions['expressions'], 1):
        print(f"Expression {i}: {expressions[i-1]}")
        print(f"  Complexity: {expr_result['analysis']['complexity_score']}")
        print(f"  Operators: {', '.join(expr_result['analysis']['operators_used'])}")
        print(f"  Fields: {', '.join(expr_result['analysis']['fields_used'])}")
        print()
    
    # Show combinations
    if suggestions['combinations']:
        print("CONDITIONAL COMBINATIONS:")
        for i, combination in enumerate(suggestions['combinations'], 1):
            print(f"  {i}. {combination}")
        print()

def example_interactive_usage():
    """Example of how to use the interactive CLI."""
    
    print("=" * 70)
    print("INTERACTIVE CLI USAGE")
    print("=" * 70)
    print("To use the interactive CLI, run:")
    print("  python alpha_cli.py --interactive")
    print()
    print("Or for a single expression with custom improvement request:")
    print("  python alpha_cli.py --expression '-abs(subtract(news_max_up_ret, news_max_dn_ret))' --improvement-request 'Reduce turnover by adding conditions'")
    print()
    print("Or for multiple expressions:")
    print("  python alpha_cli.py --expression 'ts_mean(close, 20); rank(volume); subtract(high, low)' --improvement-request 'Make more robust'")
    print()
    print("Or for conditional expressions:")
    print("  python alpha_cli.py --expression 'if_else(greater(close, ts_mean(close, 20)), 1, -1); ts_std_dev(volume, 10)' --improvement-request 'Add risk controls'")
    print()
    print("Or for batch analysis:")
    print("  python alpha_cli.py --batch 'ts_mean(close, 20)' 'rank(volume)' 'subtract(high, low)' --improvement-request 'Find related fields'")
    print()

def example_advanced_analysis():
    """Example of more complex alpha expression analysis with vector database integration."""
    
    analyzer = AlphaAnalyzer("__operator__.json")
    
    complex_expressions = [
        "ts_mean(subtract(high, low), 20)",
        "rank(ts_std_dev(close, 10))",
        "multiply(ts_mean(volume, 5), ts_corr(close, volume, 10))",
        "if_else(greater(close, ts_mean(close, 20)), 1, -1)"
    ]
    
    improvement_requests = [
        "Reduce turnover by adding conditions",
        "Make it more robust against outliers",
        "Add time decay for smoother signals",
        "Improve performance and efficiency",
        "Find related fields for combination"
    ]
    
    print("=" * 70)
    print("COMPLEX EXPRESSIONS ANALYSIS WITH VECTOR DB")
    print("=" * 70)
    
    for i, expression in enumerate(complex_expressions, 1):
        print(f"\nExpression {i}: {expression}")
        
        # Analyze with different improvement requests
        for improvement_request in improvement_requests:
            print(f"\n  {improvement_request}:")
            suggestions = analyzer.suggest_improvements(expression, improvement_request)
            
            # Show improved expression
            improved = analyzer.generate_improved_expression(expression, improvement_request)
            if improved != expression:
                print(f"    Improved: {improved}")
            
            # Show field suggestions
            if suggestions['field_suggestions']:
                print(f"    Related Fields: {len(suggestions['field_suggestions'])} found")
                for field_suggestion in suggestions['field_suggestions'][:2]:
                    print(f"      - {field_suggestion['field_name']} (Score: {field_suggestion['relevance_score']:.3f})")
        
        print("-" * 50)

def example_custom_requests():
    """Example of various custom improvement requests."""
    
    analyzer = AlphaAnalyzer("__operator__.json")
    expression = "rank(volume)"
    
    print("=" * 70)
    print("CUSTOM IMPROVEMENT REQUESTS")
    print("=" * 70)
    print(f"Base Expression: {expression}")
    print()
    
    custom_requests = [
        "Add filtering conditions",
        "Make it more stable over time",
        "Add risk controls",
        "Improve signal quality",
        "Add position sizing",
        "Make it more responsive to market changes",
        "Add sector neutralization",
        "Reduce noise in the signal",
        "Find related fields for combination"
    ]
    
    for request in custom_requests:
        print(f"Request: '{request}'")
        suggestions = analyzer.suggest_improvements(expression, request)
        
        # Show improved expression
        improved = analyzer.generate_improved_expression(expression, request)
        if improved != expression:
            print(f"  Improved: {improved}")
        else:
            print(f"  No specific improvements applied")
        
        # Show field suggestions
        if suggestions['field_suggestions']:
            print(f"  Related Fields: {len(suggestions['field_suggestions'])} found")
            for field_suggestion in suggestions['field_suggestions'][:2]:
                print(f"    - {field_suggestion['field_name']} (Score: {field_suggestion['relevance_score']:.3f})")
        
        print()

def main():
    """Run all examples."""
    
    # Example 1: Basic analysis with vector database integration
    example_analysis()
    
    # Example 2: Multiple expressions and combinations
    example_multiple_expressions()
    
    # Example 3: Field suggestions from vector database
    example_field_suggestions()
    
    # Example 4: Conditional expressions
    example_conditional_expressions()
    
    # Example 5: Interactive usage
    example_interactive_usage()
    
    # Example 6: Complex expressions with vector DB
    example_advanced_analysis()
    
    # Example 7: Custom requests
    example_custom_requests()
    
    print("=" * 70)
    print("EXAMPLES COMPLETED")
    print("=" * 70)
    print("You can now use the alpha analyzer with vector database integration!")
    print("Features:")
    print("- Multiple expressions separated by semicolons")
    print("- Vector database field suggestions")
    print("- Conditional expressions with if_else operators")
    print("- Custom improvement requests")
    print("- Expression combinations")
    print()
    print("Run 'python alpha_cli.py --help' for more options.")

if __name__ == "__main__":
    main()
