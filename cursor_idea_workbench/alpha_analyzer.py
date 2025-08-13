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
        """Generate custom improvements based on the improvement request using vector database inference."""
        improvements = []
        
        # Use vector database to find relevant operators based on the improvement request
        if self.vector_db:
            try:
                # Search for operators that match the improvement request
                operator_results = self.vector_db.search_operators(improvement_request, top_k=10)
                
                for result in operator_results:
                    # Handle different result structures from hosted embeddings vs traditional search
                    if hasattr(result, 'fields'):
                        # Hosted embeddings response format
                        metadata = result.fields
                        operator_name = metadata.get('name', '')
                        operator_desc = metadata.get('description', '')
                        operator_def = metadata.get('definition', '')
                    elif hasattr(result, 'metadata'):
                        # Traditional search response format
                        metadata = result.metadata
                        operator_name = metadata.get('name', '')
                        operator_desc = metadata.get('description', '')
                        operator_def = metadata.get('definition', '')
                    else:
                        continue
                    
                    # Skip if operator is already used in the expression
                    if operator_name in parsed.operators:
                        continue
                    
                    # Generate improvement suggestion based on operator definition from vector database
                    if operator_name and operator_def:
                        # Use the operator definition from the vector database to generate the suggestion
                        # Replace placeholders in the definition with the actual expression
                        suggestion = self._generate_operator_suggestion(operator_name, operator_def, parsed.original)
                        if suggestion:
                            improvements.append(suggestion)
                
            except Exception as e:
                print(f"Error searching for operators: {e}")
        
        return improvements[:5]  # Limit to top 5 suggestions
    
    def _generate_operator_suggestion(self, operator_name: str, operator_def: str, expression: str) -> str:
        """Generate operator suggestion based on operator definition from vector database."""
        try:
            # Parse the operator definition to understand its structure
            # Common patterns: operator_name(x), operator_name(x, y), etc.
            if '(' in operator_def and ')' in operator_def:
                # Extract parameters from definition
                params_start = operator_def.find('(')
                params_end = operator_def.rfind(')')
                params_str = operator_def[params_start + 1:params_end]
                
                # Split parameters and clean them
                params = [p.strip() for p in params_str.split(',') if p.strip()]
                
                if len(params) == 1:
                    # Single parameter operator like rank(x)
                    return f"Add {operator_name} operator: {operator_name}({expression})"
                elif len(params) == 2:
                    # Two parameter operator like hump(x, y)
                    # Use default values based on operator type
                    if operator_name == 'hump':
                        return f"Add {operator_name} operator: {operator_name}({expression}, 0.01)"
                    elif operator_name == 'ts_decay_linear':
                        return f"Add {operator_name} operator: {operator_name}({expression}, 20)"
                    elif operator_name == 'winsorize':
                        return f"Add {operator_name} operator: {operator_name}({expression}, 4)"
                    elif operator_name == 'ts_zscore':
                        return f"Add {operator_name} operator: {operator_name}({expression}, 20)"
                    else:
                        return f"Add {operator_name} operator: {operator_name}({expression}, 1)"
                elif len(params) == 3:
                    # Three parameter operator like if_else(condition, true, false)
                    if operator_name == 'if_else':
                        return f"Add conditional logic: {operator_name}(greater({expression}, 0), {expression}, 0)"
                    elif operator_name == 'trade_when':
                        return f"Add threshold filtering: {operator_name}({expression}, greater({expression}, 0), 0)"
                    else:
                        return f"Add {operator_name} operator: {operator_name}({expression}, 0, 1)"
                else:
                    # Generic case
                    return f"Add {operator_name} operator: {operator_name}({expression})"
            else:
                # Simple operator without clear parameter structure
                return f"Add {operator_name} operator: {operator_name}({expression})"
                
        except Exception as e:
            print(f"Error generating operator suggestion for {operator_name}: {e}")
            return f"Add {operator_name} operator: {operator_name}({expression})"
    
    def generate_improved_expression(self, expression: str, improvement_request: str = None) -> str:
        """Generate an improved version of the alpha expression using vector database inference."""
        parsed = self.parse_expression(expression)
        
        # Start with the original expression
        improved = parsed.original
        
        if improvement_request and self.vector_db:
            try:
                # Step 1: Search for related operators based on improvement request
                operator_results = self.vector_db.search_operators(improvement_request, top_k=5)
                relevant_operators = []
                
                for result in operator_results:
                    if hasattr(result, 'fields'):
                        metadata = result.fields
                        operator_name = metadata.get('name', '')
                        operator_desc = metadata.get('description', '')
                        operator_def = metadata.get('definition', '')
                    elif hasattr(result, 'metadata'):
                        metadata = result.metadata
                        operator_name = metadata.get('name', '')
                        operator_desc = metadata.get('description', '')
                        operator_def = metadata.get('definition', '')
                    else:
                        continue
                    
                    if operator_name and operator_name not in parsed.operators:
                        relevant_operators.append({
                            'name': operator_name,
                            'description': operator_desc,
                            'definition': operator_def
                        })
                
                # Step 2: Search for related fields from vector database
                field_suggestions = self._find_related_fields(parsed.fields, improvement_request)
                related_fields = []
                
                for field_suggestion in field_suggestions[:3]:  # Top 3 related fields
                    related_fields.append({
                        'name': field_suggestion.field_name,
                        'description': field_suggestion.description,
                        'category': field_suggestion.category,
                        'suggested_combination': field_suggestion.suggested_combination
                    })
                
                # Step 3: Generate improved expression using vector database inference
                improved = self._generate_expression_with_inference(
                    parsed, improvement_request, relevant_operators, related_fields
                )
                
            except Exception as e:
                print(f"Error in vector database inference: {e}")
                # Return original expression if vector database inference fails
                return parsed.original
        
        return improved
    
    def _generate_expression_with_inference(self, parsed: AlphaExpression, improvement_request: str, 
                                          relevant_operators: List[Dict], related_fields: List[Dict]) -> str:
        """Generate improved expression using vector database inference."""
        
        # Create a comprehensive prompt for the vector database
        prompt = self._create_improvement_prompt(parsed, improvement_request, relevant_operators, related_fields)
        
        try:
            # Search for similar expressions or patterns in the vector database
            # This could be in a "expressions" or "patterns" namespace
            expression_results = self.vector_db.search_by_text(prompt, top_k=3)
            
            if expression_results:
                # Use the most relevant result as inspiration
                best_result = expression_results[0]
                if hasattr(best_result, 'fields'):
                    metadata = best_result.fields
                elif hasattr(best_result, 'metadata'):
                    metadata = best_result.metadata
                else:
                    metadata = {}
                
                # Extract pattern or template from the result
                pattern = metadata.get('pattern', '') or metadata.get('expression', '')
                if pattern:
                    return self._apply_pattern_to_expression(parsed.original, pattern, relevant_operators)
            
        except Exception as e:
            print(f"Error searching for expression patterns: {e}")
        
        # If no pattern found, generate based on operators and fields
        return self._generate_expression_from_components(parsed, improvement_request, relevant_operators, related_fields)
    
    def _create_improvement_prompt(self, parsed: AlphaExpression, improvement_request: str,
                                 relevant_operators: List[Dict], related_fields: List[Dict]) -> str:
        """Create a comprehensive prompt for vector database inference."""
        
        prompt_parts = [
            f"Original alpha expression: {parsed.original}",
            f"Improvement request: {improvement_request}",
            f"Current operators: {', '.join(parsed.operators)}",
            f"Current fields: {', '.join(parsed.fields)}",
            f"Complexity score: {parsed.complexity}"
        ]
        
        if relevant_operators:
            prompt_parts.append("Relevant operators to consider:")
            for op in relevant_operators:
                prompt_parts.append(f"- {op['name']}: {op['description']} (Definition: {op['definition']})")
        
        if related_fields:
            prompt_parts.append("Related fields to consider:")
            for field in related_fields:
                prompt_parts.append(f"- {field['name']}: {field['description']} (Category: {field['category']})")
        
        return " | ".join(prompt_parts)
    
    def _apply_operator_from_definition(self, operator_name: str, operator_def: str, expression: str) -> str:
        """Apply operator to expression based on its definition from vector database."""
        try:
            # Parse the operator definition to understand its structure
            if '(' in operator_def and ')' in operator_def:
                # Extract parameters from definition
                params_start = operator_def.find('(')
                params_end = operator_def.rfind(')')
                params_str = operator_def[params_start + 1:params_end]
                
                # Split parameters and clean them
                params = [p.strip() for p in params_str.split(',') if p.strip()]
                
                if len(params) == 1:
                    # Single parameter operator like rank(x)
                    return f"{operator_name}({expression})"
                elif len(params) == 2:
                    # Two parameter operator like hump(x, y)
                    # Use default values based on operator type
                    if operator_name == 'hump':
                        return f"{operator_name}({expression}, 0.01)"
                    elif operator_name == 'ts_decay_linear':
                        return f"{operator_name}({expression}, 20)"
                    elif operator_name == 'winsorize':
                        return f"{operator_name}({expression}, 4)"
                    elif operator_name == 'ts_zscore':
                        return f"{operator_name}({expression}, 20)"
                    else:
                        return f"{operator_name}({expression}, 1)"
                elif len(params) == 3:
                    # Three parameter operator like if_else(condition, true, false)
                    if operator_name == 'if_else':
                        return f"{operator_name}(greater({expression}, 0), {expression}, 0)"
                    elif operator_name == 'trade_when':
                        return f"{operator_name}({expression}, greater({expression}, 0), 0)"
                    else:
                        return f"{operator_name}({expression}, 0, 1)"
                else:
                    # Generic case
                    return f"{operator_name}({expression})"
            else:
                # Simple operator without clear parameter structure
                return f"{operator_name}({expression})"
                
        except Exception as e:
            print(f"Error applying operator {operator_name} from definition: {e}")
            return f"{operator_name}({expression})"
    
    def _generate_expression_from_components(self, parsed: AlphaExpression, improvement_request: str,
                                           relevant_operators: List[Dict], related_fields: List[Dict]) -> str:
        """Generate expression from available operators and fields using vector database inference."""
        
        improved = parsed.original
        
        # Apply the most relevant operator from vector search based on its definition
        if relevant_operators:
            best_operator = relevant_operators[0]
            operator_name = best_operator['name']
            operator_def = best_operator.get('definition', '')
            
            # Use the operator definition to determine how to apply it
            improved = self._apply_operator_from_definition(operator_name, operator_def, improved)
        
        # If we have related fields and the request mentions combining, suggest incorporating them
        if related_fields and ("combine" in improvement_request.lower() or "add" in improvement_request.lower()):
            best_field = related_fields[0]
            if best_field['suggested_combination']:
                # Use the suggested combination from field suggestions
                improved = best_field['suggested_combination'].replace(
                    best_field['name'], improved
                )
        
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
