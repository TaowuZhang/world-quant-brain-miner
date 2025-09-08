#!/usr/bin/env python3
"""
Example script showing how to use the generated templates
"""

import json
import random
from typing import List, Dict

def load_templates(filename: str = 'generatedTemplate.json') -> Dict:
    """Load generated templates from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Template file {filename} not found!")
        print("Run the template generator first: python run_generator.py")
        return {}
    except Exception as e:
        print(f"❌ Error loading templates: {e}")
        return {}

def get_random_template(templates: Dict, region: str = None) -> Dict:
    """Get a random template from the loaded templates"""
    if not templates or 'templates' not in templates:
        return {}
    
    if region and region in templates['templates']:
        region_templates = templates['templates'][region]
    else:
        # Get templates from all regions
        all_templates = []
        for region_templates in templates['templates'].values():
            all_templates.extend(region_templates)
        region_templates = all_templates
    
    if not region_templates:
        return {}
    
    return random.choice(region_templates)

def analyze_template(template: Dict) -> None:
    """Analyze and display template information"""
    if not template:
        print("No template to analyze")
        return
    
    print(f"📊 Template Analysis:")
    print(f"   Region: {template.get('region', 'Unknown')}")
    print(f"   Expression: {template.get('template', 'N/A')}")
    print(f"   Operators used: {', '.join(template.get('operators_used', []))}")
    print(f"   Fields used: {', '.join(template.get('fields_used', []))}")
    print()

def get_templates_by_region(templates: Dict) -> Dict[str, List[Dict]]:
    """Get templates organized by region"""
    if not templates or 'templates' not in templates:
        return {}
    
    return templates['templates']

def get_templates_by_operator(templates: Dict, operator_name: str) -> List[Dict]:
    """Get all templates that use a specific operator"""
    if not templates or 'templates' not in templates:
        return []
    
    matching_templates = []
    for region_templates in templates['templates'].values():
        for template in region_templates:
            if operator_name in template.get('operators_used', []):
                matching_templates.append(template)
    
    return matching_templates

def get_templates_by_field(templates: Dict, field_name: str) -> List[Dict]:
    """Get all templates that use a specific field"""
    if not templates or 'templates' not in templates:
        return []
    
    matching_templates = []
    for region_templates in templates['templates'].values():
        for template in region_templates:
            if field_name in template.get('fields_used', []):
                matching_templates.append(template)
    
    return matching_templates

def main():
    print("🔍 Template Usage Examples\n")
    
    # Load templates
    templates = load_templates()
    if not templates:
        return
    
    # Display metadata
    metadata = templates.get('metadata', {})
    print(f"📈 Template Statistics:")
    print(f"   Generated at: {metadata.get('generated_at', 'Unknown')}")
    print(f"   Total operators: {metadata.get('total_operators', 'Unknown')}")
    print(f"   Regions: {', '.join(metadata.get('regions', []))}")
    
    total_templates = sum(len(region_templates) for region_templates in templates['templates'].values())
    print(f"   Total templates: {total_templates}")
    print()
    
    # Example 1: Get random template
    print("🎲 Example 1: Random Template")
    random_template = get_random_template(templates)
    analyze_template(random_template)
    
    # Example 2: Get template from specific region
    print("🌍 Example 2: Template from USA Region")
    usa_template = get_random_template(templates, 'USA')
    analyze_template(usa_template)
    
    # Example 3: Templates by region
    print("🗺️  Example 3: Templates by Region")
    templates_by_region = get_templates_by_region(templates)
    for region, region_templates in templates_by_region.items():
        print(f"   {region}: {len(region_templates)} templates")
    print()
    
    # Example 4: Templates using specific operator
    print("⚙️  Example 4: Templates using 'ts_rank' operator")
    ts_rank_templates = get_templates_by_operator(templates, 'ts_rank')
    print(f"   Found {len(ts_rank_templates)} templates using ts_rank")
    if ts_rank_templates:
        analyze_template(ts_rank_templates[0])
    
    # Example 5: Templates using specific field
    print("📊 Example 5: Templates using 'close' field")
    close_templates = get_templates_by_field(templates, 'close')
    print(f"   Found {len(close_templates)} templates using close field")
    if close_templates:
        analyze_template(close_templates[0])
    
    # Example 6: Show all unique operators used
    print("🔧 Example 6: All Unique Operators Used")
    all_operators = set()
    for region_templates in templates['templates'].values():
        for template in region_templates:
            all_operators.update(template.get('operators_used', []))
    
    print(f"   Found {len(all_operators)} unique operators:")
    for op in sorted(all_operators):
        print(f"     - {op}")
    print()
    
    # Example 7: Show all unique fields used
    print("📈 Example 7: All Unique Fields Used")
    all_fields = set()
    for region_templates in templates['templates'].values():
        for template in region_templates:
            all_fields.update(template.get('fields_used', []))
    
    print(f"   Found {len(all_fields)} unique fields:")
    for field in sorted(all_fields):
        print(f"     - {field}")
    print()
    
    print("✅ Examples completed! Use these functions in your own code.")

if __name__ == '__main__':
    main()
