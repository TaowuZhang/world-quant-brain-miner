#!/usr/bin/env python3
"""
Example queries for the WorldQuant Miner Pinecone vector database.
This script demonstrates various ways to search and analyze financial metrics.
"""

from query_vector_database import WorldQuantMinerQuery
import json

def example_basic_search():
    """Example of basic text-based search."""
    print("=== Basic Text Search Examples ===")
    
    # Initialize the client
    client = WorldQuantMinerQuery()
    
    # Search queries
    search_terms = [
        "revenue",
        "profit margin", 
        "cash flow",
        "debt",
        "earnings per share"
    ]
    
    for term in search_terms:
        print(f"\nSearching for: '{term}'")
        results = client.search_by_text(term, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.metadata.get('name', 'N/A')} (Score: {result.score:.4f})")
            print(f"     Category: {result.metadata.get('category', 'N/A')}")

def example_category_filtering():
    """Example of filtering by category."""
    print("\n=== Category Filtering Examples ===")
    
    client = WorldQuantMinerQuery()
    
    categories = ["Fundamental", "Analyst"]
    
    for category in categories:
        print(f"\nMetrics in category: '{category}'")
        results = client.search_by_category(category, top_k=5)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.metadata.get('name', 'N/A')}")
            print(f"     Description: {result.metadata.get('description', 'N/A')}")

def example_specific_metric_lookup():
    """Example of looking up specific metrics by ID."""
    print("\n=== Specific Metric Lookup ===")
    
    client = WorldQuantMinerQuery()
    
    # Example metric IDs from your data
    metric_ids = [
        "fnd6_newa1v1300_ivncf",  # Investing Activities - Net Cash Flow
        "fnd6_aqi",               # Acquisitions - Income Contribution
        "est_ffo"                 # Funds From Operation - Summary on Estimations, Mean
    ]
    
    for metric_id in metric_ids:
        print(f"\nLooking up metric ID: {metric_id}")
        metric = client.get_metric_by_id(metric_id)
        
        if metric:
            print(f"  Name: {metric.metadata.get('name', 'N/A')}")
            print(f"  Category: {metric.metadata.get('category', 'N/A')}")
            print(f"  Description: {metric.metadata.get('description', 'N/A')}")
            print(f"  Timestamp: {metric.metadata.get('timestamp', 'N/A')}")
        else:
            print(f"  Metric not found")

def example_database_analysis():
    """Example of analyzing the database structure."""
    print("\n=== Database Analysis ===")
    
    client = WorldQuantMinerQuery()
    
    # Get database summary
    summary = client.get_metrics_summary()
    print("Database Summary:")
    print(json.dumps(summary, indent=2))
    
    # Get all categories
    categories = client.get_all_categories()
    print(f"\nAvailable categories: {categories}")

def example_advanced_search():
    """Example of advanced search with filters."""
    print("\n=== Advanced Search Examples ===")
    
    client = WorldQuantMinerQuery()
    
    # Search for capital-related metrics in Fundamental category
    print("Searching for 'capital' in Fundamental category:")
    filter_dict = {"category": {"$eq": "Fundamental"}}
    results = client.search_by_text("capital", top_k=5, filter_dict=filter_dict)
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.metadata.get('name', 'N/A')} (Score: {result.score:.4f})")
        print(f"     Description: {result.metadata.get('description', 'N/A')}")

def example_export_data():
    """Example of exporting data to CSV."""
    print("\n=== Data Export Example ===")
    
    client = WorldQuantMinerQuery()
    
    # Export a sample of metrics
    print("Exporting metrics to CSV...")
    client.export_metrics_to_csv("worldquant_metrics_export.csv")
    print("Export completed!")

def main():
    """Run all example functions."""
    try:
        example_basic_search()
        example_category_filtering()
        example_specific_metric_lookup()
        example_database_analysis()
        example_advanced_search()
        example_export_data()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
