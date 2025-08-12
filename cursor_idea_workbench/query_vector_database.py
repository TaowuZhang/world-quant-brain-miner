import pinecone
import os
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import pandas as pd

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass

class WorldQuantMinerQuery:
    """
    A class to query the WorldQuant Miner Pinecone vector database
    containing financial metrics and fundamental data.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Pinecone client and connect to the index.
        
        Args:
            api_key (str): Pinecone API key. If None, will try to get from environment.
        """
        if api_key is None:
            api_key = os.getenv('PINECONE_API_KEY')
            if api_key is None:
                raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize Pinecone client
        self.pc = pinecone.Pinecone(api_key=api_key)
        
        # Connect to the WorldQuant Miner index
        self.index_name = "worldquant-miner"
        self.index = self.pc.Index(self.index_name)
        
        # Index configuration
        self.dimensions = 1024
        self.metric = "cosine"
        
        print(f"Connected to Pinecone index: {self.index_name}")
        print(f"Index stats: {self.index.describe_index_stats()}")
    
    def search_by_text(self, query_text: str, top_k: int = 10, 
                      filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar financial metrics by text query using hosted embeddings.
        
        Args:
            query_text (str): The text to search for
            top_k (int): Number of results to return
            filter_dict (dict): Optional metadata filters (not supported with hosted embeddings)
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Use Pinecone's hosted embedding model for automatic text-to-vector conversion
            results = self.index.search(
                namespace="data-fields",
                query={
                    "inputs": {"text": query_text},
                    "top_k": top_k
                }
            )
            # Handle the new response format from hosted embeddings
            if hasattr(results, 'result') and hasattr(results.result, 'hits'):
                return results.result.hits
            elif hasattr(results, 'matches'):
                return results.matches
            else:
                return []
        except Exception as e:
            print(f"Error searching with hosted embeddings: {e}")
            # Fallback to traditional vector search
            try:
                dummy_vector = [0.1] * self.dimensions
                results = self.index.query(
                    vector=dummy_vector,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )
                return results.matches
            except Exception as e2:
                print(f"Error in fallback vector search: {e2}")
                return []
            
    
    def search_by_category(self, category: str, top_k: int = 50) -> List[Dict]:
        """
        Search for metrics within a specific category.
        
        Args:
            category (str): Category to filter by (e.g., "Fundamental", "Analyst")
            top_k (int): Number of results to return
            
        Returns:
            List of search results
        """
        filter_dict = {"category": {"$eq": category}}
        
        try:
            # Use hosted embeddings with category search
            results = self.index.search(
                namespace="data-fields",
                query={
                    "inputs": {"text": category},
                    "top_k": top_k
                }
            )
            return results.matches
        except Exception as e:
            print(f"Error searching category with hosted embeddings: {e}")
            # Fallback to dummy vector search
            dummy_vector = [0.0] * self.dimensions
            results = self.index.query(
                vector=dummy_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            return results.matches
    
    def get_metric_by_id(self, metric_id: str) -> Optional[Dict]:
        """
        Retrieve a specific metric by its ID.
        
        Args:
            metric_id (str): The ID of the metric to retrieve
            
        Returns:
            Metric data if found, None otherwise
        """
        try:
            result = self.index.fetch(ids=[metric_id])
            if metric_id in result.vectors:
                return result.vectors[metric_id]
            return None
        except Exception as e:
            print(f"Error fetching metric {metric_id}: {e}")
            return None
    
    def get_all_categories(self) -> List[str]:
        """
        Get all unique categories in the database.
        
        Returns:
            List of unique category names
        """
        stats = self.index.describe_index_stats()
        if 'namespaces' in stats and '' in stats['namespaces']:
            # This is a simplified approach - in practice you might need to scan
            # the entire index to get all categories
            return ["Fundamental", "Analyst", "Technical", "Market"]  # Based on your data
        return []
    
    def search_similar_metrics(self, metric_name: str, top_k: int = 10) -> List[Dict]:
        """
        Find metrics similar to a given metric name.
        
        Args:
            metric_name (str): Name of the metric to find similar ones for
            top_k (int): Number of similar metrics to return
            
        Returns:
            List of similar metrics
        """
        return self.search_by_text(metric_name, top_k=top_k)
    
    def search_operators(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        Search for operators in the operators namespace using hosted embeddings.
        
        Args:
            query_text (str): The text to search for
            top_k (int): Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Use Pinecone's hosted embedding model for automatic text-to-vector conversion
            results = self.index.search(
                namespace="operators",
                query={
                    "inputs": {"text": query_text},
                    "top_k": top_k
                }
            )
            # Handle the new response format from hosted embeddings
            if hasattr(results, 'result') and hasattr(results.result, 'hits'):
                return results.result.hits
            elif hasattr(results, 'matches'):
                return results.matches
            else:
                return []
        except Exception as e:
            print(f"Error searching operators with hosted embeddings: {e}")
            # Fallback to traditional vector search
            try:
                dummy_vector = [0.1] * self.dimensions
                results = self.index.query(
                    vector=dummy_vector,
                    top_k=top_k,
                    include_metadata=True,
                    namespace="operators"
                )
                return results.matches
            except Exception as e2:
                print(f"Error in fallback operator search: {e2}")
                return []
    
    def search_data_fields(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        Search for data fields in the data-fields namespace using hosted embeddings.
        
        Args:
            query_text (str): The text to search for
            top_k (int): Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Use Pinecone's hosted embedding model for automatic text-to-vector conversion
            results = self.index.search(
                namespace="data-fields",
                query={
                    "inputs": {"text": query_text},
                    "top_k": top_k
                }
            )
            # Handle the new response format from hosted embeddings
            if hasattr(results, 'result') and hasattr(results.result, 'hits'):
                return results.result.hits
            elif hasattr(results, 'matches'):
                return results.matches
            else:
                return []
        except Exception as e:
            print(f"Error searching data fields with hosted embeddings: {e}")
            # Fallback to traditional vector search
            try:
                dummy_vector = [0.1] * self.dimensions
                results = self.index.query(
                    vector=dummy_vector,
                    top_k=top_k,
                    include_metadata=True,
                    namespace="data-fields"
                )
                return results.matches
            except Exception as e2:
                print(f"Error in fallback data fields search: {e2}")
                return []
    
    def analyze_metric_trends(self, metric_pattern: str = None) -> pd.DataFrame:
        """
        Analyze trends in metrics based on timestamps.
        
        Args:
            metric_pattern (str): Optional pattern to filter metrics
            
        Returns:
            DataFrame with trend analysis
        """
        # Get all metrics (this might be expensive for large datasets)
        # In practice, you might want to implement pagination
        try:
            results = self.index.search(
                namespace="data-fields",
                query={
                    "inputs": {"text": metric_pattern or "financial metrics"},
                    "top_k": 1000
                }
            )
            results = results.matches
        except Exception as e:
            print(f"Error getting metrics with hosted embeddings: {e}")
            results = self.search_by_text("", top_k=1000)  # Fallback
        
        data = []
        for match in results:
            metadata = match.metadata
            if 'timestamp' in metadata:
                data.append({
                    'id': match.id,
                    'name': metadata.get('name', ''),
                    'category': metadata.get('category', ''),
                    'description': metadata.get('description', ''),
                    'timestamp': metadata.get('timestamp', ''),
                    'score': match.score
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def export_metrics_to_csv(self, filename: str = "worldquant_metrics.csv"):
        """
        Export all metrics to a CSV file.
        
        Args:
            filename (str): Output filename
        """
        df = self.analyze_metric_trends()
        df.to_csv(filename, index=False)
        print(f"Exported {len(df)} metrics to {filename}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the metrics database.
        
        Returns:
            Dictionary with database statistics
        """
        stats = self.index.describe_index_stats()
        
        # Get sample metrics for analysis
        sample_results = self.search_by_text("", top_k=100)
        
        categories = {}
        for match in sample_results:
            category = match.metadata.get('category', 'Unknown')
            categories[category] = categories.get(category, 0) + 1
        
        return {
            'total_records': stats.get('total_vector_count', 0),
            'dimensions': self.dimensions,
            'metric': self.metric,
            'categories': categories,
            'sample_size': len(sample_results)
        }


def main():
    """
    Example usage of the WorldQuantMinerQuery class.
    """
    # Initialize the query client
    # API key will be loaded from environment variable PINECONE_API_KEY
    try:
        query_client = WorldQuantMinerQuery()
        
        # Example 1: Search for cash flow related metrics
        print("\n=== Searching for cash flow metrics ===")
        cash_flow_results = query_client.search_by_text("cash flow", top_k=5)
        for i, result in enumerate(cash_flow_results, 1):
            print(f"{i}. {result.metadata.get('name', 'N/A')} (Score: {result.score:.4f})")
            print(f"   Category: {result.metadata.get('category', 'N/A')}")
            print(f"   Description: {result.metadata.get('description', 'N/A')}")
            print()
        
        # Example 2: Get all Fundamental metrics
        print("\n=== Fundamental metrics ===")
        fundamental_results = query_client.search_by_category("Fundamental", top_k=10)
        for i, result in enumerate(fundamental_results, 1):
            print(f"{i}. {result.metadata.get('name', 'N/A')}")
        
        # Example 3: Get database summary
        print("\n=== Database Summary ===")
        summary = query_client.get_metrics_summary()
        print(json.dumps(summary, indent=2))
        
        # Example 4: Export to CSV
        print("\n=== Exporting metrics to CSV ===")
        query_client.export_metrics_to_csv("worldquant_metrics_sample.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install pinecone-client sentence-transformers pandas")


if __name__ == "__main__":
    main()
