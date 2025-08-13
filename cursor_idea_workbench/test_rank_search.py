#!/usr/bin/env python3

from query_vector_database import WorldQuantMinerQuery

# Initialize vector database
db = WorldQuantMinerQuery()

# Search for rank operator
print("Searching for 'rank' operator...")
results = db.search_operators('rank', top_k=10)

print(f"Found {len(results)} operators:")
for i, result in enumerate(results, 1):
    if hasattr(result, 'fields'):
        # Hosted embeddings response format
        metadata = result.fields
        operator_name = metadata.get('name', 'N/A')
        operator_desc = metadata.get('description', 'N/A')
        operator_def = metadata.get('definition', 'N/A')
        score = result._score
    elif hasattr(result, 'metadata'):
        # Traditional search response format
        metadata = result.metadata
        operator_name = metadata.get('name', 'N/A')
        operator_desc = metadata.get('description', 'N/A')
        operator_def = metadata.get('definition', 'N/A')
        score = result.score
    else:
        continue
    
    print(f"{i}. {operator_name} (Score: {score:.4f})")
    print(f"   Description: {operator_desc}")
    print(f"   Definition: {operator_def}")
    print()

# Also search for "Use Rank function" to see what comes up
print("\nSearching for 'Use Rank function'...")
results2 = db.search_operators('Use Rank function', top_k=5)

print(f"Found {len(results2)} operators:")
for i, result in enumerate(results2, 1):
    if hasattr(result, 'fields'):
        metadata = result.fields
        operator_name = metadata.get('name', 'N/A')
        operator_desc = metadata.get('description', 'N/A')
        score = result._score
    elif hasattr(result, 'metadata'):
        metadata = result.metadata
        operator_name = metadata.get('name', 'N/A')
        operator_desc = metadata.get('description', 'N/A')
        score = result.score
    else:
        continue
    
    print(f"{i}. {operator_name} (Score: {score:.4f})")
    print(f"   Description: {operator_desc}")
    print()
