import asyncio
import sys
import os
import csv
import argparse
import traceback
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print("Importing modules...")
try:
    from infrastructure.db.qdrant import QdrantRepository
    print("✓ QdrantRepository imported")
except Exception as e:
    print(f"✗ Failed to import QdrantRepository: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from infrastructure.ml.embedding_service_impl import EmbeddingServiceImpl
    print("✓ EmbeddingServiceImpl imported")
except Exception as e:
    print(f"✗ Failed to import EmbeddingServiceImpl: {e}")
    traceback.print_exc()
    sys.exit(1)


async def fill_qdrant_from_csv(csv_file: str, text_col: str = "Text_Cleaned", id_col: str = "Id"):
    print(f"\n{'='*60}")
    print(f"Starting data load from {csv_file}")
    print(f"{'='*60}\n")
    
    print("Initializing Qdrant repository...")
    try:
        qdrant_repo = QdrantRepository()
        print(f"✓ Qdrant client initialized: {qdrant_repo.host}:{qdrant_repo.port}")
    except Exception as e:
        print(f"✗ Failed to initialize QdrantRepository: {e}")
        traceback.print_exc()
        return
    
    print("Initializing embedding service...")
    try:
        embedding_service = EmbeddingServiceImpl()
        print("✓ Embedding service initialized")
    except Exception as e:
        print(f"✗ Failed to initialize EmbeddingServiceImpl: {e}")
        traceback.print_exc()
        return
    
    print(f"\nModel name: {embedding_service.model_name}")
    print("Getting embedding dimension...")
    try:
        vector_size = embedding_service.get_embedding_dimension()
        print(f"✓ Vector size: {vector_size}")
    except Exception as e:
        print(f"✗ Failed to get embedding dimension: {e}")
        traceback.print_exc()
        return
    
    print("Creating/checking Qdrant collection...")
    try:
        success = qdrant_repo.create_collection(vector_size=vector_size)
        print(f"✓ Collection created/exists: {success}")
    except Exception as e:
        print(f"✗ Failed to create collection: {e}")
        traceback.print_exc()
        return
    
    data = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=';')
            for row in reader:
                if text_col in row and row[text_col].strip():
                    data.append({
                        'id': row.get(id_col, len(data)),
                        'text': row[text_col].strip(),
                        'metadata': {k: v for k, v in row.items() if k not in [text_col, id_col]}
                    })
    except FileNotFoundError:
        print(f"CSV file not found: {csv_file}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    print(f"Loaded {len(data)} records from CSV")

    points = []
    for i, item in enumerate(data):
        text = item['text']
        print(f"Processing {i + 1}/{len(data)}: {text[:50]}...")
        embedding = embedding_service.encode_text(text)
        
        point = {
            "id": int(item['id']),
            "vector": embedding,
            "payload": {
                "answer": text,
                "answer_id": item['id'],
                "is_visible": True,
                **item['metadata']
            }
        }
        points.append(point)
    
    if points:
        print(f"Uploading {len(points)} points to Qdrant...")
        qdrant_repo.client.upsert(
            collection_name=qdrant_repo.collection_name,
            points=points
        )
        print(f"Uploaded {len(points)} points to Qdrant")
    else:
        print("No points to upload")
    
    print("CSV import completed!")


if __name__ == "__main__":
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        csv_path = os.path.join(project_root, 'ml-service', 'data', 'Answers__202507071202.csv')
        print(f"Project root: {project_root}")
        print(f"CSV path: {csv_path}")
        print(f"CSV file exists: {os.path.exists(csv_path)}")
        
        if not os.path.exists(csv_path):
            print(f"✗ CSV file not found at: {csv_path}")
            print("Available files in ml-service/data:")
            data_dir = os.path.join(project_root, 'ml-service', 'data')
            if os.path.exists(data_dir):
                for f in os.listdir(data_dir):
                    print(f"  - {f}")
            sys.exit(1)
        
        asyncio.run(fill_qdrant_from_csv(csv_path, 'Text_Cleaned', 'Id'))
        print("\n✓ Script completed successfully!")
    except Exception as e:
        print(f"\n✗ Script failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
