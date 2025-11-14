import asyncio
import sys
import os
import csv
import argparse
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from infrastructure.db.qdrant import QdrantRepository
from infrastructure.ml.embedding_service_impl import EmbeddingServiceImpl


async def fill_qdrant_from_csv(csv_file: str, text_col: str = "Text_Cleaned", id_col: str = "Id"):
    print(f"Starting data load from {csv_file}")
    qdrant_repo = QdrantRepository()
    embedding_service = EmbeddingServiceImpl()
    
    print(f"Model name: {embedding_service.model_name}")
    print(f"Vector size: {embedding_service.get_embedding_dimension()}")
    vector_size = embedding_service.get_embedding_dimension()
    success = qdrant_repo.create_collection(vector_size=vector_size)
    print(f"Collection created: {success}")
    
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
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    csv_path = os.path.join(project_root, 'src', 'data', 'Answers__202507071202.csv')
    asyncio.run(fill_qdrant_from_csv(csv_path, 'Text_Cleaned', 'Id'))
