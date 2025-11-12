#!/usr/bin/env python3
import sys
import os
import grpc

# Добавляем src в путь
sys.path.insert(0, 'src')

# Импортируем gRPC модули
from infrastructure.grpc import arasaka_pb2
from infrastructure.grpc import arasaka_pb2_grpc

def test_search():
    """Тестируем поиск по вопросам"""
    
    # Подключаемся к сервису
    channel = grpc.insecure_channel('localhost:8001')
    stub = arasaka_pb2_grpc.ArasakaServiceStub(channel)
    
    try:
        # Тест поиска на русском
        print("Testing Russian Search...")
        search_request = arasaka_pb2.SearchRequest(
            query="почему я не могу зарегистрироваться на курс?"
        )
        search_response = stub.SearchAnswers(search_request)
        
        print(f"Found {search_response.total_found} results")
        print(f"Query: {search_response.query}")
        
        for i, result in enumerate(search_response.results, 1):
            print(f"\nResult {i}:")
            print(f"   ID: {result.id}")
            if len(result.answer) > 500:
                print(f"   Answer: {result.answer[:500]}...")
            else:
                print(f"   Answer: {result.answer}")
            
    except grpc.RpcError as e:
        print(f"gRPC Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        channel.close()

if __name__ == "__main__":
    print("Russian Search Test")
    print("=" * 50)
    test_search()
