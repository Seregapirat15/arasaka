#!/usr/bin/env python3
import sys
import os
import grpc

sys.path.insert(0, 'src')

from infrastructure.grpc import arasaka_pb2
from infrastructure.grpc import arasaka_pb2_grpc

def test_search():
    """Тестируем поиск по вопросам"""
    
    channel = grpc.insecure_channel('localhost:8001')
    stub = arasaka_pb2_grpc.ArasakaServiceStub(channel)
    
    try:
        print("Testing Russian Search...")
        search_request = arasaka_pb2.SearchRequest(
            query="что меня зачислили на курс на lms.bmstu.ru?",
            limit=10,
            score_threshold=0.0
        )
        search_response = stub.SearchAnswers(
            search_request,
            timeout=60.0
        )
        
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
