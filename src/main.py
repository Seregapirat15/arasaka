import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

try:
    from infrastructure.grpc.server import serve
    if __name__ == "__main__":
        serve()
except Exception as e:
    print(f"Error: {e}")
    raise