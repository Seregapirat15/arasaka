# Shared Proto Definitions

This directory contains shared gRPC protocol buffer definitions used by multiple services.

## Structure

- `arasaka.proto` - Main service definitions for ML service and MAX bot communication

## Generating Python Files

To generate Python files from proto definitions:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. arasaka.proto
```

This will generate:
- `arasaka_pb2.py` - Message definitions
- `arasaka_pb2_grpc.py` - Service definitions

## Usage

Services import generated files from this directory to ensure consistency across the system.

