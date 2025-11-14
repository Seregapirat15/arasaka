# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY src/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY docker-compose.yml .

# Model will be downloaded on first run

# Create non-root user and set up cache directories
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /home/app/.cache/huggingface /home/app/.cache/torch && \
    chown -R app:app /app /home/app/.cache
USER app

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('localhost', 8001), timeout=5)"

# Run the application
CMD ["python", "src/main.py"]
