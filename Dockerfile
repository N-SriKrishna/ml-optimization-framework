FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install -e .

# Run tests to verify installation
RUN PYTHONPATH=. pytest tests/ -v

CMD ["python", "examples/real_world/02_optimize_yolov8.py"]
