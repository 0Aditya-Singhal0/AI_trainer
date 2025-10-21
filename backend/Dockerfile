FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pyproject.toml and uv.lock files
COPY pyproject.toml uv.lock ./

# Install uv and sync dependencies
RUN pip install --no-cache-dir uv && \
    uv sync

# Copy backend source code (since we're building from the backend context)
COPY . .

EXPOSE 8000

# Run the FastAPI application
CMD ["uv", "run", "backend/main.py"]