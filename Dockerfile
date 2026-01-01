# Dockerfile for Enterprise RAG System
FROM python:3.11-slim

WORKDIR /app

# System dependencies for PostgreSQL and other libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python package manager (pipx for better tooling)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy Python dependencies and install
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Application code
COPY src/ src/
COPY scripts/ scripts/

# Create data directory
RUN mkdir -p /app/data

# Non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Health check - endpoint will be available after startup
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
