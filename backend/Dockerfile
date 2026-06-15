# Start from official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for psycopg2 and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to start the backend
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]