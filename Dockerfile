# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY . .

# Expose port
EXPOSE 8000

# Run the application - CHANGED THIS LINE
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]