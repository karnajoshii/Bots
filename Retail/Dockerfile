# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for FAISS)
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn python-dotenv

# Copy project files
COPY api.py .
COPY faiss_db ./faiss_db

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Expose port 5000
EXPOSE 5000

# Run Flask app with Gunicorn
CMD ["gunicorn", "api:app", "--bind", "0.0.0.0:5000", "--timeout", "120"]
