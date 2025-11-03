# Use a lightweight Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install system dependencies
# We need to install system dependencies for building some packages (like dlib, though we skipped it)
# and for running the application (like libgl1 for opencv-python)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libopenblas-dev \
    liblapack-dev \
    cmake \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . $APP_HOME

# Expose the port the FastAPI application will run on
EXPOSE 8000

# Command to run the application
# We use uvicorn to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
