# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install minimal system dependencies required for building C/C++ packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire support_assistant package
COPY support_assistant/ /app/support_assistant/

# Configure offline mock baseline as default
ENV MOCK_LLM=1
ENV PYTHONPATH=/app

# Expose port 7860
EXPOSE 7860

# Serve the FastAPI application via uvicorn
CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]