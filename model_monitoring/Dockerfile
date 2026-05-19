FROM python:3.11-slim

WORKDIR /app

# Install Evidently UI
RUN pip install --no-cache-dir "evidently[ui]"

# Create workspace directory
RUN mkdir -p /app/workspace

EXPOSE 8000

# Start Evidently UI
CMD ["evidently", "ui", "--host", "0.0.0.0", "--port", "8000", "--workspace", "/app/workspace"]