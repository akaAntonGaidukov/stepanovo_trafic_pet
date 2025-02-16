FROM python:3.9-slim

# Install system dependencies + CA certificates
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy PostgreSQL SSL certificates
COPY postgres-certs/server.crt /usr/local/share/ca-certificates/postgres.crt
RUN update-ca-certificates

COPY yolo_s_best_result_3000.pt .
COPY mask.png .

COPY . .

CMD ["python", "main.py"]