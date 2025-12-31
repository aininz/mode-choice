FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install runtime deps (CPU-only torch)
COPY requirements.docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.docker.txt

# Copy code
COPY pyproject.toml ./
COPY src/ src/
COPY app/ app/

# Copy runtime assets (dataset + model files)
COPY runtime_assets/ runtime_assets/

# Install package without re-resolving deps
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "/app/app/Home.py", "--server.address=0.0.0.0", "--server.port=8501"]
