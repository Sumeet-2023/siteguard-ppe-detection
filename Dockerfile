FROM python:3.11-slim

# libGL and libglib are required by OpenCV even in headless builds
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

WORKDIR /app
COPY requirements.txt .

# The CPU-only torch index avoids pulling ~2GB of unused CUDA wheels.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY siteguard/ ./siteguard/
COPY models/best.onnx ./models/

USER appuser
EXPOSE 8000
CMD ["uvicorn", "siteguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
