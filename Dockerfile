FROM python:3.10-slim

WORKDIR /app

# System libs needed by OpenCV and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pin to v7.0 — this is the branch that matches best.pt and avoids
# the register_pytree_node error caused by newer YOLOv5 master
RUN git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git

COPY best.pt .
COPY server.py .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]