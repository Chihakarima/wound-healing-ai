FROM python:3.11-slim

# opencv-python (headless import path) needs libGL.so.1 and libglib at runtime,
# not just at pip-install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# CPU-only wheels first: the default PyPI torch/torchvision wheels pull in
# CUDA dependencies (~2GB) that a CPU-only container never uses. requirements.txt
# also lists torch/torchvision unpinned, so the later install below is a no-op
# for them and only installs the remaining packages.
RUN pip install --no-cache-dir --default-timeout=180 --retries 10 \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --default-timeout=180 --retries 10 -r requirements.txt

COPY . .

EXPOSE 8501

# data/, outputs/ and chatbot/chroma_db are excluded from the build context
# (see .dockerignore) and expected to be bind-mounted at runtime (see
# docker-compose.yml) rather than baked into the image.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
