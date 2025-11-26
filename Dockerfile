# ============================
# 1️⃣ STAGE 1 — BUILDER
# ============================
FROM python:3.10 AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only requirements first (for build cache)
COPY API/requirements.txt /app/

# System dependencies (for OpenCV & YOLO compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-dri \
    libjpeg-dev \
    zlib1g-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies into a temporary prefix
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install minimal CPU PyTorch (YOLO dependency)
RUN pip install --no-cache-dir --prefix=/install \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ============================
# 2️⃣ STAGE 2 — RUNTIME
# ============================
FROM python:3.10 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime system dependencies (lighter than builder)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-dri \
    libjpeg62-turbo \
    zlib1g \
    libwebp7 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy Django project code
COPY API/ /app/

# Expose Django port
EXPOSE 8000

# Collect static files (safe even if not used)
RUN python manage.py collectstatic --noinput || true

# Set YOLO to CPU by default via environment variable (optional)
ENV ULTRALYTICS_DEFAULT_DEVICE=cpu

# Start Django with Gunicorn
CMD ["gunicorn", "API.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3"]
