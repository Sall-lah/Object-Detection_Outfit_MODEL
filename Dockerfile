FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory for the Django API
WORKDIR /app

# Copy requirements first for caching
COPY API/requirements.txt /app/

# Install system dependencies needed for OpenCV & YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-dri \
    libjpeg-dev \
    zlib1g-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install all python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire API folder into the container
COPY API/ /app/

# Expose Django port
EXPOSE 8000

# Collect static files (safe even if you don't use static)
RUN python manage.py runserver 0.0.0.0:8000

# Run Django using Gunicorn
CMD ["gunicorn", "API.wsgi:application", "--bind", "0.0.0.0:8000"]
