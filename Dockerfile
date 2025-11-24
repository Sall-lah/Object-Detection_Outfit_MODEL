FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first (for caching)
COPY API/requirements.txt .

# Install system dependencies needed by OpenCV + YOLO
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Django API source code
COPY API/ .

# Expose Django port
EXPOSE 8000

# Collect static files (optional but recommended)
RUN python manage.py collectstatic --noinput

# Start Django using Gunicorn (production)
CMD ["gunicorn", "projectname.wsgi:application", "--bind", "0.0.0.0:8000"]