FROM python:3.10-slim

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project
WORKDIR /app
COPY . .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Collect static if needed
RUN python manage.py collectstatic --noinput || true

# Expose port
EXPOSE 8000

# Run the Django app
CMD ["gunicorn", "yourproject.wsgi:application", "--bind", "0.0.0.0:8000"]