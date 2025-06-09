# Gunakan image python resmi
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Salin semua file ke dalam container
COPY . /app

# Install dependensi sistem (libGL untuk OpenCV)
RUN apt-get update && apt-get install -y libgl1-mesa-glx && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Jalankan aplikasi Flask
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
