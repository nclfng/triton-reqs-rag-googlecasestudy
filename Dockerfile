FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py courses.py ./
COPY static ./static
COPY data/index.json data/vectors.npy data/courses.json ./data/
# Hugging Face Spaces uses 7860; Render sets $PORT.
CMD ["sh", "-c", "uvicorn app:app --no-access-log --host 0.0.0.0 --port ${PORT:-7860}"]
