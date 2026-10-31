FROM python:3.11-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model artifacts are copied into service/model/ by the CD workflow (or manually) before build.
COPY app ./app
COPY model ./model

RUN useradd -m appuser
USER appuser

ENV PORT=8080 MODEL_DIR=/srv/model
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
