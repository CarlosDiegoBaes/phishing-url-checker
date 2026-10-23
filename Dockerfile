# Serves the phishguard web UI on port 8000 with the saved model.joblib.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY phishguard/ phishguard/
COPY model.joblib .

RUN useradd --create-home --uid 1000 app
USER 1000

EXPOSE 8000

CMD ["uvicorn", "--factory", "phishguard.web:create_app", "--host", "0.0.0.0", "--port", "8000"]
