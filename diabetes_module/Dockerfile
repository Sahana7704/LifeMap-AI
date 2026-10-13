FROM python:3.11-slim

WORKDIR /app

# Create non-root system user
RUN adduser --disabled-password --gecos "" appuser

# Install serving dependencies only (pinned versions, no shap/torch/matplotlib)
COPY requirements_serving.txt .
RUN pip install --no-cache-dir -r requirements_serving.txt

# Copy source code and pre-trained models
COPY src/ ./src/
COPY models/ ./models/

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PORT=8005
ENV DIABETES_MODELS_DIR=/app/models

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8005)}/health')"

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8005}"]
