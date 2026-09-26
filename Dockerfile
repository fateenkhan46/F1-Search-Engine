# ============================================================
# F1 Smart Search Engine — Dockerfile
# Each instruction creates a "layer". Docker caches layers, so
# things that change rarely go FIRST, code that changes often goes LAST.
# ============================================================

# 1) Base image: a small Linux + Python 3.11. "slim" = smaller, faster pulls.
FROM python:3.11-slim

# 2) Environment variables baked into the image
#    - no .pyc files, logs flushed immediately (so Cloud Run shows them live)
#    - PORT defaults to 8080 (Cloud Run overrides it at runtime)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# 3) All following commands run inside /app in the container
WORKDIR /app

# 4) Copy ONLY requirements first -> this layer is cached until
#    requirements.txt changes, so code edits don't reinstall everything.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 5) Now copy the rest of the project (filtered by .dockerignore)
COPY . .

# 6) Don't run as root (security best practice interviewers ask about)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# 7) Documentation: the port the app listens on
EXPOSE 8080

# 8) Local health check using Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8080\")}/_stcore/health')" || exit 1

# 9) Start command. "sh -c" lets ${PORT} be expanded at runtime.
#    0.0.0.0 = accept traffic from outside the container (localhost would not).
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
