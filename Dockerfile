# Dockerfile
# Stage 1: Build dependency wheel compilation context
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --user .

# Stage 2: Final lightweight target execution container
FROM python:3.10-slim AS runner

WORKDIR /app

# Enforce explicit global runtime parameters for trading isolation
ENV TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN groupadd -r quantuser && useradd -r -g quantuser quantuser

# Pull the compiled site packages from the builder stage
COPY --from=builder /root/.local /home/quantuser/.local
COPY --chown=quantuser:quantuser config.py data_engine.py model.py train.py test.py ./

ENV PATH=/home/quantuser/.local/bin:$PATH

USER quantuser

ENTRYPOINT ["python", "test.py"]