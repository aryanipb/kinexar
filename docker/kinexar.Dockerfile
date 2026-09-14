FROM python:3.11-slim

ARG KINEXAR_DEFAULT_FAMILY=full
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV KINEXAR_DEFAULT_FAMILY=${KINEXAR_DEFAULT_FAMILY}
ENV PYTHONPATH=/app/src:/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    zstd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      "numpy>=1.26,<3" \
      "pyarrow>=18,<20" \
      "scipy>=1.14,<1.17" \
      "pillow>=11,<13" \
      "torch>=2.4,<2.7"

RUN chmod +x /app/docker/kinexar-entrypoint.sh

ENTRYPOINT ["/app/docker/kinexar-entrypoint.sh"]
