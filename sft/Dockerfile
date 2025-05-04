FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/couchgpt-rag-service-account.json
ENV PYTHONUNBUFFERED=1
ENV TQDM_DISABLE=false

RUN apt-get update && apt-get install -y \
    build-essential curl wget unzip git vim screen \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir pipenv \
    && pip install --no-cache-dir gpustat nvidia-ml-py3 \
    && rm -rf /var/lib/apt/lists/*

# Create user and app directory
RUN useradd -ms /bin/bash app -d /home/app -u 1000 \
    && mkdir -p /app /app/secrets \
    && chown app:app /app /app/secrets

WORKDIR /app

# Install dependencies first as root
ADD Pipfile Pipfile.lock /app/
RUN pipenv sync

# Copy script and fix permissions before switching user
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh && chown app:app /app/docker-entrypoint.sh

# Switch to app user
USER app

# Add source after switching (optional)
COPY --chown=app:app . /app

ENTRYPOINT ["/bin/bash", "/app/docker-entrypoint.sh"]
