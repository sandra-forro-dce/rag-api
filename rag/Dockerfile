# Use the official Debian-hosted Python image
FROM python:3.11-slim-bookworm

ARG DEBIAN_PACKAGES="build-essential git screen vim"

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python wants UTF-8 locale
ENV LANG=C.UTF-8

# Tell pipenv where the shell is
ENV PYENV_SHELL=/bin/bash

# Disable output buffering
ENV PYTHONUNBUFFERED=1

# Install OS packages, pipenv, create user, and prepare directories
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 -p "$(openssl passwd -1 Passw0rd)" && \
    mkdir -p /app && \
    mkdir -p /persistent && \
    chown app:app /persistent && \
    chown app:app /app

# Switch to non-root user
USER app
WORKDIR /app

# Install Python dependencies
COPY --chown=app:app Pipfile Pipfile.lock /app/
RUN pipenv sync

# ✅ Add your app code (this was missing!)
COPY --chown=app:app . /app

# Make entrypoint script executable (if used)
RUN chmod +x /app/docker-entrypoint.sh

# Default entrypoint (can override in Docker Compose)
ENTRYPOINT ["/bin/bash", "/app/docker-entrypoint.sh"]
