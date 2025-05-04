FROM python:3.11-slim-buster

ENV PYENV_SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential git && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    mkdir -p /app

WORKDIR /app

# Install dependencies as root
ADD Pipfile Pipfile.lock /app/
RUN pipenv sync

# Copy entrypoint and fix permissions
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Optional: add source
ADD . /app

ENTRYPOINT ["/bin/bash", "/app/docker-entrypoint.sh"]
