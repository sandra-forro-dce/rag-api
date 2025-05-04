FROM python:3.11-slim-bookworm

ARG DEBIAN_PACKAGES="build-essential git screen vim"

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYENV_SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1

RUN set -ex; \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 -p "$(openssl passwd -1 Passw0rd)" && \
    mkdir -p /app /persistent && \
    chown app:app /persistent /app

WORKDIR /app

# Install dependencies and set up script as root
ADD Pipfile Pipfile.lock /app/
RUN pipenv sync

COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh && chown app:app /app/docker-entrypoint.sh

# Switch to non-root user
USER app

ENTRYPOINT ["/bin/bash", "/app/docker-entrypoint.sh"]
