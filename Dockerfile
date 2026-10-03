FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[kafka]"

COPY config ./config
COPY contracts ./contracts

# data/, models/ and reports/ are expected as volumes so artifacts survive the container
VOLUME ["/app/data", "/app/models", "/app/reports"]

ENTRYPOINT ["fraudpipe"]
CMD ["run-all"]
