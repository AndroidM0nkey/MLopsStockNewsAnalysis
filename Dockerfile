FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y curl && apt-get clean
ENV POETRY_VERSION=1.6.1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry --version

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-setuptools \
    python3-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
RUN poetry install --no-root
COPY . ./
CMD ["poetry", "run", "python", "client.py"]
