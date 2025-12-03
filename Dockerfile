FROM apache/airflow:2.7.0-python3.10

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python dependencies
COPY requirements.txt /requirements.txt
ARG AIRFLOW_VERSION=2.7.0
ARG PYTHON_VERSION=3.10
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt --constraint "${CONSTRAINT_URL}"

# Install spaCy model (optional but recommended)
RUN python -m spacy download en_core_web_sm || echo "spaCy model installation failed, will use basic tokenizer"

