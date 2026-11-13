FROM python:3.12-slim

WORKDIR /code

# Install system dependencies needed for LightGBM, XGBoost, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create standard non-root user with UID 1000 required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR $HOME/app

COPY --chown=user:user . $HOME/app

EXPOSE 7860

CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:7860"]
