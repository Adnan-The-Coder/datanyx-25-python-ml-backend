FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Force the server to run on port 8000 instead of default 80
ENV PORT=8000

# The base image already sets WORKDIR to /app

# Copy dependencies file
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the ML models module
COPY ./ml_models.py /app/ml_models.py

# Copy the trained models directory
COPY ./models /app/models

# Copy the data directory
COPY ./data /app/data

# Copy the FastAPI app directory
COPY ./app /app/app

# Tell Docker the container listens on port 8000
EXPOSE 8000
