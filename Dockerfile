FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# The base image already sets the WORKDIR to /app

# Copy dependencies file
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the entire 'app' directory which should contain 'main.py'
COPY ./app /app/app