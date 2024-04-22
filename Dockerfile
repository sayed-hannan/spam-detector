# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

WORKDIR /app

# Copy the project files into the container
COPY . /app


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Specify the command to run when the container starts
CMD ["python", "src/main.py"]

EXPOSE 80