# Use an official Python runtime as a parent image
FROM python:3.8

WORKDIR /app

# Copy the project files into the container
COPY . /src/
COPY . data/spam.csv


# Install Python dependencies
RUN pip install -r requirements.txt


# Specify the command to run when the container starts
CMD ["python", "src/main.py"]

EXPOSE 8000