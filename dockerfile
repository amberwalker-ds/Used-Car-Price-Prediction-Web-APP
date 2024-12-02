FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Set PYTHONPATH to include /app for module imports
ENV PYTHONPATH=/app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will listen on
EXPOSE 8080

# Run the Flask app
CMD ["python", "api/app.py"]