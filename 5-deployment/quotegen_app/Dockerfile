# Use a slim Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml quotegen.toml uv.lock README.md ./

# Install project dependencies (excluding the project itself)
# This caches the dependencies layer
RUN uv sync --all-extras

# Argument for the wheel file
ARG WHEEL_FILE

# Copy the built wheel and install it
COPY dist /app/dist
RUN pip install /app/dist/quotegen-*.whl

# Copy the trained model artifacts
COPY artefacts /app/artefacts

# Copy the backend application code
COPY backend /app/backend

# Install backend-specific requirements
RUN uv pip install -r /app/backend/requirements.txt

# Set the working directory for the backend
WORKDIR /app/backend

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]