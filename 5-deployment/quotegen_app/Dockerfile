# Use a slim Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# --- Dependency Installation ---
# Copy all requirement/config files
COPY pyproject.toml quotegen.toml uv.lock README.md ./
COPY backend/requirements.txt ./backend/requirements.txt

# Install backend dependencies (fastapi, uvicorn) GLOBALLY
# Added --system flag
RUN uv pip install -r /app/backend/requirements.txt --system

# Copy the built wheel from your local 'dist' folder
ARG WHEEL_FILE
COPY dist /app/dist

# Install the quotegen wheel and its dependencies (torch, numpy, etc) GLOBALLY
# Added --system flag
RUN uv pip install /app/dist/quotegen-*.whl --system

# --- Code & Artifacts Copying ---
# Copy the trained model artifacts
COPY artefacts /app/artefacts

# Copy the backend application code
COPY backend /app/backend

# Set the working directory for the backend
WORKDIR /app/backend

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]