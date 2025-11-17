# Makefile for the Quote Generator App

# Variables
PYTHON = python
UV = uv
PROJECT_NAME = quotegen
WHEEL_FILE = $(shell ls dist/$(PROJECT_NAME)-*.whl 2>/dev/null | head -n 1)
MODEL_FILE = artefacts/markov_model.json
DOCKER_IMAGE = quotegen

# Phony targets
.PHONY: all build_wheel train build_docker run_docker clean

all: run_docker

# Check for wheel, build if not exists
build_wheel:
	@if [ -z "$(WHEEL_FILE)" ]; then \
		echo "Wheel not found. Building..."; \
		$(UV) build --clean; \
	else \
		echo "Wheel already exists: $(WHEEL_FILE)"; \
	fi

# Check for model, train if not exists
train:
	@if [ ! -f "$(MODEL_FILE)" ]; then \
		echo "Model file not found. Training model..."; \
		$(PYTHON) src/quotegen/main.py; \
	else \
		echo "Model file already exists: $(MODEL_FILE)"; \
	fi

# Build the docker image
build_docker: build_wheel train
	@echo "Building Docker image: $(DOCKER_IMAGE)..."
	@# We need to get the wheel filename again after it's been built
	@WHEEL_FILE_POST_BUILD=$$(ls dist/$(PROJECT_NAME)-*.whl 2>/dev/null | head -n 1); \
	if [ -z "$$WHEEL_FILE_POST_BUILD" ]; then \
		echo "Error: Wheel file not found after build."; \
		exit 1; \
	fi; \
	echo "Using wheel: $$WHEEL_FILE_POST_BUILD"; \
	docker build --build-arg WHEEL_FILE=$$WHEEL_FILE_POST_BUILD -t $(DOCKER_IMAGE) .

# Run the docker container
run_docker: build_docker
	@echo "Running Docker container $(DOCKER_IMAGE) on port 8000..."
	@docker run -p 8000:8000 $(DOCKER_IMAGE)

# Clean up build artifacts
clean:
	@echo "Cleaning up..."
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf src/$(PROJECT_NAME).egg-info
	@rm -rf .pytest_cache
	@rm -rf .ruff_cache
	@find . -name '__pycache__' -type d -exec rm -r {} +
	@echo "Cleanup complete."