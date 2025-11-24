# Quotegen app — Deployment summary

## Summary
The Quotegen app is a lightweight, full-stack web application designed to generate unique quotes using a Markov Chain model. It features a Python backend (FastAPI) serving a pre-trained machine learning model and a modern, responsive HTML/CSS frontend. The application is containerized with Docker and deployed to Google Cloud Run, leveraging Artifact Registry for image storage and Cloud Build for CI/CD.

## Key Learnings & Achievements
- Cloud Run Deployment: Successfully deployed a stateless containerized application to Google Cloud Run.
- Artifact Registry: Migrated to Artifact Registry for secure Docker image storage (replacing Container Registry).
- Build Optimization: Used `.gcloudignore` effectively to exclude heavy local files (node_modules, venv, Windows .dll files), reducing build upload size from ~600MB to ~40MB.
- Container Configuration:
  - Listens on `0.0.0.0` (not `127.0.0.1`) to accept external traffic.
  - Dynamically handles the `PORT` environment variable provided by Cloud Run.
  - Increased memory allocation to `2GiB` to handle large JSON model files.
- Frontend Integration: Connected a static HTML/JS frontend to the FastAPI backend and configured FastAPI to serve it at the root URL.
- Public Access: Configured IAM policies to allow unauthenticated public access to the service.

## Deployed URL
- Live App: https://quotegen-app-436381107832.europe-west4.run.app/

## Project Structure & Key Files
- Root: `5-deployment/quotegen_app/`
  - Backend: `5-deployment/quotegen_app/backend/`
  - App entry point (FastAPI): `5-deployment/quotegen_app/backend/app.py`
  - Frontend assets (served statically): `5-deployment/quotegen_app/backend/static/` (index.html, styles.css)
  - Dependencies: `5-deployment/quotegen_app/backend/requirements.txt`
  - Source package: `5-deployment/quotegen_app/src/quotegen/` (models, datatools, main)
    - Logic: `models.py` (Markov Chain logic), `datatools.py` (data processing)
  - Artifacts: `5-deployment/quotegen_app/artefacts/` (markov_model.json)
  - Configuration & CI/CD:
    - `5-deployment/quotegen_app/cloudbuild.yaml` (Cloud Build pipeline)
    - `5-deployment/quotegen_app/.gcloudignore` (files to exclude from uploads)
    - `5-deployment/quotegen_app/Dockerfile` (container definition)

## Technical Notes
- Endpoint: The frontend fetches generated quotes from the `/generate` endpoint (example: `/generate?num_words=5&temperature=1.0`).
- Model Loading: The model JSON (~288MB) is loaded into memory on startup; this requires sufficient RAM (configured as 2GiB on Cloud Run).
- Logging: Custom logging is configured to stream to `stdout` so logs appear in Google Cloud Logs.

[Go back to Homepage](../README.md)


