### quotegen

This project generates new quotes based on a dataset of 500k+ quotes, using a word-level Markov chain model.

Setup

First, build the environment:

```bash
uv sync --all-extras
```

(We use --all-features to also install fastapi.)

Activate the environment (on UNIX-based systems):

```bash
source .venv/bin/activate
```

Training

To process the data and "train" the model (i.e., build the Markov transition dictionary):

```bash
python src/quotegen/main.py
```

This will:

- Download the manann/quotes-500k dataset from Kaggle.
- Process the quotes.
- Build the Markov model and save it to artefacts/markov_model.json.
- Save the config to artefacts/config.json.

Building the Wheel

If you want to build the quotegen package into a wheel:

```bash
uv build --clean
```

This will produce a dist folder with the wheel file.

Running the Backend API

You can run the backend directly:

```bash
cd backend
python app.py
```

This will start the API at https://www.google.com/search?q=http://127.0.0.1:80.

Building and Running with Docker (Recommended)

A Makefile is provided to automate the process.

Build everything & run the container:

```bash
make run_docker
```

This command will:

- Check for the quotegen wheel and build it if it doesn't exist.
- Check for the markov_model.json and train the model if it doesn't exist.
- Build the Docker image named quotegen.
- Run the container, mapping port 8000 to your local machine.

Access the App:
Once it's running, you can access your app at http://127.0.0.1:8000.

Makefile Targets

- make build_wheel: Builds the python package.
- make train: Downloads data and creates the model artifacts.
- make build_docker: Builds the docker image (depends on wheel and model).
- make run_docker: Runs the built docker image.
