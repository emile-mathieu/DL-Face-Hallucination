#!/bin/bash

set -e

eval "$(conda shell.bash hook)"

ENV_NAME="DL2"

echo "Checking environment..."

if conda env list | grep -q "$ENV_NAME"; then
    echo "Env $ENV_NAME already exists. Updating..."
    conda env update -f environment.yml --prune
else
    echo "Creating env $ENV_NAME..."
    conda env create -f environment.yml
fi

echo "Activating environment..."
conda activate $ENV_NAME

echo "Preparing dataset..."
python setup_data.py

echo "Setup complete."