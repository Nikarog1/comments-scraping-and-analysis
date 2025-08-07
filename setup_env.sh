#!/bin/bash

export ENV_NAME=$(grep '^name:' environment.yml | awk '{print $2}') 
export PYTHON_VER=$(grep 'python=:' environment.yml | sed 's/.*=//')

if conda env list | grep -q "^$ENV_NAME[[:space:]]"; then
    echo "Updating existing environment: $ENV_NAME"
    conda env update -n $ENV_NAME -f environment.yml --prune
else
    echo "Creating new environment: $ENV_NAME"
    conda env create -f environment.yml
fi

python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ${PYTHON_VER}: ${ENV_NAME}"
echo "Environment '$ENV_NAME' is ready with Python $PYTHON_VER"
