#!/bin/bash

# Exit on error
set -e

# Desired Python version
PYTHON_VERSION=3.11.4
VENV_NAME=pybind11_env

echo "Installing Python $PYTHON_VERSION with pyenv (if not already installed)..."
pyenv install -s $PYTHON_VERSION

echo "Creating virtualenv $VENV_NAME..."
pyenv virtualenv -f $PYTHON_VERSION $VENV_NAME

echo "Setting local environment to $VENV_NAME..."
pyenv local $VENV_NAME

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install pybind11 numpy

echo "Environment setup complete."