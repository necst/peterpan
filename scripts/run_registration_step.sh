#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST_BIN="$SCRIPT_DIR/host_overlay.exe"
DATASET_GEN="$SCRIPT_DIR/generate_dataset.sh"
FIG6_PY="$SCRIPT_DIR/figure6.py"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

n_row=512
n_col=512
n_couples_list=(256)
device_id="${1:-0}"
tx=30
ty=30
ang=25
reps=50

setup_figure6_env() {
    local WORK_DIR="$1"
    local PYTHON_VERSION="3.11.4"
    local ENV_NAME="figure6"
    local REQUIREMENTS_FILE_LOCAL="$WORK_DIR/requirements.txt"

    if ! command -v pyenv >/dev/null 2>&1; then
        echo "Errore: pyenv non trovato nel PATH"
        exit 1
    fi

    export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
    export PATH="$PYENV_ROOT/bin:$PATH"

    eval "$(pyenv init -)"

    if [ -d "$PYENV_ROOT/plugins/pyenv-virtualenv" ]; then
        eval "$(pyenv virtualenv-init -)"
    else
        echo "Errore: plugin pyenv-virtualenv non trovato in $PYENV_ROOT/plugins/pyenv-virtualenv"
        exit 1
    fi

    if ! pyenv versions --bare | grep -Fxq "$PYTHON_VERSION"; then
        echo "Installing Python $PYTHON_VERSION with pyenv..."
        pyenv install "$PYTHON_VERSION"
    else
        echo "Python $PYTHON_VERSION already available"
    fi

    if ! pyenv versions --bare | grep -Fxq "$ENV_NAME"; then
        echo "Creating virtualenv $ENV_NAME..."
        pyenv virtualenv "$PYTHON_VERSION" "$ENV_NAME"
    else
        echo "Virtualenv $ENV_NAME already exists"
    fi

    if [ ! -f "$REQUIREMENTS_FILE_LOCAL" ]; then
        echo "Errore: requirements.txt non trovato: $REQUIREMENTS_FILE_LOCAL"
        exit 1
    fi

    echo "Installing Python packages from $REQUIREMENTS_FILE_LOCAL ..."
    PYENV_VERSION="$ENV_NAME" pyenv exec python -m pip install --upgrade pip
    PYENV_VERSION="$ENV_NAME" pyenv exec python -m pip install -r "$REQUIREMENTS_FILE_LOCAL"
}

# ============================================================
# Checks
# ============================================================

if [ ! -f "$HOST_BIN" ]; then
    echo "Errore: eseguibile non trovato: $HOST_BIN"
    exit 1
fi

if [ ! -x "$HOST_BIN" ]; then
    echo "Errore: eseguibile non eseguibile: $HOST_BIN"
    exit 1
fi

if [ ! -f "$DATASET_GEN" ]; then
    echo "Errore: script dataset non trovato: $DATASET_GEN"
    exit 1
fi

if [ ! -x "$DATASET_GEN" ]; then
    echo "Errore: script dataset non eseguibile: $DATASET_GEN"
    exit 1
fi

if [ ! -f "$FIG6_PY" ]; then
    echo "Errore: script figure6.py non trovato: $FIG6_PY"
    exit 1
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Errore: requirements.txt non trovato: $REQUIREMENTS_FILE"
    exit 1
fi

cd "$SCRIPT_DIR"

echo "Generating dataset..."
"$DATASET_GEN" "$n_row" "$n_col" 512

for n_couples in "${n_couples_list[@]}"; do
    echo "> Running for depth = $n_couples"
    "$HOST_BIN" "$n_couples" "$n_row" "$n_col" "$tx" "$ty" "$ang" "$reps" "$device_id"
    echo "--------------------"
done

echo
echo "Setting up pyenv environment for Figure 6..."
setup_figure6_env "$SCRIPT_DIR"

echo "Running Figure 6 Python script..."
PYENV_VERSION="figure6" pyenv exec python "$FIG6_PY"

echo "DONE"