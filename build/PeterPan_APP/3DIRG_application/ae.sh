#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src/sw"

usage() {
    echo "Uso:"
    echo "  SW: $0 SW <opencv_dir> <itk_dir> <fixed_img_or_dir> <moving_img_or_dir> <fig_num> [auto|fixed]"
    echo "  HW: $0 HW <opencv_dir> <itk_dir> <fixed_img_or_dir> <moving_img_or_dir> <xclbin_path> <device_id> <fig_num> [auto|fixed]"
    echo
    echo "  fig_num può essere solo 7 oppure 8"
    echo "  per fig_num=8 puoi scegliere opzionalmente:"
    echo "    auto  -> Dz=32, zmode=auto"
    echo "    fixed -> Dz=numero slice, zmode=fixed, zstart=0, zend=Z-1"
    exit 1
}

if [ "$#" -lt 6 ]; then
    usage
fi

MODE="$1"
OPENCV_DIR="$2"
ITK_DIR="$3"
FIXED_IMG="$4"
MOVING_IMG="$5"

if [ ! -d "$SRC_DIR" ]; then
    echo "Errore: directory sorgente non trovata: $SRC_DIR"
    exit 1
fi

if [ ! -d "$OPENCV_DIR" ]; then
    echo "Errore: OpenCV_DIR non trovato: $OPENCV_DIR"
    exit 1
fi

if [ ! -d "$ITK_DIR" ]; then
    echo "Errore: ITK_DIR non trovato: $ITK_DIR"
    exit 1
fi

setup_figure7_env() {
    local FIG7_DIR="$1"
    local PYTHON_VERSION="3.11.4"
    local ENV_NAME="figure7"
    local REQUIREMENTS_FILE="$FIG7_DIR/requirements.txt"

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

    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        echo "Errore: requirements.txt non trovato: $REQUIREMENTS_FILE"
        exit 1
    fi

    echo "Installing Python packages from $REQUIREMENTS_FILE ..."
    PYENV_VERSION="$ENV_NAME" pyenv exec python -m pip install --upgrade pip
    PYENV_VERSION="$ENV_NAME" pyenv exec python -m pip install -r "$REQUIREMENTS_FILE"
}

setup_figure8_env() {
    local FIG8_DIR="$1"
    local PYTHON_VERSION="3.11.4"
    local ENV_NAME="figure8"
    local REQUIREMENTS_FILE="$FIG8_DIR/requirements.txt"

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

    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        echo "Errore: requirements.txt non trovato: $REQUIREMENTS_FILE"
        exit 1
    fi

    echo "Installing Python packages from $REQUIREMENTS_FILE ..."
    PYENV_VERSION="$ENV_NAME" pyenv exec python -m pip install --upgrade pip
    PYENV_VERSION="$ENV_NAME" pyenv exec python -m pip install -r "$REQUIREMENTS_FILE"
}

run_figure7() {
    local BIN="$1"
    local CT_DIR="$2"
    local PET_DIR="$3"
    local XCLBIN="$4"
    local DEVICE_ID="$5"

    local OUT_DIR="$SCRIPT_DIR/Figure_7"
    local LOG_DIR="$OUT_DIR/logs"
    local FIG7_PY="$OUT_DIR/figure7.py"
    local REQUIREMENTS_FILE="$OUT_DIR/requirements.txt"

    local MASK="otsu"
    local TYPE="uint8"
    local GRADIENT="manual"
    local NUM_ITER="5"
    local OMP_THREADS="32"
    local N_RUNS="1"

    mkdir -p "$LOG_DIR"

    if [ ! -d "$OUT_DIR" ]; then
        echo "Errore: directory Figure_7 non trovata: $OUT_DIR"
        exit 1
    fi

    if [ ! -f "$FIG7_PY" ]; then
        echo "Errore: script figure7.py non trovato: $FIG7_PY"
        exit 1
    fi

    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        echo "Errore: file requirements.txt non trovato: $REQUIREMENTS_FILE"
        exit 1
    fi

    local Z_CT
    local Z_PET

    Z_CT=$(find "$CT_DIR" -maxdepth 1 -type f | grep -Ei '\.png$' | wc -l)
    Z_PET=$(find "$PET_DIR" -maxdepth 1 -type f | grep -Ei '\.png$' | wc -l)

    if [[ "$Z_CT" -ne "$Z_PET" ]]; then
        echo "Errore: numero di slice diverso tra CT ($Z_CT) e PET ($Z_PET)" >&2
        exit 1
    fi

    local Z="$Z_CT"
    echo "Volume con Z = $Z slice"

    cd "$OUT_DIR"

    for DZ in 32 64; do
        echo
        echo "==============================="
        echo "      TEST CON DZ = $DZ (zmode=auto)"
        echo "==============================="
        echo "Volume con Z = $Z slice, DZ = $DZ"

        for NUM_LEVELS in {1..8}; do
            echo "######## num_levels = $NUM_LEVELS (DZ = $DZ, zmode=auto) ########"

            for (( run=1; run<=N_RUNS; run++ )); do
                echo ">>> Run DZ=$DZ, num_levels=$NUM_LEVELS, repetition=$run / $N_RUNS"

                LOG="$LOG_DIR/run_auto_DZ${DZ}_levels${NUM_LEVELS}_rep${run}.log"

                OMP_NUM_THREADS="$OMP_THREADS" "$BIN" \
                    "$CT_DIR" \
                    "$PET_DIR" \
                    "$DZ" \
                    --mask="$MASK" \
                    --type="$TYPE" \
                    --xclbin="$XCLBIN" \
                    --gradient="$GRADIENT" \
                    --zmode=auto \
                    --num_iter="$NUM_ITER" \
                    --num_levels="$NUM_LEVELS" \
                    --device_id="$DEVICE_ID" \
                    >"$LOG" 2>&1
            done
        done
    done

    echo
    echo "Setting up pyenv environment for Figure 7..."
    setup_figure7_env "$OUT_DIR"

    echo "Running Figure 7 Python script..."
    PYENV_VERSION="figure7" pyenv exec python "$FIG7_PY"
}

run_figure8() {
    local BIN="$1"
    local REF_FOLDER="$2"
    local FLT_FOLDER="$3"
    local MODE_LOCAL="$4"
    local FIG8_RUN_MODE="${5:-auto}"
    local XCLBIN="${6:-}"
    local DEVICE_ID="${7:-}"

    local FIG8_DIR="$SCRIPT_DIR/Figure_8"
    local OUTPUT_LINK="$FIG8_DIR/output_folder"
    local FIG8_PY="$FIG8_DIR/figure8.py"
    local FIG8_REQUIREMENTS="$FIG8_DIR/requirements.txt"

    local TYPE="png"
    local GRADIENT="itk"
    local MASK="otsu"
    local NUM_LEVELS="4"
    local NUM_ITER="5"

    if [ ! -d "$REF_FOLDER" ]; then
        echo "Errore: reference folder non trovata: $REF_FOLDER"
        exit 1
    fi

    if [ ! -d "$FLT_FOLDER" ]; then
        echo "Errore: floating folder non trovata: $FLT_FOLDER"
        exit 1
    fi

    mkdir -p "$FIG8_DIR"

    local VOLUME_DEPTH_REF
    local VOLUME_DEPTH_FLT

    VOLUME_DEPTH_REF=$(find "$REF_FOLDER" -maxdepth 1 -type f | grep -Ei '\.png$' | wc -l)
    VOLUME_DEPTH_FLT=$(find "$FLT_FOLDER" -maxdepth 1 -type f | grep -Ei '\.png$' | wc -l)

    if [[ "$VOLUME_DEPTH_REF" -ne "$VOLUME_DEPTH_FLT" ]]; then
        echo "Errore: numero di slice diverso tra reference ($VOLUME_DEPTH_REF) e floating ($VOLUME_DEPTH_FLT)" >&2
        exit 1
    fi

    if [[ "$VOLUME_DEPTH_REF" -eq 0 ]]; then
        echo "Errore: nessuna slice PNG trovata nelle directory di input" >&2
        exit 1
    fi

    local DZ
    local ZMODE
    local ZSTART=0
    local ZEND=$((VOLUME_DEPTH_REF - 1))

    case "$FIG8_RUN_MODE" in
        auto)
            DZ="32"
            ZMODE="auto"
            ;;
        fixed)
            DZ="$VOLUME_DEPTH_REF"
            ZMODE="fixed"
            ;;
        *)
            echo "Errore: modalità Figure 8 non valida: $FIG8_RUN_MODE (usa auto o fixed)"
            exit 1
            ;;
    esac

    echo "Reference folder: $REF_FOLDER"
    echo "Floating folder: $FLT_FOLDER"
    echo "Figure 8 working dir: $FIG8_DIR"
    echo "Volume depth: $VOLUME_DEPTH_REF"
    echo "Figure 8 mode: $FIG8_RUN_MODE"
    echo "Dz: $DZ"

    cd "$FIG8_DIR"
    mkdir -p output

    if [ "$MODE_LOCAL" = "HW" ]; then
        if [ -z "$XCLBIN" ] || [ -z "$DEVICE_ID" ]; then
            echo "Errore: in modalità HW per Figure 8 servono xclbin e device_id"
            exit 1
        fi

        if [ "$ZMODE" = "fixed" ]; then
            "$BIN" \
                "$REF_FOLDER" \
                "$FLT_FOLDER" \
                "$DZ" \
                --type="$TYPE" \
                --mask="$MASK" \
                --gradient="$GRADIENT" \
                --zmode=fixed \
                --zstart="$ZSTART" \
                --zend="$ZEND" \
                --num_levels="$NUM_LEVELS" \
                --num_iter="$NUM_ITER" \
                --xclbin="$XCLBIN" \
                --device_id="$DEVICE_ID"
        else
            "$BIN" \
                "$REF_FOLDER" \
                "$FLT_FOLDER" \
                "$DZ" \
                --type="$TYPE" \
                --mask="$MASK" \
                --gradient="$GRADIENT" \
                --zmode=auto \
                --num_levels="$NUM_LEVELS" \
                --num_iter="$NUM_ITER" \
                --xclbin="$XCLBIN" \
                --device_id="$DEVICE_ID"
        fi
    else
        if [ "$ZMODE" = "fixed" ]; then
            "$BIN" \
                "$REF_FOLDER" \
                "$FLT_FOLDER" \
                "$DZ" \
                --type="$TYPE" \
                --mask="$MASK" \
                --gradient="$GRADIENT" \
                --zmode=fixed \
                --zstart="$ZSTART" \
                --zend="$ZEND" \
                --num_levels="$NUM_LEVELS" \
                --num_iter="$NUM_ITER"
        else
            "$BIN" \
                "$REF_FOLDER" \
                "$FLT_FOLDER" \
                "$DZ" \
                --type="$TYPE" \
                --mask="$MASK" \
                --gradient="$GRADIENT" \
                --zmode=auto \
                --num_levels="$NUM_LEVELS" \
                --num_iter="$NUM_ITER"
        fi
    fi

    rm -rf "$OUTPUT_LINK"
    ln -sfn "$FIG8_DIR/output" "$OUTPUT_LINK"

    if [ "$FIG8_RUN_MODE" = "auto" ]; then
        local CSV_SRC="$FIG8_DIR/output_folder/timings_dz32_levels4.csv"
        local CSV_DST="$FIG8_DIR/timings_dz32_levels4.csv"

        if [ ! -f "$CSV_SRC" ]; then
            echo "Errore: file timings non trovato: $CSV_SRC"
            exit 1
        fi

        cp "$CSV_SRC" "$CSV_DST"
        echo "Copiato: $CSV_SRC -> $CSV_DST"

        if [ ! -f "$FIG8_PY" ]; then
            echo "Errore: script figure8.py non trovato: $FIG8_PY"
            exit 1
        fi

        if [ ! -f "$FIG8_REQUIREMENTS" ]; then
            echo "Errore: requirements.txt non trovato: $FIG8_REQUIREMENTS"
            exit 1
        fi

        echo "Setting up pyenv environment for Figure 8..."
        setup_figure8_env "$FIG8_DIR"

        echo "Running Figure 8 Python script..."
        PYENV_VERSION="figure8" pyenv exec python "$FIG8_PY"
    fi

    echo "Output disponibile in: $FIG8_DIR/output"
}

if [ "$MODE" = "SW" ]; then
    FIG_NUM="$6"
    FIG8_MODE="${7:-auto}"

    if [ "$FIG_NUM" != "7" ] && [ "$FIG_NUM" != "8" ]; then
        echo "Errore: fig_num deve essere 7 oppure 8"
        exit 1
    fi

    BUILD_DIR="$SCRIPT_DIR/build_sw"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    echo "Preparing SW Registration..."

    cmake "$SRC_DIR" \
        -DOpenCV_DIR="$OPENCV_DIR" \
        -DITK_DIR="$ITK_DIR"

    make -j

    BIN="$BUILD_DIR/peterpan_pyramidal_sw"

    if [ "$FIG_NUM" = "7" ]; then
        echo "Errore: Figure 7 richiede modalità HW"
        exit 1
    else
        run_figure8 "$BIN" "$FIXED_IMG" "$MOVING_IMG" "SW" "$FIG8_MODE"
    fi

elif [ "$MODE" = "HW" ]; then
    if [ "$#" -lt 8 ]; then
        echo "Errore: in modalità HW devi specificare anche xclbin, device_id e fig_num"
        usage
    fi

    XCLBIN_PATH="$6"
    DEVICE_ID="$7"
    FIG_NUM="$8"
    FIG8_MODE="${9:-auto}"

    if [ "$FIG_NUM" != "7" ] && [ "$FIG_NUM" != "8" ]; then
        echo "Errore: fig_num deve essere 7 oppure 8"
        exit 1
    fi

    BUILD_DIR="$SCRIPT_DIR/build_hw"

    if [ ! -f "$XCLBIN_PATH" ]; then
        echo "Errore: xclbin non trovato: $XCLBIN_PATH"
        exit 1
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    echo "Preparing HW Registration..."

    cmake "$SRC_DIR" \
        -DOpenCV_DIR="$OPENCV_DIR" \
        -DITK_DIR="$ITK_DIR" \
        -DHW_REG=ON

    make -j

    BIN="$BUILD_DIR/peterpan_pyramidal_sw"

    if [ "$FIG_NUM" = "7" ]; then
        run_figure7 "$BIN" "$FIXED_IMG" "$MOVING_IMG" "$XCLBIN_PATH" "$DEVICE_ID"
    else
        run_figure8 "$BIN" "$FIXED_IMG" "$MOVING_IMG" "HW" "$FIG8_MODE" "$XCLBIN_PATH" "$DEVICE_ID"
    fi

else
    echo "Errore: il primo parametro deve essere SW oppure HW"
    usage
fi
