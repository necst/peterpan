#!/bin/bash
set -e  # Stop on any error

MODE=$1

if [[ "$MODE" != "sw" && "$MODE" != "hw" ]]; then
    echo "Usage: $0 <sw|hw>"
    exit 1
fi

DEPLOY_DIR=../3DIRG_wrapper_deploy

# Remove and recreate the deploy directory
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

if [[ "$MODE" == "sw" ]]; then
    make build_pybind PYBIND_MODE=sw
    # Check for the compiled pybind wrapper
    if [ ! -f build/trilli_wrapper.so ]; then
        echo "ERROR: you must compile the pybind wrapper with make build_pybind PYBIND_MODE=sw in order to proceed"
        exit 1
    fi

    # Copy .so, .py and .python-version
    cp -r build "$DEPLOY_DIR/"
    cp *.py "$DEPLOY_DIR/"

    if [ -f .python-version ]; then
        cp .python-version "$DEPLOY_DIR/.python-version"
    else
        echo "WARNING: .python-version not found in current directory. The first time, you should run setup env or manually activate the pybind env"
    fi

else  # MODE == "hw"
    # Check if constants.h exists
    if [ ! -f ../common/constants.h ]; then
        echo "ERROR: you must config the trilli folder to generate constants file"
        exit 1
    fi


    # Copy 3DIRG_application and common
    cd ..
    cp -r 3DIRG_application 3DIRG_wrapper_deploy/
    cp -r common 3DIRG_wrapper_deploy/

    # Now copy the xclbin *inside* the copied application folder
    cp bitstreams/STEP_32IPE.xclbin 3DIRG_wrapper_deploy/3DIRG_application/overlay_hw.xclbin

    # Go back into original directory
    cd 3DIRG_application
fi
