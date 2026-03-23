#!/bin/bash

# MIT License

# Copyright (c) 2025 Paolo Salvatore Galfano, Giuseppe Sorrentino

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

source /opt/xrt/2023.1/setup.sh
source /home/xilinx/Vitis/2023.1/settings64.sh

# Install pre-commit if not present ("system-wide" for vs-code integration)
if ! command -v pre-commit &> /dev/null
then
    pip install -q pre-commit
    
    # Check if installation was successful
    if ! command -v pre-commit &> /dev/null
    then
        echo "Warning: pre-commit not installed" >&2
    fi
fi

pre-commit install &> /dev/null

if [ ! -d ".venv" ]; then
  make python-venv
fi

if [ -z "$VIRTUAL_ENV" ] || [ "$VIRTUAL_ENV" != "$(realpath .venv)" ]; then
  source .venv/bin/activate
fi

