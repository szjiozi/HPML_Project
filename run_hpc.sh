#!/bin/bash
set -e

# --- Configuration ---
# Set the Python version required by the project
REQUIRED_PYTHON_VERSION="3.9"
# --- End Configuration ---

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to compare version strings (handles versions like 3.10.1)
version_ge() {
    [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

echo "===== LongRefiner HPC Execution Script ====="

# 1. Check for CUDA
echo -n "Step 1: Checking for CUDA... "
if ! command_exists nvidia-smi || ! nvidia-smi | grep -q "CUDA Version"; then
    echo "ERROR: CUDA not found."
    echo "Please ensure NVIDIA drivers are installed and CUDA is available in your environment."
    exit 1
fi
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9.]*\).*/\1/')
echo "Found CUDA Version $CUDA_VERSION"

# 2. Check for Python version
echo -n "Step 2: Checking for Python version... "
if ! command_exists python3; then
    echo "ERROR: python3 command not found."
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if ! version_ge "$PYTHON_VERSION" "$REQUIRED_PYTHON_VERSION"; then
    echo "ERROR: Python version $REQUIRED_PYTHON_VERSION or higher is required. Found $PYTHON_VERSION."
    exit 1
fi
echo "Found Python $PYTHON_VERSION"

# 3. Check for uv and install if not present
echo -n "Step 3: Checking for uv... "
if ! command_exists uv; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to the current shell's PATH
    source "$HOME/.cargo/env"
    echo "uv installed successfully."
else
    echo "Found."
fi

# 4. Sync project environment and dependencies
echo "Step 4: Syncing project environment with pyproject.toml..."
# 'uv sync' will create the .venv if it doesn't exist and install all dependencies,
# including the project itself in editable mode by default.
uv sync

echo "Environment is up to date."

# 5. Run the quick start script using the project environment
echo "Step 5: Running the quick start script with 'uv run'..."
echo "-------------------------------------------"
# 'uv run' automatically uses the project's .venv, no 'source' activation is needed.
uv run python scripts/quick_start.py
echo "-------------------------------------------"
echo "Script finished successfully."

echo "===== Execution Complete ====="
