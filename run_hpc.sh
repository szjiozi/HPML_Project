#!/bin/bash
#
#
#SBATCH --job-name=long_refiner_job   # Job name
#SBATCH --account=ece_gy_9143-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=slurm_logs/job_%j.out      # %j will be replaced with the Job ID
#SBATCH --error=slurm_logs/job_%j.err       # %j will be replaced with the Job ID
#SBATCH --time=04:00:00                 # Maximum job run time (HH:MM:SS)
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --ntasks-per-node=1             # 1 task per node
#SBATCH --cpus-per-task=8               # Request 8 CPU cores per task (for data loading)
#SBATCH --mem=64G                       # Request 64GB of memory
#SBATCH --gres=gpu:1                    # **Most critical: Request 1 GPU**
#

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

# Main execution function
main_execution() {
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
        # Use . instead of source for sh compatibility, and use the correct path
        if [ -f "$HOME/.local/bin/env" ]; then
            . "$HOME/.local/bin/env"
        fi
        # Also add to PATH directly as fallback
        export PATH="$HOME/.local/bin:$PATH"
        echo "uv installed successfully."
    else
        echo "Found."
    fi

    # 4. Sync project environment and dependencies
    echo "Step 4: Syncing project environment with pyproject.toml..."
    # 'uv sync' will create the .venv if it doesn't exist and install all dependencies,
    # including the project itself in editable mode by default.
    # Use the system Python version that was detected earlier
    uv sync --python "$PYTHON_VERSION"

    echo "Environment is up to date."

    # 5. Run the quick start script using the project environment
    echo "Step 5: Running the quick start script with 'uv run'..."
    echo "-------------------------------------------"
    # 'uv run' automatically uses the project's .venv, no 'source' activation is needed.
    bash scripts/evaluation/run_eval.sh
    echo "-------------------------------------------"
    echo "Script finished successfully."

    echo "===== Execution Complete ====="
}

# Change to the script's directory to ensure relative paths work correctly
# Use a method compatible with both bash and sh
if [ -n "$BASH_SOURCE" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    # Fallback for sh
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
cd "$SCRIPT_DIR" || {
    echo "ERROR: Cannot change to script directory: $SCRIPT_DIR"
    exit 1
}

# Check if we're already inside Singularity
INSIDE_SINGULARITY=false
if [ -n "$SINGULARITY_NAME" ] || [ -n "$APPTAINER_NAME" ]; then
    INSIDE_SINGULARITY=true
fi

# Pre-execution setup (always execute, even inside Singularity)
echo "===== LongRefiner HPC Setup ====="

# 0. Setup environment file
ENV_FILE="$SCRIPT_DIR/.env"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"
echo -n "Step 0: Setting up .env file... "
if [ ! -f "$ENV_FILE" ]; then
    if [ ! -f "$ENV_EXAMPLE" ]; then
        echo "WARNING: .env.example not found. Creating empty .env file."
        touch "$ENV_FILE"
        echo "Created empty .env file."
    else
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        echo "Created .env from .env.example"
    fi
else
    echo "Found existing .env"
fi

# Load environment variables from .env (only if file exists and is readable)
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ] && [ -r "$ENV_FILE" ]; then
    set -a
    # Use . instead of source for sh compatibility, with absolute path
    . "$ENV_FILE"
    set +a
else
    echo "WARNING: .env file not found or not readable at $ENV_FILE. Continuing without loading environment variables."
fi

# 0.1. Update repository with git pull
echo -n "Step 0.1: Updating repository... "
if [ -d ".git" ]; then
    # Check if a specific branch is requested
    if [ -n "$GH_BRANCH" ]; then
        CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
        if [ "$CURRENT_BRANCH" != "$GH_BRANCH" ]; then
            echo "Switching to branch $GH_BRANCH..."
            git checkout "$GH_BRANCH" 2>/dev/null || git checkout -b "$GH_BRANCH" 2>/dev/null || echo "Warning: Could not switch to branch $GH_BRANCH"
        fi
    fi
    # Pull latest changes
    git pull
    echo "Repository updated."
else
    echo "Not a git repository, skipping update."
fi

# 0.2. Execute Singularity bash if specified and not already inside Singularity
if [ "$INSIDE_SINGULARITY" = "false" ] && [ -n "$SINGULARITY_BASH_PATH" ]; then
    echo "Step 0.2: Executing in Singularity environment..."
    echo "-------------------------------------------"
    if [ -f "$SINGULARITY_BASH_PATH" ]; then
        # Use the SCRIPT_DIR already defined at the top of the script
        SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]:-$0}")"
        # Execute the script inside Singularity by passing command via stdin
        # The Singularity bash script will execute the command in the Singularity environment
        echo "cd '$SCRIPT_DIR' && bash '$SCRIPT_PATH'" | bash "$SINGULARITY_BASH_PATH"
        exit $?
    else
        echo "ERROR: Singularity bash script not found at $SINGULARITY_BASH_PATH"
        exit 1
    fi
else
    # Execute directly (either no Singularity path specified, or already inside Singularity)
    if [ "$INSIDE_SINGULARITY" = "true" ]; then
        echo "Already inside Singularity, proceeding with main execution..."
    fi
    main_execution
fi
