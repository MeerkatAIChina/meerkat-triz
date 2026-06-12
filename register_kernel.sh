#!/bin/bash
# Register venv_v5 as a Jupyter kernel on DGX Spark

VENV_PATH="/home/chinux/jupyterlab/meerkatai/venv_v5"
KERNEL_NAME="venv_v5"
DISPLAY_NAME="Python (venv_v5)"

echo "=== Registering Jupyter Kernel ==="
echo "Venv path: $VENV_PATH"
echo "Kernel name: $KERNEL_NAME"
echo ""

# Check venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Install ipykernel
echo "Installing ipykernel..."
pip install ipykernel

# Register kernel
echo "Registering kernel '$KERNEL_NAME'..."
python -m ipykernel install --user --name="$KERNEL_NAME" --display-name "$DISPLAY_NAME"

# Verify
echo ""
echo "=== Installed Kernels ==="
jupyter kernelspec list

echo ""
echo "=== Done ==="
echo "Open JupyterLab, click Kernel -> Change kernel, and select '$DISPLAY_NAME'"
