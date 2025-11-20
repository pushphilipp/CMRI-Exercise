#!/bin/bash
# CMRI Exercise Environment Activation Script
# Usage: source activate_env.sh

# Check if script is being sourced (not executed)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "❌ ERROR: This script must be sourced, not executed!"
    echo "Usage: source activate_env.sh"
    echo "   or: . activate_env.sh"
    exit 1
fi

echo "🔬 Activating CMRI Exercise Virtual Environment"
echo "================================================"

# Get the directory where this script is located (portable across users/systems)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to project directory (same directory as this script)
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Check if activation was successful
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated successfully"
else
    echo "❌ ERROR: Failed to activate virtual environment"
    return 1
fi
echo "📁 Project directory: $(pwd)"
echo "🐍 Python path: $(which python)"
echo "📦 Installed packages:"
echo "   - NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "   - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "   - TorchKbNUFFT: $(python -c 'import torchkbnufft; print(torchkbnufft.__version__)')"
echo "   - SciPy: $(python -c 'import scipy; print(scipy.__version__)')"
echo "   - Matplotlib: $(python -c 'import matplotlib; print(matplotlib.__version__)')"

echo ""
echo "🚀 Ready to work on CMRI exercises!"
echo "� You should now see (venv) in your prompt"
echo "�📝 To run lab04: cd lab04 && python lab04.py"
echo "🔗 To deactivate: deactivate"
echo "================================================"