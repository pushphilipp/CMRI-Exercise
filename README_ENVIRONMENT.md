# CMRI Exercise Environment

This project contains exercises for Computational Magnetic Resonance Imaging (CMRI) course.

## Environment Setup

A Python virtual environment has been configured with all necessary dependencies.

### Quick Start

1. **Activate the environment (IMPORTANT: use `source`, not direct execution):**
   ```bash
   # From the project directory:
   source activate_env.sh
   
   # Or from anywhere, using full path:
   source /path/to/CMRI-Exercise/activate_env.sh
   
   # Alternative short form:
   . activate_env.sh
   ```
   
   **Note:** You should see `(venv)` appear in your command prompt after successful activation.

2. **Run lab exercises:**
   ```bash
   cd lab04
   python lab04.py
   ```

3. **Deactivate when done:**
   ```bash
   deactivate
   ```

### Installed Packages

- **NumPy** (2.0.2) - Numerical computing
- **PyTorch** (2.8.0+cu128) - Deep learning framework with CUDA support
- **TorchKbNUFFT** (1.4.0) - k-space Non-Uniform Fast Fourier Transform
- **SciPy** (1.13.1) - Scientific computing
- **Matplotlib** (3.9.4) - Plotting and visualization

### Manual Environment Management

If you prefer manual control:

```bash
# Activate
source venv/bin/activate

# Install additional packages (if needed)
pip install package_name

# Deactivate
deactivate
```

### Dependencies

All dependencies are listed in `requirements.txt`. To recreate the environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
CMRI-Exercise/
├── lab04/
│   ├── lab04.py          # Main lab exercise file
│   ├── utils.py          # Utility functions
│   ├── grid.py           # Gridding operations
│   └── radial_data.mat   # Data file
├── venv/                 # Virtual environment
├── requirements.txt      # Python dependencies
└── activate_env.sh       # Environment activation script
```

## CUDA Support

PyTorch is installed with CUDA support. CUDA availability: ✅ True

This enables GPU acceleration for tensor operations when a compatible GPU is available.