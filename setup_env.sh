#!/usr/bin/env bash
# setup_env.sh
# ============
# Create and configure the Python virtual environment for fedlab_zmq.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# After setup, activate the venv with:
#   source venv/bin/activate
#
# Notes:
#   - Requires Python 3.10+ (tested with 3.12.4)
#   - PyTorch is installed as CPU-only by default.
#     For CUDA support, edit the TORCH_INSTALL_CMD variable below.
#   - The script is safe to re-run; it will skip venv creation if
#     the venv already exists.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

VENV_DIR="venv"
PYTHON="${PYTHON:-python3}"
REQUIREMENTS="requirements.txt"

# PyTorch install command.
# Default: CPU-only (works on all platforms, including macOS Apple Silicon via MPS).
# For CUDA 12.1: change to:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
TORCH_INSTALL_CMD="pip install torch torchvision"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }

# ─────────────────────────────────────────────────────────────────────────────
# 1. Check Python version
# ─────────────────────────────────────────────────────────────────────────────

info "Checking Python version..."
PYTHON_VERSION=$("$PYTHON" --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 10 ]; then
    error "Python 3.10+ required. Found: $PYTHON_VERSION"
    error "Install a newer Python or set PYTHON=/path/to/python3.12"
    exit 1
fi
success "Python $PYTHON_VERSION found."

# ─────────────────────────────────────────────────────────────────────────────
# 2. Create virtual environment
# ─────────────────────────────────────────────────────────────────────────────

if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment '$VENV_DIR' already exists. Skipping creation."
    warn "To recreate from scratch: rm -rf $VENV_DIR && ./setup_env.sh"
else
    info "Creating virtual environment in '$VENV_DIR'..."
    "$PYTHON" -m venv "$VENV_DIR"
    success "Virtual environment created."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Activate venv
# ─────────────────────────────────────────────────────────────────────────────

info "Activating virtual environment..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
success "Virtual environment activated."

# ─────────────────────────────────────────────────────────────────────────────
# 4. Upgrade pip
# ─────────────────────────────────────────────────────────────────────────────

info "Upgrading pip..."
pip install --upgrade pip --quiet
success "pip upgraded."

# ─────────────────────────────────────────────────────────────────────────────
# 5. Install PyTorch first (separate to support CUDA index URL)
# ─────────────────────────────────────────────────────────────────────────────

info "Installing PyTorch and torchvision..."
info "Command: $TORCH_INSTALL_CMD"
eval "$TORCH_INSTALL_CMD"
success "PyTorch installed."

# ─────────────────────────────────────────────────────────────────────────────
# 6. Install remaining requirements
# ─────────────────────────────────────────────────────────────────────────────

if [ ! -f "$REQUIREMENTS" ]; then
    error "requirements.txt not found in $(pwd)"
    error "Run this script from the fedlab_zmq project root."
    exit 1
fi

info "Installing requirements from $REQUIREMENTS..."
# Install everything except torch/torchvision (already installed above)
pip install -r "$REQUIREMENTS" --quiet
success "All requirements installed."

# ─────────────────────────────────────────────────────────────────────────────
# 7. Verification: import test
# ─────────────────────────────────────────────────────────────────────────────

info "Running import verification..."

python3 - <<'PYEOF'
import sys

checks = [
    ("zmq",         "pyzmq"),
    ("msgpack",     "msgpack"),
    ("torch",       "torch"),
    ("torchvision", "torchvision"),
    ("numpy",       "numpy"),
    ("pandas",      "pandas"),
    ("yaml",        "PyYAML"),
    ("streamlit",   "streamlit"),
    ("plotly",      "plotly"),
    ("tqdm",        "tqdm"),
    ("matplotlib",  "matplotlib"),
]

ok = True
for module, package in checks:
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "?")
        print(f"  [OK]  {package:<20} {ver}")
    except ImportError as e:
        print(f"  [FAIL] {package:<20} NOT FOUND — {e}", file=sys.stderr)
        ok = False

# Quick fedlab_zmq module check
try:
    sys.path.insert(0, ".")
    from hardware.profiles import DEVICE_PROFILES
    from hardware.energy_model import comm_energy_j
    rpi4 = DEVICE_PROFILES["raspberry_pi_4"]
    e = comm_energy_j(1e6, rpi4, "uplink")
    print(f"  [OK]  fedlab_zmq.hardware   Shannon 1MB uplink = {e*1000:.2f} mJ")
except Exception as exc:
    print(f"  [FAIL] fedlab_zmq.hardware   {exc}", file=sys.stderr)
    ok = False

if ok:
    print("\nAll imports successful. Environment is ready.")
else:
    print("\nSome imports failed. Check the errors above.", file=sys.stderr)
    sys.exit(1)
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
# 8. Done
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "  Setup complete!"
echo ""
echo "  To activate the environment in a new shell:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  To run the benchmark:"
echo "    python run_experiment.py --benchmark --rounds 100"
echo ""
echo "  To launch the dashboard:"
echo "    streamlit run dashboard/app.py"
echo "========================================================"
