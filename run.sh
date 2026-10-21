#!/usr/bin/env bash

# ==============================================================================
# YOLO-MP: AI Face Recognition Attendance System
# Single-command setup, cleaner, database initializer, and runner
# ==============================================================================

set -e

# Colors for terminal styling
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}"
echo "=========================================================="
echo "          ⚡ YOLO-MP: AI ATTENDANCE SYSTEM ⚡             "
echo "=========================================================="
echo -e "${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. CLEAN UP OBSOLETE / BROKEN FILES
echo -e "${YELLOW}[1/5] Cleaning up legacy and redundant files...${NC}"
rm -rf attendancedb encodings/a.py recognize.py templates/index.html
mkdir -p encodings dataset static/uploads static/css

# 2. CHECK PYTHON ENVIRONMENT & VENV
echo -e "${YELLOW}[2/5] Checking Python environment...${NC}"
PYTHON_CMD="python3"

if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}❌ Error: Python 3 is not installed or not in PATH.${NC}"
    exit 1
fi

VENV_DIR="$SCRIPT_DIR/.venv"

# If user has an existing system or ai environment, detect it or create local venv
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${CYAN}Creating virtual environment at .venv...${NC}"
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Reuse local GPU/PyTorch packages if available to avoid 2GB re-download
if [ -d "/home/sk1029/ai-gpu-env/lib/python3.14/site-packages" ]; then
    echo "/home/sk1029/ai-gpu-env/lib/python3.14/site-packages" > "$VENV_DIR/lib64/python3.14/site-packages/ai_gpu.pth" 2>/dev/null || true
fi

# Prevent PyPI IPv6 network timeouts by forcing IPv4
cat << 'EOF' > "$VENV_DIR/lib64/python3.14/site-packages/sitecustomize.py"
import socket
_orig = socket.getaddrinfo
def _ipv4(*args, **kwargs):
    return _orig(args[0], args[1], socket.AF_INET, *(args[3:] if len(args) > 3 else ()), **kwargs)
socket.getaddrinfo = _ipv4
EOF

echo -e "${GREEN}✅ Virtual environment active: $(which python)${NC}"

# 3. VERIFY & INSTALL DEPENDENCIES
echo -e "${YELLOW}[3/5] Verifying installed packages...${NC}"
if ! python -c "import flask, cv2, pandas" &> /dev/null; then
    echo -e "${CYAN}Installing dependencies from requirements.txt...${NC}"
    pip install -r requirements.txt
else
    echo -e "${GREEN}✅ Core dependencies already satisfied.${NC}"
fi

# 4. INITIALIZE DATABASE & DIRECTORIES
echo -e "${YELLOW}[4/5] Initializing database and face biometric storage...${NC}"
python -c "
import database
database.init_db()
print('✅ Database tables initialized.')
"

# 5. EXECUTION DISPATCHER
echo -e "${GREEN}${BOLD}"
echo "=========================================================="
echo "  🚀 SYSTEM READY! "
echo "=========================================================="
echo -e "${NC}"

MODE="${1:-web}"

if [ "$MODE" == "camera" ] || [ "$MODE" == "--camera" ]; then
    echo -e "${CYAN}Starting Standalone OpenCV Desktop Camera (Press 'q' in window to exit)...${NC}"
    python camera.py
elif [ "$MODE" == "train" ] || [ "$MODE" == "--train" ]; then
    echo -e "${CYAN}Running face training on 'dataset/' folder...${NC}"
    python train_faces.py
else
    PORT="${PORT:-5000}"
    echo -e "${BOLD}Portal Credentials:${NC}"
    echo -e "  👤 Admin:   Username: ${CYAN}admin${NC}   | Password: ${CYAN}admin123${NC}"
    echo -e "  👤 Teacher: Username: ${CYAN}teacher${NC} | Password: ${CYAN}teacher123${NC}"
    echo ""
    echo -e "🌐 Web Dashboard running at: ${GREEN}${BOLD}http://127.0.0.1:${PORT}${NC}"
    echo -e "Press Ctrl+C to terminate the server.\n"
    
    python app.py
fi
