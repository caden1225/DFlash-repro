#!/bin/bash
# ============================================================================
# DFlash Mac Benchmark - One-Click Runner
# ============================================================================
# Usage:
#   chmod +x scripts/run_mac_benchmark.sh
#   ./scripts/run_mac_benchmark.sh
#
# This script will:
#   1. Check environment (mlx, mlx-lm)
#   2. Run quick benchmark with official DFlash model
#   3. Run full benchmark if quick test passes
# ============================================================================

set -e

echo "========================================================================"
echo "  DFlash Mac Benchmark Runner"
echo "  Hardware: MacBook (48GB Unified Memory)"
echo "========================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python3 found: $(python3 --version)"

# Check dependencies
echo ""
echo "Checking dependencies..."

python3 -c "import mlx" 2>/dev/null && echo -e "${GREEN}✓${NC} mlx installed" || {
    echo -e "${YELLOW}⚠ mlx not installed${NC}"
    echo "  Install: pip install mlx mlx-lm"
    exit 1
}

python3 -c "import mlx_lm" 2>/dev/null && echo -e "${GREEN}✓${NC} mlx-lm installed" || {
    echo -e "${YELLOW}⚠ mlx-lm not installed${NC}"
    echo "  Install: pip install mlx-lm"
    exit 1
}

python3 -c "import transformers" 2>/dev/null && echo -e "${GREEN}✓${NC} transformers installed" || {
    echo -e "${YELLOW}⚠ transformers not installed${NC}"
    echo "  Install: pip install transformers"
    exit 1
}

python3 -c "import datasets" 2>/dev/null && echo -e "${GREEN}✓${NC} datasets installed" || {
    echo -e "${YELLOW}⚠ datasets not installed${NC}"
    echo "  Install: pip install datasets"
    exit 1
}

# Check MPS availability
echo ""
echo "Checking Metal (MPS)..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print(f'  ${GREEN}✓${NC} MPS available')
else:
    print(f'  ${YELLOW}⚠ MPS not available, will use CPU${NC}')
"

echo ""
echo "========================================================================"
echo "  Step 1: Quick Test (5 samples, official model)"
echo "========================================================================"
cd "$(dirname "$0")/.."
python3 -m dflash_reproduce.mac_benchmark --quick

echo ""
echo "========================================================================"
echo "  Step 2: Full Benchmark (50 samples)"
echo "========================================================================"
read -p "Run full benchmark? This may take 10-20 minutes [y/N]: " confirm
if [[ $confirm == [yY] ]]; then
    python3 -m dflash_reproduce.mac_benchmark --official --num-samples 50 --output-dir ./mac_eval_results
else
    echo "Skipped full benchmark."
fi

echo ""
echo "========================================================================"
echo "  Benchmark Complete!"
echo "  Results saved to: ./mac_eval_results/"
echo "========================================================================"
