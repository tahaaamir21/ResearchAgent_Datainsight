#!/bin/bash
# ============================================================================
# Launch Streamlit UI for Research Intelligence Platform
# ============================================================================

echo ""
echo "============================================================================"
echo " Multi-Agent Research Intelligence Platform"
echo " Starting Streamlit Web UI..."
echo "============================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.11+ from https://python.org"
    exit 1
fi

echo "Python version:"
python3 --version

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: .env file not found!"
    echo ""
    echo "Please create a .env file with your API keys:"
    echo "  1. Rename env.example to .env: mv env.example .env"
    echo "  2. Add your GROQ_API_KEY"
    echo "  3. Optionally add SERPAPI_KEY"
    echo ""
    echo "Or set environment variables:"
    echo "  export GROQ_API_KEY=your_key_here"
    echo ""
    read -p "Press Enter to continue anyway..."
fi

# Check dependencies
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo ""
    echo "Streamlit not installed. Installing dependencies..."
    pip3 install -r requirements.txt
fi

echo ""
echo "Starting Streamlit UI..."
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================================================"
echo ""

streamlit run 9_Deployment/app.py

