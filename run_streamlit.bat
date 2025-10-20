@echo off
REM ============================================================================
REM Launch Streamlit UI for Research Intelligence Platform
REM ============================================================================

echo.
echo ============================================================================
echo  Multi-Agent Research Intelligence Platform
echo  Starting Streamlit Web UI...
echo ============================================================================
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Checking for .env file...
if not exist ".env" (
    echo WARNING: .env file not found!
    echo.
    echo Please create a .env file with your API keys:
    echo   1. Rename env.example to .env
    echo   2. Add your GROQ_API_KEY
    echo   3. Optionally add SERPAPI_KEY
    echo.
    echo Or set environment variables:
    echo   set GROQ_API_KEY=your_key_here
    echo.
    pause
)

echo.
echo Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Streamlit not installed
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting Streamlit UI...
echo.
echo The app will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ============================================================================
echo.

streamlit run 9_Deployment\app.py

pause

