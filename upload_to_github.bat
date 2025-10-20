@echo off
REM ============================================================================
REM Quick Upload to GitHub - Windows
REM ============================================================================

echo.
echo ============================================================================
echo  Upload Multi-Agent Research Platform to GitHub
echo ============================================================================
echo.

cd /d "%~dp0"

echo Step 1: Checking git installation...
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed
    echo Please install Git from: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo ✓ Git is installed
echo.

echo Step 2: Initializing git repository...
git init
echo.

echo Step 3: Adding files...
git add .
echo.

echo Step 4: Checking what will be uploaded...
echo.
echo Files to be uploaded:
echo ============================================================================
git status
echo ============================================================================
echo.

echo IMPORTANT: Make sure .env is NOT listed above!
echo.
pause

echo Step 5: Committing changes...
git commit -m "Initial commit: Multi-Agent Research Intelligence Platform"
echo.

echo Step 6: Connecting to GitHub...
echo.
echo Please enter your GitHub repository URL:
echo Example: https://github.com/username/repo-name.git
echo.
set /p REPO_URL="Repository URL: "

if "%REPO_URL%"=="" (
    echo ERROR: Repository URL cannot be empty
    pause
    exit /b 1
)

git remote add origin %REPO_URL%
echo.

echo Step 7: Setting main branch...
git branch -M main
echo.

echo Step 8: Pushing to GitHub...
echo.
echo You may be prompted for credentials:
echo - Username: Your GitHub username
echo - Password: Use Personal Access Token (not your actual password)
echo.
echo Get token from: https://github.com/settings/tokens
echo.
pause

git push -u origin main

if errorlevel 1 (
    echo.
    echo ============================================================================
    echo  Upload Failed!
    echo ============================================================================
    echo.
    echo Common issues:
    echo 1. Wrong repository URL
    echo 2. Authentication failed - use Personal Access Token
    echo 3. Repository already has files - see GITHUB_UPLOAD_INSTRUCTIONS.md
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo  Success! Your project is now on GitHub! 🎉
echo ============================================================================
echo.
echo Next steps:
echo 1. Go to your GitHub repository in browser
echo 2. Add description and topics
echo 3. Deploy to Streamlit Cloud (see DEPLOYMENT_GUIDE.md)
echo.
echo Repository URL: %REPO_URL%
echo.
pause

