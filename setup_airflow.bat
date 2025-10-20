@echo off
REM ============================================================================
REM AIRFLOW SETUP SCRIPT FOR WINDOWS
REM Automated setup for Research Intelligence Platform with Airflow
REM ============================================================================

echo ==================================
echo 🚀 Airflow Setup Script (Windows)
echo ==================================
echo.

REM Check if Docker is installed
echo Checking Docker installation...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed!
    echo Please install Docker Desktop from: https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)
echo ✓ Docker found
echo.

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed!
    pause
    exit /b 1
)
echo ✓ Docker Compose found
echo.

echo ==================================
echo 📝 Environment Configuration
echo ==================================
echo.

REM Check if .env exists
if not exist .env (
    echo ⚠️  .env file not found. Creating...
    
    set /p groq_key="Enter your GROQ_API_KEY: "
    set /p serp_key="Enter your SERPAPI_KEY (optional, press Enter to skip): "
    
    REM Create .env file
    (
        echo # API Keys
        echo GROQ_API_KEY=%groq_key%
        echo SERPAPI_KEY=%serp_key%
        echo.
        echo # Airflow Configuration
        echo _AIRFLOW_WWW_USER_USERNAME=airflow
        echo _AIRFLOW_WWW_USER_PASSWORD=airflow
        echo AIRFLOW_UID=50000
    ) > .env
    
    echo ✓ .env file created
) else (
    echo ✓ .env file exists
)
echo.

echo ==================================
echo 📂 Creating Directories
echo ==================================
if not exist logs mkdir logs
if not exist plugins mkdir plugins
if not exist chroma_db mkdir chroma_db
echo ✓ Created logs/ plugins/ chroma_db/
echo.

echo ==================================
echo 🔧 Initializing Airflow Database
echo ==================================
echo This may take a few minutes...
docker-compose -f docker-compose-airflow.yml up airflow-init

if %errorlevel% neq 0 (
    echo ❌ Airflow initialization failed
    pause
    exit /b 1
)
echo ✓ Airflow initialized successfully
echo.

echo ==================================
echo 🚀 Starting Airflow Services
echo ==================================
docker-compose -f docker-compose-airflow.yml up -d

if %errorlevel% neq 0 (
    echo ❌ Failed to start Airflow
    pause
    exit /b 1
)
echo ✓ Airflow services started
echo.

echo ==================================
echo ⏳ Waiting for services to be ready...
echo ==================================
timeout /t 15 /nobreak >nul
echo.

echo ==================================
echo ✅ SETUP COMPLETE!
echo ==================================
echo.
echo 📊 Access Airflow UI:
echo    URL: http://localhost:8080
echo    Username: airflow
echo    Password: airflow
echo.
echo 🎯 Next Steps:
echo    1. Open http://localhost:8080 in your browser
echo    2. Login with credentials above
echo    3. Go to Admin → Variables
echo    4. Set 'research_query' to your research topic
echo    5. Enable and trigger the DAG 'research_intelligence_pipeline'
echo.
echo 📚 For more info, read AIRFLOW_README.md
echo.
echo 🛠️  Useful Commands:
echo    View logs:    docker-compose -f docker-compose-airflow.yml logs -f
echo    Stop Airflow: docker-compose -f docker-compose-airflow.yml down
echo    Restart:      docker-compose -f docker-compose-airflow.yml restart
echo.
echo ==================================
pause

