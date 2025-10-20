#!/bin/bash

# ============================================================================
# AIRFLOW SETUP SCRIPT
# Automated setup for Research Intelligence Platform with Airflow
# ============================================================================

echo "=================================="
echo "🚀 Airflow Setup Script"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
echo "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed!${NC}"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker found${NC}"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed!${NC}"
    echo "Please install Docker Compose from: https://docs.docker.com/compose/install/"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose found${NC}"

echo ""
echo "=================================="
echo "📝 Environment Configuration"
echo "=================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating...${NC}"
    
    # Prompt for API keys
    read -p "Enter your GROQ_API_KEY: " groq_key
    read -p "Enter your SERPAPI_KEY (optional, press Enter to skip): " serp_key
    
    # Create .env file
    cat > .env << EOF
# API Keys
GROQ_API_KEY=${groq_key}
SERPAPI_KEY=${serp_key}

# Airflow Configuration
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
AIRFLOW_UID=50000
EOF
    
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

echo ""
echo "=================================="
echo "📂 Creating Directories"
echo "=================================="

# Create necessary directories
mkdir -p logs plugins chroma_db
echo -e "${GREEN}✓ Created logs/ plugins/ chroma_db/${NC}"

echo ""
echo "=================================="
echo "🔧 Initializing Airflow Database"
echo "=================================="

# Initialize Airflow
echo "This may take a few minutes..."
docker-compose -f docker-compose-airflow.yml up airflow-init

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Airflow initialized successfully${NC}"
else
    echo -e "${RED}❌ Airflow initialization failed${NC}"
    exit 1
fi

echo ""
echo "=================================="
echo "🚀 Starting Airflow Services"
echo "=================================="

# Start Airflow services
docker-compose -f docker-compose-airflow.yml up -d

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Airflow services started${NC}"
else
    echo -e "${RED}❌ Failed to start Airflow${NC}"
    exit 1
fi

echo ""
echo "=================================="
echo "⏳ Waiting for services to be ready..."
echo "=================================="

sleep 10

# Check if webserver is running
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Airflow webserver is ready!${NC}"
        break
    fi
    
    attempt=$((attempt + 1))
    echo "Waiting... ($attempt/$max_attempts)"
    sleep 5
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${YELLOW}⚠️  Webserver took longer than expected. Check logs:${NC}"
    echo "docker-compose -f docker-compose-airflow.yml logs airflow-webserver"
fi

echo ""
echo "=================================="
echo "✅ SETUP COMPLETE!"
echo "=================================="
echo ""
echo "📊 Access Airflow UI:"
echo "   URL: http://localhost:8080"
echo "   Username: airflow"
echo "   Password: airflow"
echo ""
echo "🎯 Next Steps:"
echo "   1. Open http://localhost:8080 in your browser"
echo "   2. Login with credentials above"
echo "   3. Go to Admin → Variables"
echo "   4. Set 'research_query' to your research topic"
echo "   5. Enable and trigger the DAG 'research_intelligence_pipeline'"
echo ""
echo "📚 For more info, read AIRFLOW_README.md"
echo ""
echo "🛠️  Useful Commands:"
echo "   View logs:    docker-compose -f docker-compose-airflow.yml logs -f"
echo "   Stop Airflow: docker-compose -f docker-compose-airflow.yml down"
echo "   Restart:      docker-compose -f docker-compose-airflow.yml restart"
echo ""
echo "=================================="

