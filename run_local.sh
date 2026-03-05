#!/bin/bash

# Logistics Document Intelligence Assistant - Local Run Script
# This script starts both the FastAPI backend and Streamlit frontend

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Logistics Document Intelligence Assistant...${NC}"

# 0. Ensure logs directory exists
mkdir -p logs

# 1. Kill any existing processes on ports 8000 and 8501
echo -e "${YELLOW}Checking for existing processes on ports 8000 and 8501...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8501 | xargs kill -9 2>/dev/null

# 2. Check for virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# 3. Ensure .env file exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo -e "${YELLOW}Creating .env from .env.example...${NC}"
        cp .env.example .env
    else
        echo -e "${RED}.env.example not found. Please create environment variables manually.${NC}"
    fi
fi

# 4. Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend:$(pwd)/backend/src
export BACKEND_URL="http://localhost:8000"

# 5. Start Backend in background
echo -e "${GREEN}Starting Backend API (FastAPI) on port 8000...${NC}"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &
BACKEND_PID=$!

# 6. Wait for backend to be ready
echo -e "${YELLOW}Waiting for backend to start...${NC}"
MAX_RETRIES=10
RETRY_COUNT=0
while ! curl -s http://localhost:8000/ping > /dev/null; do
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo -e "${RED}Backend failed to start after $MAX_RETRIES retries. Check logs/backend.log${NC}"
        kill $BACKEND_PID
        exit 1
    fi
done
echo -e "${GREEN}Backend is ready!${NC}"

# 7. Start Frontend
echo -e "${GREEN}Starting Frontend (Streamlit) on port 8501...${NC}"
cd frontend
streamlit run app.py
