#!/bin/bash

# Script to start all AI servers for the Agar.io game
# Usage: ./start_all_servers.sh

echo "========================================================================"
echo "🚀 Starting All AI Servers for Agar.io Game"
echo "========================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Run from project root (one level up from scripts/)
cd "$SCRIPT_DIR/.."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "📂 Working directory: $SCRIPT_DIR"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create it first with: python3 -m venv venv"
    exit 1
fi

# Function to start a server
start_server() {
    local name=$1
    local script=$2
    local port=$3

    echo -e "${YELLOW}Starting $name on port $port...${NC}"

    # Check if port is already in use
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "  ⚠️  Port $port is already in use. Skipping..."
    else
        nohup venv/bin/python $script > logs/${name}.log 2>&1 &
        local pid=$!
        echo $pid > logs/${name}.pid
        sleep 2

        if ps -p $pid > /dev/null 2>&1; then
            echo -e "  ${GREEN}✅ $name started (PID: $pid)${NC}"
        else
            echo -e "  ❌ Failed to start $name. Check logs/${name}.log"
        fi
    fi
    echo ""
}

# Create logs directory
mkdir -p logs

echo "========================================================================"
echo "Starting AI Servers..."
echo "========================================================================"
echo ""

# Start each server
# Start each server
start_server "RL-Server" "backend/rl_server.py" 5001
start_server "Giant-Server" "backend/giant_server.py" 5002
start_server "SFT-Server" "backend/sft_server.py" 5003
start_server "SFT-Post-Server" "backend/sft_post_server.py" 5004

echo "========================================================================"
echo "✅ All servers started!"
echo "========================================================================"
echo ""
echo "Server URLs:"
echo "  🤖 RL Agent:            http://localhost:5001"
echo "  🧮 Giant Agent:         http://localhost:5002"
echo "  🧠 SFT Agent:           http://localhost:5003"
echo "  🚀 SFT Post-Training:   http://localhost:5004"
echo ""
echo "Logs:"
echo "  Check logs/ directory for server logs"
echo ""
echo "To stop all servers:"
echo "  ./stop_all_servers.sh"
echo ""
echo "Now open index.html in your browser to play!"
echo "========================================================================"
