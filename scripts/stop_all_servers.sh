#!/bin/bash

# Script to stop all AI servers
# Usage: ./stop_all_servers.sh

echo "========================================================================"
echo "🛑 Stopping All AI Servers"
echo "========================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Run from project root (one level up from scripts/)
cd "$SCRIPT_DIR/.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Function to stop a server
stop_server() {
    local name=$1
    local pid_file="logs/${name}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "Stopping $name (PID: $pid)..."
            kill $pid
            sleep 1

            if ps -p $pid > /dev/null 2>&1; then
                echo -e "  ${RED}Force killing $name...${NC}"
                kill -9 $pid
            fi

            echo -e "  ${GREEN}✅ $name stopped${NC}"
        else
            echo -e "  ⚠️  $name not running"
        fi
        rm -f "$pid_file"
    else
        echo -e "  ⚠️  No PID file for $name"
    fi
}

# Stop each server
stop_server "RL-Server"
stop_server "Giant-Server"
stop_server "SFT-Server"
stop_server "SFT-Post-Server"

# Also kill any remaining Python processes on these ports
echo ""
echo "Checking for any remaining processes on ports 5001-5004..."
for port in 5001 5002 5003 5004; do
    pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        echo -e "${RED}Killing process on port $port (PID: $pid)${NC}"
        kill -9 $pid 2>/dev/null
    fi
done

echo ""
echo "========================================================================"
echo "✅ All servers stopped!"
echo "========================================================================"
