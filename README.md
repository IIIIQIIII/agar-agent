# Agar.io AI Agent

This project is a comprehensive AI development system for Agar.io, featuring Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT) pipelines. It includes a game environment, training infrastructure, backend inference servers, and a web-based frontend for visualization.

## Project Structure

- **`agario_core/`**: Core game logic, environment definitions, and agent implementations.
- **`backend/`**: Python servers (`giant_server.py`, `rl_server.py`) for serving AI models to the frontend.
- **`frontend/`**: A web-based interface (HTML/JS/CSS) to visualize the game and agent performance.
- **`training/`**: Training pipelines for RL and SFT, including data processing and model optimization.
- **`models/`**: Directory for storing trained model checkpoints.
- **`scripts/`**: Utility shell scripts for starting servers, running demos, and managing training jobs.
- **`tests/`**: Unit tests for ensuring system stability.

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js (optional, for advanced frontend dev)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/IIIIQIIII/agar-agent.git
    cd agar-agent
    ```

2.  **Set up the Python environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

### Usage

**Quick Demo:**
Run the quick demo script to see the agent in action:
```bash
./scripts/quick_demo.sh
```

**Start All Servers:**
To start the backend servers for full interaction:
```bash
./scripts/start_all_servers.sh
```

Then open `frontend/index.html` in your browser.

## License

[MIT](LICENSE)
