const canvas = document.getElementById('gameCanvas');
const ctx = canvas ? canvas.getContext('2d') : null;
const scoreEl = document.getElementById('score');
const finalScoreEl = document.getElementById('final-score');
const startScreen = document.getElementById('start-screen');
const gameOverScreen = document.getElementById('game-over-screen');
const startBtn = document.getElementById('start-btn');
const restartBtn = document.getElementById('restart-btn');

// AI Control Elements
const aiModeToggle = document.getElementById('ai-toggle');
const modeLabel = document.getElementById('mode-label');
const aiTypeSelect = document.getElementById('ai-type-select');
const aiTypeSelector = document.getElementById('ai-type-selector');
const rlActions = document.getElementById('rl-actions');
const giantActions = document.getElementById('giant-actions');
const sftActions = document.getElementById('sft-actions');
const sftPostActions = document.getElementById('sft-post-actions');
const loadModelBtn = document.getElementById('load-model-btn');
const trainBtn = document.getElementById('train-btn');
const trainingStatus = document.getElementById('training-status');
const giantStatsBtn = document.getElementById('giant-stats-btn');
const sftLoadModelBtn = document.getElementById('sft-load-model-btn');
const sftStatsBtn = document.getElementById('sft-stats-btn');
const sftPostLoadModelBtn = document.getElementById('sft-post-load-model-btn');
const sftPostStatsBtn = document.getElementById('sft-post-stats-btn');

// Game Constants
const WORLD_SIZE = 4000;
const INITIAL_PLAYER_RADIUS = 20;
const FOOD_COUNT = 500;
const ENEMY_COUNT = 30;
const COLORS = ['#ff4757', '#2ed573', '#1e90ff', '#ffa502', '#a4b0be', '#5352ed', '#ff6b81'];
const MIN_SPLIT_RADIUS = 30;
const MERGE_TIME = 15000; // 15 seconds
const SPLIT_BOOST = 20;
const MAX_BLOBS = 16;
const RL_SERVER_URL = 'http://localhost:5001';
const GIANT_SERVER_URL = 'http://localhost:5002';
const SFT_SERVER_URL = 'http://localhost:5003';
const SFT_POST_SERVER_URL = 'http://localhost:5004';

// AI Type State
let aiType = 'giant'; // 'rl', 'giant', 'sft', or 'sft_post'

// Game State
let gameRunning = false;
let score = 0;
let animationId;
let aiMode = false;
let aiActionInterval = null;

// Camera
let camera = { x: 0, y: 0 };

// Entities
let players = []; // Array of player blobs
let foods = [];
let enemies = [];
let mouse = { x: 0, y: 0 };
let aiTarget = { x: WORLD_SIZE / 2, y: WORLD_SIZE / 2 }; // AI controlled target

// Utility Functions
function randomRange(min, max) {
    return Math.random() * (max - min) + min;
}

function randomColor() {
    return COLORS[Math.floor(Math.random() * COLORS.length)];
}

function getDistance(x1, y1, x2, y2) {
    return Math.hypot(x2 - x1, y2 - y1);
}

// AI Functions
function extractGameState() {
    // Extract team state for coordinated multi-blob control
    const obs = new Array(49).fill(0);  // Now 49 dimensions

    if (players.length === 0) return obs;

    // Calculate team centroid (center of all blobs)
    let totalX = 0, totalY = 0, totalMass = 0;
    for (const blob of players) {
        const mass = blob.radius * blob.radius;  // Area as proxy for mass
        totalX += blob.x * mass;
        totalY += blob.y * mass;
        totalMass += mass;
    }
    const centroidX = totalX / totalMass;
    const centroidY = totalY / totalMass;
    const avgRadius = Math.sqrt(totalMass / players.length);

    // Team state (normalized)
    obs[0] = centroidX / WORLD_SIZE;
    obs[1] = centroidY / WORLD_SIZE;
    obs[2] = avgRadius / 100.0;
    obs[3] = players.length / 16.0;  // Blob count (normalized)

    // Find nearest 5 food items (relative to centroid)
    const foodDistances = foods.map(food => ({
        dist: getDistance(centroidX, centroidY, food.x, food.y),
        food: food
    })).sort((a, b) => a.dist - b.dist);

    for (let i = 0; i < Math.min(5, foodDistances.length); i++) {
        const idx = 4 + i * 4;  // Start at 4 now
        const { dist, food } = foodDistances[i];

        // Calculate relative angle to food from centroid
        const dx = food.x - centroidX;
        const dy = food.y - centroidY;
        const angle = Math.atan2(dy, dx);

        obs[idx] = Math.cos(angle);
        obs[idx + 1] = Math.sin(angle);
        obs[idx + 2] = food.radius / 10.0;
        obs[idx + 3] = dist / WORLD_SIZE;
    }

    // Find nearest 5 enemies (relative to centroid)
    const enemyDistances = enemies.map(enemy => ({
        dist: getDistance(centroidX, centroidY, enemy.x, enemy.y),
        enemy: enemy
    })).sort((a, b) => a.dist - b.dist);

    for (let i = 0; i < Math.min(5, enemyDistances.length); i++) {
        const idx = 24 + i * 5;  // Start at 24 now
        const { dist, enemy } = enemyDistances[i];

        // Calculate relative angle to enemy from centroid
        const dx = enemy.x - centroidX;
        const dy = enemy.y - centroidY;
        const angle = Math.atan2(dy, dx);

        obs[idx] = Math.cos(angle);
        obs[idx + 1] = Math.sin(angle);
        obs[idx + 2] = enemy.radius / 100.0;
        obs[idx + 3] = dist / WORLD_SIZE;
        obs[idx + 4] = avgRadius / enemy.radius;  // Relative size
    }

    return obs;
}

async function getAIAction() {
    // Get action from AI agent (RL, Giant, or SFT)
    try {
        if (players.length === 0) return null;

        if (aiType === 'rl') {
            // RL Agent - uses observation vector
            const observation = extractGameState();
            const response = await fetch(`${RL_SERVER_URL}/api/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ observation: observation })
            });

            if (!response.ok) {
                console.error('Failed to get action from RL model');
                return null;
            }

            const data = await response.json();
            return data;
        } else if (aiType === 'giant') {
            // Giant Agent - uses game state directly
            const gameState = extractGiantGameState();
            const response = await fetch(`${GIANT_SERVER_URL}/api/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(gameState)
            });

            if (!response.ok) {
                console.error('Failed to get action from Giant Agent');
                return null;
            }

            const data = await response.json();
            return data;
        } else if (aiType === 'sft') {
            // SFT Agent - uses observation vector (same as RL)
            const observation = extractGameState();

            // Debug: 检查观察向量
            if (!observation || observation.length !== 49) {
                console.error('Invalid observation vector:', observation);
                return null;
            }

            const response = await fetch(`${SFT_SERVER_URL}/api/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ observation: observation })
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error('Failed to get action from SFT model:', errorData);
                console.error('Observation vector length:', observation.length);
                console.error('First 5 values:', observation.slice(0, 5));
                return null;
            }

            const data = await response.json();
            return data;
        } else if (aiType === 'sft_post') {
            // SFT Post-Training Agent - uses observation vector (PPO model)
            const observation = extractGameState();

            if (!observation || observation.length !== 49) {
                console.error('Invalid observation vector:', observation);
                return null;
            }

            const response = await fetch(`${SFT_POST_SERVER_URL}/api/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ observation: observation })
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error('Failed to get action from SFT Post-Training model:', errorData);
                return null;
            }

            const data = await response.json();
            return data;
        }
    } catch (error) {
        console.error('Error getting AI action:', error);
        return null;
    }
}

function extractGiantGameState() {
    // Extract game state for Giant Agent
    if (players.length === 0) return null;

    // Calculate team centroid
    let totalX = 0, totalY = 0;
    for (const blob of players) {
        totalX += blob.x;
        totalY += blob.y;
    }
    const avgX = totalX / players.length;
    const avgY = totalY / players.length;

    // Calculate average radius
    let totalArea = 0;
    for (const blob of players) {
        totalArea += Math.PI * blob.radius * blob.radius;
    }
    const avgRadius = Math.sqrt(totalArea / (Math.PI * players.length));

    // Format foods
    const formattedFoods = foods.map(f => ({
        x: f.x,
        y: f.y,
        radius: f.radius
    }));

    // Format enemies with velocity estimation
    const formattedEnemies = enemies.map(e => ({
        x: e.x,
        y: e.y,
        radius: e.radius,
        dx: e.dx || 0,
        dy: e.dy || 0
    }));

    return {
        player_x: avgX,
        player_y: avgY,
        player_radius: avgRadius,
        foods: formattedFoods,
        enemies: formattedEnemies
    };
}

function applyAIAction(actionData) {
    // Apply AI action to game with smoothing
    if (!actionData || players.length === 0) return;

    const player = players[0];
    const targetAngle = actionData.angle * Math.PI / 180;

    // Smooth angle transitions to prevent jitter
    // Calculate smallest angle difference
    let angleDiff = targetAngle - prevAIAngle;
    while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
    while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

    // Blend with previous angle
    const smoothedAngle = prevAIAngle + angleDiff * ANGLE_SMOOTHING;
    prevAIAngle = smoothedAngle;

    // Set AI target based on smoothed angle
    const distance = 500;
    aiTarget.x = player.x + Math.cos(smoothedAngle) * distance;
    aiTarget.y = player.y + Math.sin(smoothedAngle) * distance;

    // Handle split action
    if (actionData.split && player.radius >= MIN_SPLIT_RADIUS) {
        splitBlobs();
    }
}

async function aiControlLoop() {
    // Main AI control loop with smoothing
    if (!aiMode || !gameRunning) return;

    const actionData = await getAIAction();
    if (actionData) {
        applyAIAction(actionData);
    }
}

// Track previous AI target for smoothing
let prevAIAngle = 0;
const ANGLE_SMOOTHING = 0.3; // Blend 30% new, 70% old

async function loadModel() {
    // Load trained model from server
    try {
        const response = await fetch(`${RL_SERVER_URL}/api/model/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: 'models/rl_agent/agario_ppo' })
        });

        const data = await response.json();
        if (response.ok) {
            alert('Model loaded successfully!');
        } else {
            alert(`Failed to load model: ${data.error}`);
        }
    } catch (error) {
        alert(`Error loading model: ${error.message}\nMake sure RL server is running on port 5000`);
    }
}

async function startTraining() {
    // Start training on server
    try {
        trainingStatus.classList.remove('hidden');

        // Show training modal
        showTrainingModal();

        const response = await fetch(`${RL_SERVER_URL}/api/train/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ timesteps: 10000 })
        });

        const data = await response.json();
        if (response.ok) {
            // Poll for training status and update visualization
            const statusInterval = setInterval(async () => {
                const statusResp = await fetch(`${RL_SERVER_URL}/api/train/status`);
                const statusData = await statusResp.json();

                updateTrainingMetrics(statusData);

                if (!statusData.active) {
                    clearInterval(statusInterval);
                    trainingStatus.classList.add('hidden');
                    document.getElementById('training-state').textContent = 'Completed!';
                    setTimeout(() => {
                        alert('Training completed! You can now load the model.');
                    }, 1000);
                }
            }, 2000); // Update every 2 seconds
        } else {
            trainingStatus.classList.add('hidden');
            hideTrainingModal();
            alert(`Failed to start training: ${data.error}`);
        }
    } catch (error) {
        trainingStatus.classList.add('hidden');
        hideTrainingModal();
        alert(`Error starting training: ${error.message}\nMake sure RL server is running on port 5001`);
    }
}

// Training Visualization
let rewardChart = null;
const trainingModal = document.getElementById('training-modal');
const closeModalBtn = document.getElementById('close-modal-btn');

function showTrainingModal() {
    trainingModal.classList.remove('hidden');
    initRewardChart();
}

function hideTrainingModal() {
    trainingModal.classList.add('hidden');
}

function initRewardChart() {
    const ctx = document.getElementById('reward-chart').getContext('2d');

    if (rewardChart) {
        rewardChart.destroy();
    }

    rewardChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Episode Reward',
                data: [],
                borderColor: '#ff4757',
                backgroundColor: 'rgba(255, 71, 87, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Training Reward Over Time'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Reward'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Episode'
                    }
                }
            }
        }
    });
}

function updateTrainingMetrics(statusData) {
    // Update metric cards
    document.getElementById('episodes-count').textContent = statusData.num_episodes || 0;
    document.getElementById('avg-reward').textContent = (statusData.mean_reward || 0).toFixed(2);
    document.getElementById('training-state').textContent = statusData.active ? 'Training...' : 'Completed';

    // Update chart
    if (rewardChart && statusData.recent_rewards && statusData.recent_rewards.length > 0) {
        const rewards = statusData.recent_rewards;
        const startEpisode = Math.max(0, statusData.num_episodes - rewards.length);

        rewardChart.data.labels = rewards.map((_, i) => startEpisode + i + 1);
        rewardChart.data.datasets[0].data = rewards;
        rewardChart.update('none'); // Update without animation for performance
    }
}


// Initialization
function init() {
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('mousemove', (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    });
    window.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && gameRunning && !aiMode) {
            splitBlobs();
        }
    });

    if (startBtn) startBtn.addEventListener('click', startGame);
    if (restartBtn) restartBtn.addEventListener('click', startGame);

    // AI Control Listeners
    if (aiModeToggle) {
        aiModeToggle.addEventListener('change', (e) => {
            aiMode = e.target.checked;
            if (modeLabel) modeLabel.textContent = aiMode ? 'AI Mode' : 'Manual Mode';

            // Update UI to show/hide AI type selector
            updateAIUI();

            if (aiMode) {
                // Start AI control loop - high frequency for fast reaction
                if (aiActionInterval) clearInterval(aiActionInterval);
                aiActionInterval = setInterval(aiControlLoop, 100); // 🚀 10 actions per second for fast reaction
            } else {
                // Stop AI control loop
                if (aiActionInterval) {
                    clearInterval(aiActionInterval);
                    aiActionInterval = null;
                }
            }
        });
    } else {
        console.error('AI Mode toggle not found!');
    }

    if (loadModelBtn) loadModelBtn.addEventListener('click', loadModel);
    if (trainBtn) trainBtn.addEventListener('click', startTraining);

    if (closeModalBtn) closeModalBtn.addEventListener('click', hideTrainingModal);

    // AI Type Selector
    if (aiTypeSelect) {
        aiTypeSelect.addEventListener('change', (e) => {
            aiType = e.target.value;
            updateAIUI();
        });
        // Initialize UI
        updateAIUI();
    }

    // Giant Agent Stats Button
    if (giantStatsBtn) {
        giantStatsBtn.addEventListener('click', showGiantStats);
    }

    const closeGiantModalBtn = document.getElementById('close-giant-modal-btn');
    if (closeGiantModalBtn) {
        closeGiantModalBtn.addEventListener('click', hideGiantStatsModal);
    }

    // SFT Agent Buttons
    if (sftLoadModelBtn) {
        sftLoadModelBtn.addEventListener('click', loadSFTModel);
    }

    if (sftStatsBtn) {
        sftStatsBtn.addEventListener('click', showSFTStats);
    }

    const closeSftModalBtn = document.getElementById('close-sft-modal-btn');
    if (closeSftModalBtn) {
        closeSftModalBtn.addEventListener('click', hideSFTStatsModal);
    }

    // SFT Post-Training Agent Buttons
    if (sftPostLoadModelBtn) {
        sftPostLoadModelBtn.addEventListener('click', loadSFTPostModel);
    }

    if (sftPostStatsBtn) {
        sftPostStatsBtn.addEventListener('click', showSFTPostStats);
    }

    const closeSftPostModalBtn = document.getElementById('close-sft-post-modal-btn');
    if (closeSftPostModalBtn) {
        closeSftPostModalBtn.addEventListener('click', hideSFTPostStatsModal);
    }
}

function updateAIUI() {
    // Show/hide appropriate action panels based on AI type
    if (aiType === 'rl') {
        if (rlActions) rlActions.classList.remove('hidden');
        if (giantActions) giantActions.classList.add('hidden');
        if (sftActions) sftActions.classList.add('hidden');
        if (sftPostActions) sftPostActions.classList.add('hidden');
    } else if (aiType === 'giant') {
        if (rlActions) rlActions.classList.add('hidden');
        if (giantActions) giantActions.classList.remove('hidden');
        if (sftActions) sftActions.classList.add('hidden');
        if (sftPostActions) sftPostActions.classList.add('hidden');
    } else if (aiType === 'sft') {
        if (rlActions) rlActions.classList.add('hidden');
        if (giantActions) giantActions.classList.add('hidden');
        if (sftActions) sftActions.classList.remove('hidden');
        if (sftPostActions) sftPostActions.classList.add('hidden');
    } else if (aiType === 'sft_post') {
        if (rlActions) rlActions.classList.add('hidden');
        if (giantActions) giantActions.classList.add('hidden');
        if (sftActions) sftActions.classList.add('hidden');
        if (sftPostActions) sftPostActions.classList.remove('hidden');
    }

    // Show AI type selector only when AI mode is enabled
    if (aiTypeSelector) {
        aiTypeSelector.style.display = aiMode ? 'block' : 'none';
    }
}

function resizeCanvas() {
    if (!canvas) {
        console.error('Canvas element not found!');
        return;
    }
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

function spawnFood() {
    foods = [];
    for (let i = 0; i < FOOD_COUNT; i++) {
        foods.push({
            x: randomRange(0, WORLD_SIZE),
            y: randomRange(0, WORLD_SIZE),
            radius: randomRange(3, 6),
            color: randomColor()
        });
    }
}

function spawnEnemies() {
    enemies = [];
    const playerStartX = WORLD_SIZE / 2;
    const playerStartY = WORLD_SIZE / 2;
    const minSafeDistance = 600; // Minimum distance from player start position

    for (let i = 0; i < ENEMY_COUNT; i++) {
        let radius = randomRange(15, 50);
        let x, y, distance;

        // Keep trying until we find a position far enough from player start
        do {
            x = randomRange(0, WORLD_SIZE);
            y = randomRange(0, WORLD_SIZE);
            distance = getDistance(x, y, playerStartX, playerStartY);
        } while (distance < minSafeDistance);

        enemies.push({
            x: x,
            y: y,
            radius: radius,
            color: randomColor(),
            dx: randomRange(-2, 2),
            dy: randomRange(-2, 2),
            speed: 30 / radius // Larger enemies move slower
        });
    }
}

function startGame() {
    gameRunning = true;
    score = 0;
    scoreEl.textContent = score;
    startScreen.classList.add('hidden');
    gameOverScreen.classList.add('hidden');

    players = [{
        x: WORLD_SIZE / 2,
        y: WORLD_SIZE / 2,
        radius: INITIAL_PLAYER_RADIUS,
        color: '#ff4757',
        dx: 0,
        dy: 0,
        recombineTime: 0
    }];

    spawnFood();
    spawnEnemies();
    animate();
}

function splitBlobs() {
    let newBlobs = [];
    let currentCount = players.length;

    players.forEach(blob => {
        if (currentCount >= MAX_BLOBS) return;
        if (blob.radius < MIN_SPLIT_RADIUS) return;

        const newArea = (Math.PI * blob.radius * blob.radius) / 2;
        const newRadius = Math.sqrt(newArea / Math.PI);

        blob.radius = newRadius;
        blob.recombineTime = Date.now() + MERGE_TIME;

        // Calculate split direction
        const targetX = aiMode ? aiTarget.x : (mouse.x + camera.x);
        const targetY = aiMode ? aiTarget.y : (mouse.y + camera.y);
        const angle = Math.atan2(targetY - blob.y, targetX - blob.x);

        newBlobs.push({
            x: blob.x + Math.cos(angle) * blob.radius * 2,
            y: blob.y + Math.sin(angle) * blob.radius * 2,
            radius: newRadius,
            color: blob.color,
            dx: Math.cos(angle) * SPLIT_BOOST,
            dy: Math.sin(angle) * SPLIT_BOOST,
            recombineTime: Date.now() + MERGE_TIME
        });
        currentCount++;
    });

    players = players.concat(newBlobs);
}

function gameOver() {
    gameRunning = false;
    cancelAnimationFrame(animationId);
    finalScoreEl.textContent = score;
    gameOverScreen.classList.remove('hidden');
}

// Game Loop
function update() {
    if (!gameRunning) return;

    // Calculate Camera Center
    let totalX = 0, totalY = 0;
    players.forEach(p => {
        totalX += p.x;
        totalY += p.y;
    });

    if (players.length === 0) {
        gameOver();
        return;
    }

    const centerX = totalX / players.length;
    const centerY = totalY / players.length;

    camera.x += (centerX - canvas.width / 2 - camera.x) * 0.1;
    camera.y += (centerY - canvas.height / 2 - camera.y) * 0.1;

    // Update Player Blobs
    const targetX = aiMode ? aiTarget.x : (mouse.x + camera.x);
    const targetY = aiMode ? aiTarget.y : (mouse.y + camera.y);

    for (let i = 0; i < players.length; i++) {
        let blob = players[i];

        const angle = Math.atan2(targetY - blob.y, targetX - blob.x);
        const baseSpeed = 30 / Math.pow(blob.radius, 0.4);

        blob.x += Math.cos(angle) * baseSpeed;
        blob.y += Math.sin(angle) * baseSpeed;

        blob.x += blob.dx;
        blob.y += blob.dy;
        blob.dx *= 0.9;
        blob.dy *= 0.9;

        blob.x = Math.max(blob.radius, Math.min(WORLD_SIZE - blob.radius, blob.x));
        blob.y = Math.max(blob.radius, Math.min(WORLD_SIZE - blob.radius, blob.y));
    }

    // Blob Physics
    for (let i = 0; i < players.length; i++) {
        for (let j = i + 1; j < players.length; j++) {
            let b1 = players[i];
            let b2 = players[j];

            const dist = getDistance(b1.x, b1.y, b2.x, b2.y);
            const minDist = b1.radius + b2.radius;

            if (dist < minDist) {
                if (b1.recombineTime < Date.now() && b2.recombineTime < Date.now()) {
                    const area = Math.PI * b1.radius * b1.radius + Math.PI * b2.radius * b2.radius;
                    b1.radius = Math.sqrt(area / Math.PI);
                    b2.toBeRemoved = true;
                } else {
                    const angle = Math.atan2(b2.y - b1.y, b2.x - b1.x);
                    const overlap = minDist - dist;
                    const force = overlap / 50;

                    b1.x -= Math.cos(angle) * force;
                    b1.y -= Math.sin(angle) * force;
                    b2.x += Math.cos(angle) * force;
                    b2.y += Math.sin(angle) * force;
                }
            }
        }
    }

    players = players.filter(p => !p.toBeRemoved);

    // Update Enemies
    enemies.forEach(enemy => {
        enemy.x += enemy.dx;
        enemy.y += enemy.dy;

        if (enemy.x - enemy.radius < 0 || enemy.x + enemy.radius > WORLD_SIZE) enemy.dx *= -1;
        if (enemy.y - enemy.radius < 0 || enemy.y + enemy.radius > WORLD_SIZE) enemy.dy *= -1;

        players.forEach(blob => {
            const dist = getDistance(blob.x, blob.y, enemy.x, enemy.y);
            if (dist < blob.radius + enemy.radius) {
                if (blob.radius > enemy.radius * 1.1) {
                    const area = Math.PI * blob.radius * blob.radius + Math.PI * enemy.radius * enemy.radius;
                    blob.radius = Math.sqrt(area / Math.PI);
                    score += Math.floor(enemy.radius);
                    scoreEl.textContent = score;
                    enemy.x = randomRange(0, WORLD_SIZE);
                    enemy.y = randomRange(0, WORLD_SIZE);
                    enemy.radius = randomRange(15, 50);
                } else if (enemy.radius > blob.radius * 1.1) {
                    blob.toBeRemoved = true;
                }
            }
        });
    });

    players = players.filter(p => !p.toBeRemoved);

    // Eat Food
    for (let i = foods.length - 1; i >= 0; i--) {
        const food = foods[i];
        let eaten = false;

        for (let j = 0; j < players.length; j++) {
            const blob = players[j];
            const dist = getDistance(blob.x, blob.y, food.x, food.y);
            if (dist < blob.radius + food.radius) {
                const area = Math.PI * blob.radius * blob.radius + Math.PI * food.radius * food.radius;
                blob.radius = Math.sqrt(area / Math.PI);
                score += 1;
                scoreEl.textContent = score;
                eaten = true;
                break;
            }
        }

        if (eaten) {
            foods.splice(i, 1);
            foods.push({
                x: randomRange(0, WORLD_SIZE),
                y: randomRange(0, WORLD_SIZE),
                radius: randomRange(3, 6),
                color: randomColor()
            });
        }
    }
}

function draw() {
    if (!ctx || !canvas) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    ctx.translate(-camera.x, -camera.y);

    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 5;
    ctx.strokeRect(0, 0, WORLD_SIZE, WORLD_SIZE);

    foods.forEach(food => {
        ctx.beginPath();
        ctx.arc(food.x, food.y, food.radius, 0, Math.PI * 2);
        ctx.fillStyle = food.color;
        ctx.fill();
        ctx.closePath();
    });

    enemies.forEach(enemy => {
        ctx.beginPath();
        ctx.arc(enemy.x, enemy.y, enemy.radius, 0, Math.PI * 2);
        ctx.fillStyle = enemy.color;
        ctx.fill();
        ctx.strokeStyle = 'rgba(0,0,0,0.1)';
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.closePath();
    });

    players.forEach(blob => {
        ctx.beginPath();
        ctx.arc(blob.x, blob.y, blob.radius, 0, Math.PI * 2);
        ctx.fillStyle = blob.color;
        ctx.fill();
        ctx.strokeStyle = aiMode ? '#2ed573' : 'white';
        ctx.lineWidth = 4;
        ctx.stroke();
        ctx.closePath();

        ctx.fillStyle = 'white';
        ctx.font = `bold ${Math.max(12, blob.radius / 2)}px Outfit`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(aiMode ? 'AI' : 'You', blob.x, blob.y);
    });

    ctx.restore();
}

function animate() {
    if (!gameRunning) return;
    animationId = requestAnimationFrame(animate);
    update();
    draw();
}

// Giant Agent Statistics Functions
async function showGiantStats() {
    try {
        const response = await fetch(`${GIANT_SERVER_URL}/api/stats`);
        if (!response.ok) {
            alert('Failed to fetch Giant Agent stats. Make sure the server is running on port 5002.');
            return;
        }

        const stats = await response.json();

        // Update metrics
        document.getElementById('giant-total-decisions').textContent = stats.total_decisions;
        document.getElementById('giant-split-rate').textContent = (stats.split_rate * 100).toFixed(1) + '%';
        document.getElementById('giant-max-radius').textContent = stats.max_radius_seen.toFixed(0);

        // Update phase distribution bars
        const total = stats.total_decisions;
        const phases = stats.phase_distribution;

        if (total > 0) {
            const earlyPct = (phases.early / total * 100);
            const midPct = (phases.mid / total * 100);
            const latePct = (phases.late / total * 100);

            document.getElementById('bar-early').style.width = earlyPct + '%';
            document.getElementById('bar-mid').style.width = midPct + '%';
            document.getElementById('bar-late').style.width = latePct + '%';

            document.getElementById('count-early').textContent = phases.early;
            document.getElementById('count-mid').textContent = phases.mid;
            document.getElementById('count-late').textContent = phases.late;
        } else {
            document.getElementById('bar-early').style.width = '0%';
            document.getElementById('bar-mid').style.width = '0%';
            document.getElementById('bar-late').style.width = '0%';

            document.getElementById('count-early').textContent = '0';
            document.getElementById('count-mid').textContent = '0';
            document.getElementById('count-late').textContent = '0';
        }

        // Show modal
        document.getElementById('giant-stats-modal').classList.remove('hidden');
    } catch (error) {
        alert(`Error fetching stats: ${error.message}\nMake sure Giant Agent server is running on port 5002`);
    }
}

function hideGiantStatsModal() {
    document.getElementById('giant-stats-modal').classList.add('hidden');
}

// SFT Agent Functions
async function loadSFTModel() {
    try {
        const response = await fetch(`${SFT_SERVER_URL}/api/model/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: 'models/sft_giant/best_model.pt' })
        });

        const data = await response.json();
        if (response.ok) {
            alert(`SFT Model loaded successfully!\n\nPath: ${data.path}\nDescription: ${data.model_info.description}`);
        } else {
            alert(`Failed to load SFT model: ${data.error}`);
        }
    } catch (error) {
        alert(`Error loading SFT model: ${error.message}\nMake sure SFT server is running on port 5003`);
    }
}

async function showSFTStats() {
    try {
        const response = await fetch(`${SFT_SERVER_URL}/api/stats`);
        if (!response.ok) {
            alert('Failed to fetch SFT Agent stats. Make sure the server is running on port 5003.');
            return;
        }

        const stats = await response.json();

        // Update metrics
        document.getElementById('sft-total-decisions').textContent = stats.total_decisions;
        document.getElementById('sft-split-rate').textContent = (stats.split_rate * 100).toFixed(1) + '%';
        document.getElementById('sft-model-status').textContent = stats.model_path ? 'Loaded' : 'Not Loaded';

        // Update action distribution bars (show top 5 actions)
        const actionBars = document.getElementById('sft-action-bars');
        actionBars.innerHTML = '';

        if (stats.total_decisions > 0) {
            // Get action labels
            const actionLabels = [
                'Move 0° (Right)', 'Move 45° (Down-Right)', 'Move 90° (Down)', 'Move 135° (Down-Left)',
                'Move 180° (Left)', 'Move 225° (Up-Left)', 'Move 270° (Up)', 'Move 315° (Up-Right)',
                'Split 0° (Right)', 'Split 45° (Down-Right)', 'Split 90° (Down)', 'Split 135° (Down-Left)',
                'Split 180° (Left)', 'Split 225° (Up-Left)', 'Split 270° (Up)', 'Split 315° (Up-Right)'
            ];

            // Get top 5 actions by percentage
            const actionData = stats.action_distribution.percentages.map((pct, idx) => ({
                action: idx,
                label: actionLabels[idx],
                percentage: pct,
                count: stats.action_distribution.counts[idx]
            })).sort((a, b) => b.percentage - a.percentage).slice(0, 5);

            // Create bars
            actionData.forEach(item => {
                const barDiv = document.createElement('div');
                barDiv.className = 'phase-bar';
                barDiv.innerHTML = `
                    <span class="phase-label">${item.label}</span>
                    <div class="bar-container">
                        <div class="bar bar-${item.action >= 8 ? 'late' : (item.action >= 4 ? 'mid' : 'early')}"
                             style="width: ${item.percentage}%"></div>
                    </div>
                    <span class="phase-count">${item.count}</span>
                `;
                actionBars.appendChild(barDiv);
            });
        } else {
            actionBars.innerHTML = '<p style="text-align: center; color: #999;">No data yet</p>';
        }

        // Show modal
        document.getElementById('sft-stats-modal').classList.remove('hidden');
    } catch (error) {
        alert(`Error fetching SFT stats: ${error.message}\nMake sure SFT Agent server is running on port 5003`);
    }
}

function hideSFTStatsModal() {
    document.getElementById('sft-stats-modal').classList.add('hidden');
}

// SFT Post-Training Agent Functions
async function loadSFTPostModel() {
    try {
        const response = await fetch(`${SFT_POST_SERVER_URL}/api/model/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: 'models/sft_post_training/best_model' })
        });

        const data = await response.json();
        if (response.ok) {
            alert(`SFT Post-Training Model loaded successfully!\n\nPath: ${data.path}\nType: ${data.model_info.type}\nDescription: ${data.model_info.description}`);
        } else {
            alert(`Failed to load SFT Post-Training model: ${data.error}\n\nHint: ${data.hint || 'Make sure the server is running on port 5004'}`);
        }
    } catch (error) {
        alert(`Error loading SFT Post-Training model: ${error.message}\nMake sure SFT Post-Training server is running on port 5004`);
    }
}

async function showSFTPostStats() {
    try {
        const response = await fetch(`${SFT_POST_SERVER_URL}/api/stats`);
        if (!response.ok) {
            alert('Failed to fetch SFT Post-Training stats. Make sure the server is running on port 5004.');
            return;
        }

        const stats = await response.json();

        // Update metrics
        document.getElementById('sft-post-total-decisions').textContent = stats.total_decisions;
        document.getElementById('sft-post-split-rate').textContent = (stats.split_rate * 100).toFixed(1) + '%';
        document.getElementById('sft-post-model-status').textContent = stats.model_path ? 'Loaded' : 'Not Loaded';
        document.getElementById('sft-post-model-type').textContent = stats.model_type || 'PPO';

        // Update action distribution bars (show top 5 actions)
        const actionBars = document.getElementById('sft-post-action-bars');
        actionBars.innerHTML = '';

        if (stats.total_decisions > 0) {
            // Get action labels
            const actionLabels = [
                'Move 0° (Right)', 'Move 45° (Down-Right)', 'Move 90° (Down)', 'Move 135° (Down-Left)',
                'Move 180° (Left)', 'Move 225° (Up-Left)', 'Move 270° (Up)', 'Move 315° (Up-Right)',
                'Split 0° (Right)', 'Split 45° (Down-Right)', 'Split 90° (Down)', 'Split 135° (Down-Left)',
                'Split 180° (Left)', 'Split 225° (Up-Left)', 'Split 270° (Up)', 'Split 315° (Up-Right)'
            ];

            // Get top 5 actions by percentage
            const actionData = stats.action_distribution.percentages.map((pct, idx) => ({
                action: idx,
                label: actionLabels[idx],
                percentage: pct,
                count: stats.action_distribution.counts[idx]
            })).sort((a, b) => b.percentage - a.percentage).slice(0, 5);

            // Create bars
            actionData.forEach(item => {
                const barDiv = document.createElement('div');
                barDiv.className = 'phase-bar';
                barDiv.innerHTML = `
                    <span class="phase-label">${item.label}</span>
                    <div class="bar-container">
                        <div class="bar bar-${item.action >= 8 ? 'late' : (item.action >= 4 ? 'mid' : 'early')}"
                             style="width: ${item.percentage}%"></div>
                    </div>
                    <span class="phase-count">${item.count}</span>
                `;
                actionBars.appendChild(barDiv);
            });
        } else {
            actionBars.innerHTML = '<p style="text-align: center; color: #999;">No data yet</p>';
        }

        // Show modal
        document.getElementById('sft-post-stats-modal').classList.remove('hidden');
    } catch (error) {
        alert(`Error fetching SFT Post-Training stats: ${error.message}\nMake sure server is running on port 5004`);
    }
}

function hideSFTPostStatsModal() {
    document.getElementById('sft-post-stats-modal').classList.add('hidden');
}

init();
