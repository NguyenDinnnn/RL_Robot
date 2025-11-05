// ---------------- Biến toàn cục ----------------
let gridSizeX = 10, gridSizeY = 8;
let robotPos = [0, 0], goal = [9, 7], obstacles = [], waypoints = [], visitedWaypoints = [];
let autoMoving = false, algoMoving = false, intervalId = null, debugMode = false, currentAlgo = null;
let startTime = null, endTime = null, trajectoryPoints = [];
let allAlgoData = {};
let currentAlgoData = { rewards: [], totalReward: 0 };
let isRunning = false; // Thêm flag để ngăn multiple runs

// ---------------- Chart ----------------
const ctx = document.getElementById('rewardChart').getContext('2d');
const rewardChart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
        responsive: true,
        plugins: { title: { display: true, text: 'Reward theo Step' } },
        scales: { x: { title: { display: true, text: 'Step' } }, y: { title: { display: true, text: 'Reward' } } }
    }
});

function randomColor(alpha = 1) {
    const r = Math.floor(Math.random() * 255), g = Math.floor(Math.random() * 255), b = Math.floor(Math.random() * 255);
    return `rgba(${r},${g},${b},${alpha})`;
}

function updateChart() {
    rewardChart.data.datasets = [];
    let maxSteps = 0;
    
    for (const algo in allAlgoData) {
        if (allAlgoData[algo].rewards.length > maxSteps) {
            maxSteps = allAlgoData[algo].rewards.length;
        }
    }
    rewardChart.data.labels = Array.from({ length: maxSteps }, (_, i) => i + 1);

    for (const algo in allAlgoData) {
        rewardChart.data.datasets.push({
            label: algo,
            data: allAlgoData[algo].rewards,
            borderColor: allAlgoData[algo].color,
            backgroundColor: allAlgoData[algo].color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
            fill: false,
            tension: 0.3,
            pointRadius: 0
        });
    }
    rewardChart.update();
}

// ---------------- Grid & Robot ----------------
function drawGridOnce() {
    let grid = document.getElementById("grid");
    grid.style.gridTemplateColumns = `repeat(${gridSizeX}, 35px)`;
    grid.innerHTML = "";
    for (let y = 0; y < gridSizeY; y++) {
        for (let x = 0; x < gridSizeX; x++) {
            let cell = document.createElement("div");
            cell.classList.add("cell");
            cell.dataset.x = x; 
            cell.dataset.y = y;
            if (obstacles.some(o => o[0] === x && o[1] === y)) cell.classList.add("obstacle");
            if (waypoints.some(w => w[0] === x && w[1] === y && !visitedWaypoints.some(vw => vw[0] === w[0] && vw[1] === w[1]))) 
                cell.classList.add("waypoint");
            if (x === goal[0] && y === goal[1]) cell.classList.add("goal");
            if (x === robotPos[0] && y === robotPos[1]) { 
                cell.classList.add("robot"); 
                cell.innerHTML = "🤖"; 
            }
            grid.appendChild(cell);
        }
    }
}

function updateRobot(prevPos, newPos) {
    let grid = document.getElementById("grid");
    let oldCell = grid.querySelector(`.cell[data-x='${prevPos[0]}'][data-y='${prevPos[1]}']`);
    if (oldCell) { 
        oldCell.classList.remove("robot"); 
        oldCell.innerHTML = ""; 
    }
    let newCell = grid.querySelector(`.cell[data-x='${newPos[0]}'][data-y='${newPos[1]}']`);
    if (newCell) { 
        newCell.classList.add("robot"); 
        newCell.innerHTML = "🤖"; 
    }
    trajectoryPoints.push([...newPos]);

    // Debug: Kiểm tra duplicate robots
    let robots = document.querySelectorAll('.robot');
    if (robots.length > 1) {
        console.warn(`Duplicate robots detected: ${robots.length} at positions: ${[...robots].map(r => `[${r.dataset.x},${r.dataset.y}]`).join(', ')}`);
    }

    if (newPos[0] === goal[0] && newPos[1] === goal[1]) {
        endTime = performance.now();
        let elapsed = ((endTime - startTime) / 1000).toFixed(2);
        document.getElementById("current-time").innerText = elapsed + " s";
        drawTrajectory();
    }

    waypoints.forEach(w => {
        if (w[0] === newPos[0] && w[1] === newPos[1]) {
            newCell.classList.remove("waypoint"); 
            newCell.classList.add("visited"); 
            newCell.innerHTML = "🤖";
            if (!visitedWaypoints.some(vw => vw[0] === w[0] && vw[1] === w[1])) 
                visitedWaypoints.push([w[0], w[1]]);
        }
    });
}

// ---------------- Manual & Debug ----------------
async function manualMove(dir) {
    if (!debugMode) return;
    let prevPos = [...robotPos];
    let resp = await fetch("/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action_name: dir })
    });
    let data = await resp.json();
    if (!data.error) {
        robotPos = data.state;
        visitedWaypoints = data.visited_waypoints || [];
        updateRobot(prevPos, robotPos);
    }
}

function toggleDebug() {
    debugMode = !debugMode;
    document.getElementById("manualControls").style.display = debugMode ? "block" : "none";
}

// ---------------- Auto Move ----------------
function move(dx, dy) {
    let newX = robotPos[0] + dx;
    let newY = robotPos[1] + dy;
    if (newX >= 0 && newX < gridSizeX && newY >= 0 && newY < gridSizeY) {
        if (!obstacles.some(o => o[0] === newX && o[1] === newY)) {
            let prevPos = [...robotPos];
            robotPos = [newX, newY];
            trajectoryPoints.push([...robotPos]);
            updateRobot(prevPos, robotPos);
        }
    }
}

function autoStepRandom() {
    const actions = [
        [0, 1],   // xuống
        [0, -1],  // lên
        [1, 0],   // phải
        [-1, 0],  // trái
    ];
    let action = actions[Math.floor(Math.random() * actions.length)];
    move(action[0], action[1]);
}

function autoStepGreedy() {
    const actions = [
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
    ];
    let bestAction = null;
    let bestDist = Infinity;

    for (let [dx, dy] of actions) {
        let newX = robotPos[0] + dx;
        let newY = robotPos[1] + dy;

        if (newX >= 0 && newX < gridSizeX && newY >= 0 && newY < gridSizeY) {
            if (!obstacles.some(o => o[0] === newX && o[1] === newY)) {
                let dist = Math.abs(goal[0] - newX) + Math.abs(goal[1] - newY);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestAction = [dx, dy];
                }
            }
        }
    }

    if (bestAction) move(bestAction[0], bestAction[1]);
}

let autoMode = "random";

function startAutoMove(mode = "random") {
    if (autoMoving || isRunning) {
        console.warn(`startAutoMove blocked: autoMoving=${autoMoving}, isRunning=${isRunning}`);
        return;
    }
    autoMoving = true; 
    algoMoving = false;
    autoMode = mode;
    startTime = performance.now();
    trajectoryPoints = [[...robotPos]];
    intervalId = setInterval(() => {
        if (autoMode === "random") autoStepRandom();
        else if (autoMode === "greedy") autoStepGreedy();
    }, 300);
    document.getElementById("msg").innerText = "Auto move (" + autoMode + ") bắt đầu";
}

function stopAll() {
    autoMoving = false; 
    algoMoving = false; 
    currentAlgo = null;
    isRunning = false; // Reset flag
    if (intervalId) clearInterval(intervalId);
    intervalId = null;
    document.getElementById("msg").innerText = "Đã dừng robot";
    document.getElementById("algoResult").style.display = "none";
}

// ---------------- Reset ----------------
async function resetGrid() {
    stopAll();
    let resp = await fetch("/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}"
    });
    let data = await resp.json();
    robotPos = data.state; 
    goal = data.map.goal; 
    obstacles = data.map.obstacles; 
    waypoints = data.map.waypoints || [];
    visitedWaypoints = []; 
    gridSizeX = data.map.width; 
    gridSizeY = data.map.height;
    drawGridOnce(); 
    updateRobot([0, 0], robotPos);
    document.getElementById("msg").innerText = "Môi trường đã được reset";
    document.getElementById("current-time").innerText = "-";
    trajectoryPoints = []; 
    clearTrajectory();
}

// ---------------- Run RL Algorithms step-by-step ----------------
async function runAlgorithm(algo) {
    if (isRunning) {
        console.warn(`Already running an algorithm, ignoring ${algo}`);
        return;
    }
    isRunning = true;
    stopAll();
    currentAlgo = algo;
    allAlgoData[algo] = { rewards: [], color: randomColor() };
    let totalReward = 0;
    let steps = 0;
    
    const resultDiv = document.getElementById("algoResult");
    resultDiv.style.display = "block";
    resultDiv.innerHTML = `<div class="loading"></div> Đang chạy thuật toán ${algo}...`;

    await resetGrid();
    
    intervalId = setInterval(async () => {
        try {
            let prevPos = [...robotPos];
            const resp = await fetch("/step_algorithm", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ algorithm: algo })
            });
            const data = await resp.json();

            if (data.error) {
                resultDiv.innerHTML = `❌ Lỗi: ${data.error}`;
                stopAll();
                isRunning = false;
                return;
            }

            robotPos = data.state;
            visitedWaypoints = data.visited_waypoints || [];
            updateRobot(prevPos, robotPos);
            
            totalReward += data.reward;
            steps = data.steps;
            allAlgoData[algo].rewards.push(totalReward);
            
            updateStats(
                algo,
                totalReward.toFixed(1),
                steps,
                `${visitedWaypoints.length}/${waypoints.length}`
            );

            if (data.done) {
                resultDiv.innerHTML = `
                    ✅ <strong>${algo}</strong><br>
                    🎯 Reward: ${totalReward.toFixed(1)}<br>
                    👣 Steps: ${steps}<br>
                    📍 Waypoints: ${visitedWaypoints.length}/${waypoints.length}<br>
                    🏆 <strong>Robot đã đến đích!</strong>
                `;
                stopAll();
                isRunning = false;
            } else {
                resultDiv.innerHTML = `<div class="loading"></div> Đang chạy thuật toán ${algo}...`;
            }

            updateChart();
        } catch (e) {
            console.error(e);
            resultDiv.innerHTML = "❌ Có lỗi xảy ra khi chạy thuật toán";
            stopAll();
            isRunning = false;
        }
    }, 200);
}

// ---------------- Run A* ----------------
async function runAStar() {
    if (isRunning) {
        console.warn("Already running an algorithm, ignoring A*");
        return;
    }
    isRunning = true;
    stopAll(); 
    currentAlgo = "A*";
    const resultDiv = document.getElementById("algoResult");
    resultDiv.style.display = "block";
    resultDiv.innerHTML = `<div class="loading"></div> Đang chạy thuật toán A*...`;

    try {
        let prevPos = [...robotPos];
        const resp = await fetch("/run_a_star", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({})
        });
        const data = await resp.json();
        if (data.error) { 
            resultDiv.innerHTML = `❌ Lỗi: ${data.error}`; 
            isRunning = false; 
            return; 
        }
        
        allAlgoData[data.algorithm] = { rewards: data.rewards_over_time, color: randomColor() };

        let path = data.path;
        for (let i = 1; i < path.length; i++) {
            await new Promise(r => setTimeout(r, 200));
            prevPos = [...robotPos];
            robotPos = path[i];
            visitedWaypoints = data.visited_waypoints || [];
            updateRobot(prevPos, robotPos);
        }

        updateStats(
            data.algorithm,
            data.reward.toFixed(1),
            data.steps,
            `${data.visited_waypoints.length}/${waypoints.length}`
        );

        resultDiv.innerHTML = `
            ✅ <strong>${data.algorithm}</strong><br>
            🎯 Reward: ${data.reward.toFixed(1)}<br>
            👣 Steps: ${data.steps}<br>
            📍 Waypoints: ${data.visited_waypoints.length}/${waypoints.length}<br>
            ${data.done ? "🏆 <strong>Robot đã đến đích!</strong>" : ""}
        `;

        updateChart();
        isRunning = false;
    } catch (e) {
        console.error(e); 
        resultDiv.innerHTML = "❌ Có lỗi xảy ra khi chạy A*";
        isRunning = false;
    }
}

// ---------------- Draw & Clear Trajectory ----------------
function drawTrajectory() {
    let canvas = document.getElementById("trajectory");
    let ctx = canvas.getContext("2d");
    clearTrajectory();
    if (trajectoryPoints.length < 2) return;
    ctx.beginPath();
    let scaleX = canvas.width / gridSizeX, scaleY = canvas.height / gridSizeY;
    ctx.moveTo(trajectoryPoints[0][0] * scaleX + scaleX / 2, trajectoryPoints[0][1] * scaleY + scaleY / 2);
    for (let i = 1; i < trajectoryPoints.length; i++) {
        ctx.lineTo(trajectoryPoints[i][0] * scaleX + scaleX / 2, trajectoryPoints[i][1] * scaleY + scaleY / 2);
    }
    ctx.strokeStyle = "red"; 
    ctx.lineWidth = 2; 
    ctx.stroke();
}

function clearTrajectory() {
    let canvas = document.getElementById("trajectory");
    let ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ---------------- Update Stats ----------------
function updateStats(algo, reward, steps, wp) {
    document.getElementById("current-algorithm").innerText = algo;
    document.getElementById("current-reward").innerText = reward;
    document.getElementById("current-steps").innerText = steps;
    document.getElementById("current-waypoints").innerText = wp;
}

// ---------------- Khởi tạo ----------------
resetGrid();
