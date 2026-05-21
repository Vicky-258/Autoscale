// UI Element references
const valActualRps = document.getElementById('val-actual-rps');
const valPredRps = document.getElementById('val-pred-rps');
const valState = document.getElementById('val-state');
const valReplicas = document.getElementById('val-replicas');
const trendBounds = document.getElementById('trend-bounds');
const trendBurst = document.getElementById('trend-burst');
const trendDesired = document.getElementById('trend-desired');

// Global Chart Configuration settings for aesthetics
Chart.defaults.color = '#9ca3af';
Chart.defaults.font.family = "'Outfit', sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(26, 28, 35, 0.9)';
Chart.defaults.plugins.tooltip.titleColor = '#fff';
Chart.defaults.plugins.tooltip.bodyColor = '#e5e7eb';
Chart.defaults.plugins.tooltip.borderColor = 'rgba(255, 255, 255, 0.1)';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.displayColors = true;
Chart.defaults.plugins.tooltip.boxPadding = 5;

// Data state
const maxDataPoints = 60; // Show last 60 ticks
const chartData = {
    labels: [],
    actualRps: [],
    predRps: [],
    lowerBound: [],
    upperBound: [],
    replicas: []
};

// Initialize Traffic Chart
const ctxTraffic = document.getElementById('trafficChart').getContext('2d');

// Create gradients
const actualGradient = ctxTraffic.createLinearGradient(0, 0, 0, 400);
actualGradient.addColorStop(0, 'rgba(59, 130, 246, 0.5)');
actualGradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

const boundsGradient = ctxTraffic.createLinearGradient(0, 0, 0, 400);
boundsGradient.addColorStop(0, 'rgba(245, 158, 11, 0.15)');
boundsGradient.addColorStop(1, 'rgba(245, 158, 11, 0.0)');

const trafficChart = new Chart(ctxTraffic, {
    type: 'line',
    data: {
        labels: chartData.labels,
        datasets: [
            {
                label: 'Actual RPS',
                data: chartData.actualRps,
                borderColor: '#3b82f6',
                backgroundColor: actualGradient,
                borderWidth: 2,
                pointRadius: 0,
                pointHitRadius: 10,
                tension: 0.3,
                fill: true,
                order: 2
            },
            {
                label: 'Predicted RPS (t+12)',
                data: chartData.predRps,
                borderColor: '#f59e0b',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                tension: 0.3,
                order: 3
            },
            {
                label: 'Upper Bound',
                data: chartData.upperBound,
                borderColor: 'transparent',
                backgroundColor: 'transparent',
                pointRadius: 0,
                fill: false,
                order: 1
            },
            {
                label: 'Lower Bound',
                data: chartData.lowerBound,
                borderColor: 'transparent',
                backgroundColor: boundsGradient,
                pointRadius: 0,
                fill: '-1', // Fill to Upper Bound
                order: 1
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0 // Disable internal chart animation for smoother socket streaming
        },
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                display: false // We use custom HTML legend
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                    drawBorder: false
                },
                ticks: {
                    maxTicksLimit: 10
                }
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                    drawBorder: false
                },
                beginAtZero: true,
                suggestedMax: 800
            }
        }
    }
});

// Initialize Replica Chart
const ctxReplica = document.getElementById('replicaChart').getContext('2d');
const replicaGradient = ctxReplica.createLinearGradient(0, 0, 0, 400);
replicaGradient.addColorStop(0, 'rgba(16, 185, 129, 0.4)');
replicaGradient.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

const replicaChart = new Chart(ctxReplica, {
    type: 'line',
    data: {
        labels: chartData.labels,
        datasets: [
            {
                label: 'Replicas Allocation',
                data: chartData.replicas,
                borderColor: '#10b981',
                backgroundColor: replicaGradient,
                borderWidth: 3,
                pointRadius: 0,
                stepped: true, // Step-line for replicas
                fill: true
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0
        },
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                    drawBorder: false
                },
                ticks: {
                    maxTicksLimit: 10
                }
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                    drawBorder: false
                },
                beginAtZero: true,
                suggestedMin: 0,
                suggestedMax: 20,
                ticks: {
                    stepSize: 2
                }
            }
        }
    }
});

// WebSocket Connection
let wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
let wsUrl = `${wsProtocol}//${window.location.host}/ws`;

function connectWebSocket() {
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("Connected to Real-Time Metrics Stream");
        document.querySelector('.status-indicator span').textContent = "System Live";
        document.querySelector('.status-indicator').style.color = "var(--color-success)";
        document.querySelector('.pulse-dot').style.backgroundColor = "var(--color-success)";
        document.querySelector('.pulse-dot').style.boxShadow = "0 0 8px var(--color-success)";
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'RESET') {
            // Reset arrays
            chartData.labels.length = 0;
            chartData.actualRps.length = 0;
            chartData.predRps.length = 0;
            chartData.lowerBound.length = 0;
            chartData.upperBound.length = 0;
            chartData.replicas.length = 0;
            trafficChart.update();
            replicaChart.update();
            return;
        }
        
        if (data.type === 'METRICS') {
            updateDashboard(data);
        }
    };

    ws.onclose = () => {
        console.log("WebSocket connection closed. Reconnecting in 2s...");
        document.querySelector('.status-indicator span').textContent = "Reconnecting...";
        document.querySelector('.status-indicator').style.color = "var(--color-warning)";
        document.querySelector('.pulse-dot').style.backgroundColor = "var(--color-warning)";
        document.querySelector('.pulse-dot').style.boxShadow = "0 0 8px var(--color-warning)";
        setTimeout(connectWebSocket, 2000);
    };
    
    ws.onerror = (err) => {
        console.error("WebSocket error", err);
    };
}

// Update UI
function updateDashboard(data) {
    // 1. Update text cards
    valActualRps.textContent = Math.round(data.actual_rps);
    valPredRps.textContent = Math.round(data.predicted_rps);
    
    let errBound = (data.upper_bound - data.predicted_rps).toFixed(1);
    trendBounds.textContent = `±${errBound} RPS Bounds`;
    
    // System State UI changes
    valState.textContent = data.system_state;
    valState.className = "value state-badge"; // reset
    if (data.system_state === 'NORMAL') {
        valState.classList.add('state-normal');
        document.getElementById('card-state').style.borderColor = 'rgba(255, 255, 255, 0.08)';
    } else if (data.system_state === 'UNCERTAIN') {
        valState.classList.add('state-uncertain');
        document.getElementById('card-state').style.borderColor = 'rgba(245, 158, 11, 0.4)';
    } else if (data.system_state === 'BURST') {
        valState.classList.add('state-burst');
        document.getElementById('card-state').style.borderColor = 'rgba(239, 68, 68, 0.6)';
    }
    
    // Burst indicator
    if (data.burst_state === 'BURST') {
        trendBurst.textContent = "🔥 TRAFFIC BURST DETECTED";
        trendBurst.style.color = "var(--color-danger)";
    } else if (data.burst_state === 'PERIODIC_SPIKE') {
        trendBurst.textContent = "⚠️ Minor Spike Detected";
        trendBurst.style.color = "var(--color-warning)";
    } else {
        trendBurst.textContent = "Stable Traffic";
        trendBurst.style.color = "var(--text-dim)";
    }
    
    valReplicas.textContent = data.current_replicas;
    if (data.desired_replicas !== data.current_replicas) {
        trendDesired.textContent = `Scaling to ${data.desired_replicas}...`;
        trendDesired.style.color = "var(--color-primary)";
    } else {
        trendDesired.textContent = `Target: ${data.desired_replicas} (Stable)`;
        trendDesired.style.color = "var(--text-dim)";
    }

    // 2. Update Charts
    const timeLabel = `Tick ${data.tick}`;
    
    chartData.labels.push(timeLabel);
    chartData.actualRps.push(data.actual_rps);
    chartData.predRps.push(data.predicted_rps);
    chartData.lowerBound.push(data.lower_bound);
    chartData.upperBound.push(data.upper_bound);
    chartData.replicas.push(data.current_replicas);

    // Keep sliding window
    if (chartData.labels.length > maxDataPoints) {
        chartData.labels.shift();
        chartData.actualRps.shift();
        chartData.predRps.shift();
        chartData.lowerBound.shift();
        chartData.upperBound.shift();
        chartData.replicas.shift();
    }

    // Tell chartjs to update smoothly
    trafficChart.update();
    replicaChart.update();
}

// Start app
connectWebSocket();
