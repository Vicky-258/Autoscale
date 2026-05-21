import os
import sys
import json
import time
import asyncio
import threading
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

# Auto-add path so imports work without PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import run_autoscaler
from sim.traffic import TrafficGenerator
from sim.environment import MockKubernetes
from config.logger import setup_logger

logger = setup_logger("autoscale.demo")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Web socket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to ws: {e}")

manager = ConnectionManager()

# Ensure scaling state is clean before we start
if os.path.exists("scaling_state.json"):
    os.remove("scaling_state.json")

class StopSimulation(Exception):
    pass

class RealTimeDemoClock:
    def __init__(self, max_ticks, loop_interval_s=0.5):
        self.ticks = 0
        self.max_ticks = max_ticks
        self.loop_interval_s = loop_interval_s
        self.current_time = datetime.now(timezone.utc)
        
    def sleep_and_tick(self, _ignored):
        """Sleeps in real-time but tick advances by the logical loop interval"""
        time.sleep(self.loop_interval_s)
        self.ticks += 1
        # To make it look like a real system, we advance time realistically or just by seconds
        self.current_time += timedelta(seconds=30) # A logical 30s step from settings
        if self.ticks >= self.max_ticks:
            logger.info("Simulation loop resetting for continuous demo...")
            self.ticks = 0
            # Reset traffic generator internal state
            global traffic_gen, k8s_mock
            traffic_gen = TrafficGenerator()
            k8s_mock = MockKubernetes(initial_replicas=3)
            # Remove state file
            if os.path.exists("scaling_state.json"):
                os.remove("scaling_state.json")
            # Clear frontend via a special message
            try:
                loop.call_soon_threadsafe(metrics_queue.put_nowait, {"type": "RESET"})
            except NameError:
                pass

traffic_gen = TrafficGenerator()
k8s_mock = MockKubernetes(initial_replicas=3)
clock = RealTimeDemoClock(max_ticks=200, loop_interval_s=0.5)

metrics_queue = asyncio.Queue()

def sync_metrics_callback(state_dict):
    state_dict['tick'] = clock.ticks
    state_dict['time'] = clock.current_time.isoformat()
    state_dict['type'] = 'METRICS'
    try:
        loop.call_soon_threadsafe(metrics_queue.put_nowait, state_dict)
    except NameError:
        pass # Loop not set yet

def run_simulation_thread():
    logger.info("Starting demo runner thread...")
    while True:
        try:
            run_autoscaler(
                get_rps_fn=traffic_gen.get_next,
                get_tensor_fn=traffic_gen.get_tensor,
                get_replicas_fn=k8s_mock.get_replicas,
                set_replicas_fn=k8s_mock.set_replicas,
                sleep_fn=clock.sleep_and_tick,
                get_time_fn=lambda: clock.current_time,
                metrics_callback=sync_metrics_callback
            )
        except Exception as e:
            logger.error(f"Runner thread crashed: {e}. Restarting...")
            time.sleep(2)

@app.on_event("startup")
async def startup_event():
    global loop
    loop = asyncio.get_running_loop()
    
    # Start background consumer
    async def consume_metrics():
        while True:
            metric = await metrics_queue.get()
            await manager.broadcast(metric)
            
    asyncio.create_task(consume_metrics())
    
    # Start synchronous autoscaler in a thread
    thread = threading.Thread(target=run_simulation_thread, daemon=True)
    thread.start()

# Add routes after definition
os.makedirs("web", exist_ok=True)
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("web/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    # run with reload=False because we spawn threads and maintain state
    uvicorn.run("demo:app", host="0.0.0.0", port=8000, reload=False)
