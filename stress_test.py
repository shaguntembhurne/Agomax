from pymavlink import mavutil
import pandas as pd
import time
from pathlib import Path

from agomax.pipeline import Pipeline

# ============================================================
# CONFIG
# ============================================================
NORMAL_SECONDS = 30        # collect normal data first
HZ = 10                   # telemetry rate
MODEL_DIR = "models"

# ============================================================
# CONNECT TO ARDUPILOT
# ============================================================
master = mavutil.mavlink_connection("udp:127.0.0.1:14550")
master.wait_heartbeat()
print("[OK] Connected to ArduPilot")

# ============================================================
# STATE CACHE
# ============================================================
state = {
    "altitude": None,
    "velocity_z": None,
    "roll": None,
    "pitch": None,
    "yaw": None,
    "battery": None,
}

def full_row():
    return all(v is not None for v in state.values())

# ============================================================
# STEP 1: COLLECT NORMAL DATA
# ============================================================
print(f"[STEP] Recording NORMAL data for {NORMAL_SECONDS}s")

rows = []
start = time.time()

while time.time() - start < NORMAL_SECONDS:
    msg = master.recv_match(blocking=True)
    if msg is None:
        continue

    t = msg.get_type()

    if t == "ATTITUDE":
        state["roll"] = msg.roll
        state["pitch"] = msg.pitch
        state["yaw"] = msg.yaw

    elif t == "VFR_HUD":
        state["altitude"] = msg.alt
        state["velocity_z"] = msg.climb

    elif t == "SYS_STATUS":
        state["battery"] = msg.voltage_battery / 1000.0

        if full_row():
            rows.append(state.copy())
            time.sleep(1 / HZ)

train_df = pd.DataFrame(rows)
print(f"[OK] Collected {len(train_df)} normal samples")

# ============================================================
# STEP 2: TRAIN AGOMAX
# ============================================================
pipeline = Pipeline(
    model_dir=MODEL_DIR,
    window=40,
    k=3.0,
    vote_threshold=0.5,
    confirm_steps=3,
    cooldown=10,
)

print("[STEP] Training Agomax")
pipeline.fit(train_df)
print("[OK] Training complete")

# ============================================================
# STEP 3: REAL-TIME DETECTION
# ============================================================
print("[STEP] Live anomaly detection started")

while True:
    msg = master.recv_match(blocking=True)
    if msg is None:
        continue

    t = msg.get_type()

    if t == "ATTITUDE":
        state["roll"] = msg.roll
        state["pitch"] = msg.pitch
        state["yaw"] = msg.yaw

    elif t == "VFR_HUD":
        state["altitude"] = msg.alt
        state["velocity_z"] = msg.climb

    elif t == "SYS_STATUS":
        state["battery"] = msg.voltage_battery / 1000.0

        if not full_row():
            continue

        df = pd.DataFrame([state.copy()])
        confidence, anomaly, event = pipeline.predict(df)

        if event[0] == 1:
            print("🚨 ANOMALY EVENT")
            print(state)
            print(f"confidence={confidence[0]:.2f}")
        else:
            print(f"ok | conf={confidence[0]:.2f}")

        time.sleep(1 / HZ)
