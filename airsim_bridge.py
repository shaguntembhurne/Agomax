import airsim
import time
import pandas as pd
import numpy as np
import os
import math
import argparse
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.drone_anomaly.pipeline import DronePipeline
from src.drone_anomaly.ensemble import AgomaxEnsemble

# --- HELPER: Quaternion to Euler ---
def to_euler(q_w, q_x, q_y, q_z):
    sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (q_w * q_y - q_z * q_x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll * (180/math.pi), pitch * (180/math.pi), yaw * (180/math.pi)

class AirSimBridge:
    def __init__(self):
        print("Connecting to AirSim...")
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            
            # --- THE FIX IS HERE ---
            # We explicitly tell AirSim: "The Python script is just watching. 
            # Let the HUMAN control the drone with the Keyboard."
            self.client.enableApiControl(False) 
            
            print("\033[92m[SUCCESS] Connected! Manual Control Enabled.\033[0m")
            print("-------------------------------------------------------")
            print("HOW TO FLY:")
            print("1. Click inside the 'Blocks' game window.")
            print("2. Press 'F' (or PageUp) to Takeoff.")
            print("3. Use Arrow Keys to move, A/D to rotate.")
            print("-------------------------------------------------------")
            
        except Exception as e:
            print(f"\033[91m[ERROR] Could not connect to AirSim: {e}\033[0m")
            sys.exit(1)
        
        self.battery = 100.0
        self.start_time = time.time()
        self.tick_count = 0

    def get_data_packet(self):
        state = self.client.getMultirotorState()
        kinematics = state.kinematics_estimated
        
        vx = kinematics.linear_velocity.x_val
        vy = kinematics.linear_velocity.y_val
        vz = -kinematics.linear_velocity.z_val 
        
        alt = -kinematics.position.z_val
        if alt < 0: alt = 0
        
        q = kinematics.orientation
        roll, pitch, yaw = to_euler(q.w_val, q.x_val, q.y_val, q.z_val)

        # Physics Simulation for missing sensors
        load = 10.0 + (abs(vz) * 5.0) + (math.sqrt(vx**2 + vy**2) * 0.5)
        motor_rpm = 5000 + (load * 100) + np.random.normal(0, 20)
        
        voltage_sag = load * 0.01
        self.battery -= (load * 0.0001)
        true_voltage = self.battery - voltage_sag

        self.tick_count += 1
        
        return {
            "timestamp": self.tick_count,
            "velocity_x": vx,
            "velocity_y": vy,
            "altitude": alt,
            "roll": roll,
            "pitch": pitch,
            "motor_rpm": motor_rpm,
            "battery_voltage": true_voltage,
            "root_cause": "Nominal"
        }

def run_collection_mode():
    bridge = AirSimBridge()
    data = []
    print("\n[MODE: DATA COLLECTION]")
    
    try:
        while True:
            packet = bridge.get_data_packet()
            data.append(packet)
            
            if packet['timestamp'] % 10 == 0:
                print(f"Recording... Rows: {len(data)} | Alt: {packet['altitude']:.1f}m | Bat: {packet['battery_voltage']:.1f}V", end="\r")
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nStopping collection...")
        df = pd.DataFrame(data)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/train_airsim.csv", index=False)
        print(f"SAVED {len(df)} rows to 'data/train_airsim.csv'")

def run_live_inference_mode():
    bridge = AirSimBridge()
    
    # Load Models
    print("Loading Agomax Model...")
    try:
        import joblib
        pipeline = DronePipeline(id_column="timestamp")
        detector = AgomaxEnsemble()
        detector.load_models()
        
        scaler = joblib.load("models/scaler.pkl")
        cols = joblib.load("models/scaler_cols.pkl")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Did you run 'python main.py' to train first?")
        return

    output_file = "data/live_airsim_stream.csv"
    # Reset stream file
    with open(output_file, "w") as f:
        f.write("timestamp,velocity_x,velocity_y,altitude,roll,pitch,motor_rpm,battery_voltage,is_anomaly,root_cause_feature,severity_score\n")
    
    print("\n[MODE: LIVE INFERENCE]")
    print("Streaming to Dashboard...")
    
    while True:
        raw_packet = bridge.get_data_packet()
        
        # Prepare for AI
        df_row = pd.DataFrame([raw_packet])
        # Filter and ensure column order matches training
        input_data = df_row[cols]
        
        # Predict
        scaled_vals = scaler.transform(input_data)
        X_live = pd.DataFrame(scaled_vals, columns=cols)
        results = detector.predict(X_live)
        
        # Save to Stream
        is_anom = results['is_anomaly'].iloc[0]
        root = results['root_cause_feature'].iloc[0]
        score = results['severity_score'].iloc[0]
        
        with open(output_file, "a") as f:
            row = f"{raw_packet['timestamp']},{raw_packet['velocity_x']},{raw_packet['velocity_y']},{raw_packet['altitude']},{raw_packet['roll']},{raw_packet['pitch']},{raw_packet['motor_rpm']},{raw_packet['battery_voltage']},{is_anom},{root},{score}\n"
            f.write(row)
        
        status = "\033[91mCRITICAL\033[0m" if is_anom else "\033[92mNOMINAL\033[0m"
        print(f"T:{raw_packet['timestamp']} | {status} | Alt: {raw_packet['altitude']:.1f}m", end="\r")
        
        time.sleep(0.05)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect", "live"], required=True)
    args = parser.parse_args()
    
    if args.mode == "collect":
        run_collection_mode()
    else:
        run_live_inference_mode()
