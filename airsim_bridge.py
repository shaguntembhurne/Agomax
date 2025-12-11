import airsim
import time
import pandas as pd
import numpy as np
import os
import math
import argparse
from src.drone_anomaly.pipeline import DronePipeline
from src.drone_anomaly.ensemble import AgomaxEnsemble

# --- HELPER: Quaternion to Euler (Roll/Pitch/Yaw) ---
def to_euler(q_w, q_x, q_y, q_z):
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q_w * q_y - q_z * q_x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp) # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation) - Not used in our model but good to have
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll * (180/math.pi), pitch * (180/math.pi), yaw * (180/math.pi)

class AirSimBridge:
    def __init__(self):
        print("Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("CONNECTED to AirSim!")
        
        # State tracking for simulated sensors
        self.battery = 100.0
        self.start_time = time.time()
        self.tick_count = 0

    def get_data_packet(self):
        # 1. Get State from AirSim
        state = self.client.getMultirotorState()
        kinematics = state.kinematics_estimated
        
        # 2. Extract Physics
        # Velocity
        vx = kinematics.linear_velocity.x_val
        vy = kinematics.linear_velocity.y_val
        vz = -kinematics.linear_velocity.z_val # Invert Z (AirSim Z is down)
        
        # Altitude
        alt = -kinematics.position.z_val
        if alt < 0: alt = 0
        
        # Orientation (Quaternion -> Euler)
        q = kinematics.orientation
        roll, pitch, yaw = to_euler(q.w_val, q.x_val, q.y_val, q.z_val)

        # 3. Simulate Sensor Layers (RPM & Battery) based on real Physics
        # Calculate Load
        load = 10.0 + (abs(vz) * 5.0) + (math.sqrt(vx**2 + vy**2) * 0.5)
        
        # Motor RPM (Simulated based on load)
        motor_rpm = 5000 + (load * 100) + np.random.normal(0, 20)
        
        # Battery Physics
        voltage_sag = load * 0.01
        self.battery -= (load * 0.0001) # Slow drain
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
            "root_cause": "Nominal" # Default
        }

def run_collection_mode():
    bridge = AirSimBridge()
    data = []
    print("\n[MODE: DATA COLLECTION]")
    print("Fly your drone in AirSim now! (Press Ctrl+C to stop and save)")
    
    try:
        while True:
            packet = bridge.get_data_packet()
            data.append(packet)
            
            # Print status every 50 ticks
            if packet['timestamp'] % 10 == 0:
                print(f"Collecting... Rows: {len(data)} | Alt: {packet['altitude']:.1f}m")
            
            time.sleep(0.05) # 20Hz sample rate
            
    except KeyboardInterrupt:
        print("\nStopping collection...")
        df = pd.DataFrame(data)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/train_airsim.csv", index=False)
        print(f"SAVED {len(df)} rows to 'data/train_airsim.csv'")
        print("Now run: python main.py (to train on this data)")

def run_live_inference_mode():
    bridge = AirSimBridge()
    
    # Load Agomax
    print("Loading Agomax Model...")
    pipeline = DronePipeline(id_column="timestamp")
    # Hack: We need to load the scalar columns to know input shape
    scaler_cols = pd.read_pickle("models/scaler_cols.pkl") # Assuming joblib saved it differently or use pipeline logic
    
    detector = AgomaxEnsemble()
    detector.load_models()
    pipeline.scaler = detector.models # Loading trick if needed, or stick to pipeline loading
    
    # We write to a CSV that the dashboard watches
    output_file = "data/live_airsim_stream.csv"
    # Create header
    pd.DataFrame(columns=[
        "timestamp", "velocity_x", "velocity_y", "altitude", "roll", "pitch", 
        "motor_rpm", "battery_voltage", "is_anomaly", "root_cause_feature", "severity_score"
    ]).to_csv(output_file, index=False)
    
    print("\n[MODE: LIVE INFERENCE]")
    print("Streaming data to Dashboard... Fly the drone!")
    
    # Pre-load pipeline components for speed
    import joblib
    scaler = joblib.load("models/scaler.pkl")
    cols = joblib.load("models/scaler_cols.pkl")
    
    while True:
        # 1. Get Real Data
        raw_packet = bridge.get_data_packet()
        
        # 2. Prepare for AI
        df_row = pd.DataFrame([raw_packet])
        # Filter columns
        input_data = df_row[cols]
        
        # 3. Scale & Predict
        scaled_vals = scaler.transform(input_data)
        X_live = pd.DataFrame(scaled_vals, columns=cols)
        
        results = detector.predict(X_live)
        
        # 4. Merge results
        final_row = pd.concat([df_row.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
        
        # 5. Append to Stream File (Dashboard reads this)
        final_row.to_csv(output_file, mode='a', header=False, index=False)
        
        # Console Log
        is_anom = final_row['is_anomaly'].iloc[0]
        status = "\033[91mCRITICAL\033[0m" if is_anom else "\033[92mNOMINAL\033[0m"
        print(f"T:{raw_packet['timestamp']} | {status} | Alt: {raw_packet['altitude']:.1f} | Roll: {raw_packet['roll']:.1f}")
        
        time.sleep(0.05)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect", "live"], required=True, help="collect=Train Data, live=Detection")
    args = parser.parse_args()
    
    if args.mode == "collect":
        run_collection_mode()
    else:
        run_live_inference_mode()