import airsim
import time
import pandas as pd
import numpy as np
import os
import math
import sys

# =====================================
#  MANUAL CONTROL VERSION (YOU FLY)
#  Python ONLY reads data.
# =====================================

# --- Quaternion → Euler angles ---
def to_euler(q_w, q_x, q_y, q_z):
    sinr_cosp = 2 * (q_w*q_x + q_y*q_z)
    cosr_cosp = 1 - 2 * (q_x*q_x + q_y*q_y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (q_w*q_y - q_z*q_x)
    pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi/2, sinp)

    siny_cosp = 2 * (q_w*q_z + q_x*q_y)
    cosy_cosp = 1 - 2 * (q_y*q_y + q_z*q_z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll*57.3, pitch*57.3, yaw*57.3


class ManualAirSimBridge:
    def __init__(self):
        print("Connecting to AirSim...")

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # Python grabs control temporarily to arm & takeoff
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        print("✓ Drone armed & taken off")

        # Give control BACK to user manually
        self.client.enableApiControl(False)

        print("\n=== MANUAL FLIGHT MODE ACTIVE ===")
        print("You must fly using keyboard inside Blocks window:")
        print("▶ F = Takeoff (already done)")
        print("▶ Arrow Keys = Move")
        print("▶ A/D = Rotate")
        print("▶ Space = Brake")
        print("==================================\n")

        self.tick = 0
        self.battery = 100.0

    def get_packet(self):
        state = self.client.getMultirotorState()
        k = state.kinematics_estimated

        vx = k.linear_velocity.x_val
        vy = k.linear_velocity.y_val
        vz = -k.linear_velocity.z_val

        altitude = max(0, -k.position.z_val)

        q = k.orientation
        roll, pitch, yaw = to_euler(q.w_val, q.x_val, q.y_val, q.z_val)

        # Fake motor rpm & battery (AirSim doesn't provide these)
        load = abs(vx)+abs(vy)+abs(vz)
        motor_rpm = 5000 + load*120 + np.random.normal(0, 40)
        self.battery -= 0.0002*load
        battery = max(0, self.battery)

        self.tick += 1

        return {
            "timestamp": self.tick,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "altitude": altitude,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "motor_rpm": motor_rpm,
            "battery": battery
        }


def collect():
    drone = ManualAirSimBridge()
    data = []

    print("Collecting telemetry (Ctrl+C to stop)...\n")

    try:
        while True:
            pkt = drone.get_packet()
            data.append(pkt)

            if pkt["timestamp"] % 10 == 0:
                print(f"t={pkt['timestamp']} alt={pkt['altitude']:.2f}", end="\r")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nSaving...")

        df = pd.DataFrame(data)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/train_airsim_manual.csv", index=False)

        print("Saved → data/train_airsim_manual.csv")


if __name__ == "__main__":
    collect()
