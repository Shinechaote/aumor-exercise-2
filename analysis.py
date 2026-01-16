import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Rosbags imports
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = "./bags"
GT_TOPIC = "/robot_gt" 
EST_TOPIC = "/robot_estimated"

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def get_yaw(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)

def get_error_series(gt_data, est_data):
    if not gt_data or not est_data:
        return [], []

    gt_df = pd.DataFrame(gt_data, columns=['t', 'x', 'y', 'theta']).sort_values('t')
    est_df = pd.DataFrame(est_data, columns=['t', 'x', 'y', 'theta']).sort_values('t')
    
    start_time = gt_df.iloc[0]['t']
    times = []
    errors = []

    for _, est_row in est_df.iterrows():
        t_est = est_row['t']
        closest_idx = (gt_df['t'] - t_est).abs().idxmin()
        gt_row = gt_df.loc[closest_idx]

        if abs(gt_row['t'] - t_est) > 0.1: 
            continue

        dist = np.sqrt((est_row['x'] - gt_row['x'])**2 + (est_row['y'] - gt_row['y'])**2)
        times.append(t_est - start_time)
        errors.append(dist)
        
    return times, errors

def calculate_metrics(gt_data, est_data, conv_threshold=0.5):
    """
    Returns: RMSE, Convergence Time, Stability (Std Dev of Error)
    """
    times, errors = get_error_series(gt_data, est_data)
    
    if not errors: return float('nan'), float('nan'), float('nan')

    # 1. RMSE
    rmse = np.sqrt(np.mean(np.array(errors)**2))

    # 2. Stability (Standard Deviation of the error)
    # Lower std dev means the filter is more "stable" (less jittery)
    stability = np.std(errors)

    # 3. Convergence Time
    conv_time = float('nan')
    window_size = 10
    for i in range(len(errors) - window_size):
        window = errors[i : i+window_size]
        if max(window) < conv_threshold:
            conv_time = times[i]
            break

    return rmse, conv_time, stability

def read_bag(bag_path):
    gt_data = []
    est_data = []
    
    print(f"Reading: {bag_path}")
    bag_path_obj = Path(bag_path)
    typestore = get_typestore(Stores.ROS2_HUMBLE) 

    try:
        with AnyReader([bag_path_obj], default_typestore=typestore) as reader:
            for connection, timestamp, rawdata in reader.messages():
                msg = reader.deserialize(rawdata, connection.msgtype)
                t_sec = timestamp * 1e-9

                if connection.topic == GT_TOPIC:
                    p = msg.pose.pose.position
                    q = msg.pose.pose.orientation
                    gt_data.append([t_sec, p.x, p.y, get_yaw(q)])

                elif connection.topic == EST_TOPIC:
                    if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
                        p = msg.pose.pose.position
                        q = msg.pose.pose.orientation
                        est_data.append([t_sec, p.x, p.y, get_yaw(q)])
                        
    except Exception as e:
        print(f"Error reading {bag_path}: {e}")
        
    return gt_data, est_data

# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def plot_task_results(task_name, df_summary, history_data, x_label, metric2_col="Conv Time (s)"):
    """
    Plots RMSE and a secondary metric (Convergence OR Stability).
    """
    if df_summary.empty:
        print(f"No data for {task_name}")
        return

    # --- Plot 1: RMSE Summary ---
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_summary, x=x_label, y="RMSE (m)", marker='o', linewidth=2)
    plt.title(f"{task_name}: RMSE vs {x_label}")
    plt.grid(True)
    plt.savefig(f"{task_name}_Summary_RMSE.png")
    plt.close()

    # --- Plot 2: Secondary Metric (Convergence or Stability) ---
    plt.figure(figsize=(8, 5))
    # We define color based on metric to avoid confusion
    color = 'green' if "Stability" in metric2_col else 'orange'
    
    sns.lineplot(data=df_summary, x=x_label, y=metric2_col, marker='s', color=color, linewidth=2)
    plt.title(f"{task_name}: {metric2_col} vs {x_label}")
    plt.grid(True)
    
    # Clean filename
    metric_fname = metric2_col.split(' ')[0]
    plt.savefig(f"{task_name}_Summary_{metric_fname}.png")
    plt.close()

    # --- Plot 3...N: Individual Error Histories ---
    for label_val, times, errors in history_data:
        plt.figure(figsize=(10, 6))
        step = max(1, len(times)//2000) 
        plt.plot(times[::step], errors[::step], linewidth=1.5, color='tab:blue')
        
        safe_label = str(label_val).replace('.', '_')
        plt.title(f"{task_name} Run: {label_val} {x_label}")
        plt.xlabel("Time (s)")
        plt.ylabel("Position Error (m)")
        plt.grid(True)
        plt.ylim(bottom=0)
        
        filename = f"{task_name}_History_{safe_label}_{x_label}.png"
        plt.savefig(filename)
        plt.close()

# -----------------------------------------------------------------------------
# ANALYSIS TASKS
# -----------------------------------------------------------------------------

def analyze_b1_particles(folder_path):
    print("\n--- Processing Task B1 (Particles) ---")
    results = []
    history = []
    
    experiments = {
        100: "particles_100", 500: "particles_500", 1000: "particles_1000", 
        2000: "particles_2000", 5000: "particles_5000"
    }

    if not os.path.exists(folder_path): return

    for count, name in experiments.items():
        found = next((f for f in os.listdir(folder_path) if name in f), None)
        if not found: continue
        
        gt, est = read_bag(os.path.join(folder_path, found))
        rmse, conv, stab = calculate_metrics(gt, est)
        t, err = get_error_series(gt, est)
        
        # B1 uses Convergence Time
        results.append({"Particles": count, "RMSE (m)": rmse, "Conv Time (s)": conv})
        history.append((count, t, err))

    df = pd.DataFrame(results).sort_values("Particles")
    plot_task_results("B1", df, history, "Particles", metric2_col="Conv Time (s)")

def analyze_b2_motion(folder_path):
    print("\n--- Processing Task B2 (Motion Noise) ---")
    results = []
    history = []
    
    experiments = {0.01: "noise_0_01", 0.05: "noise_0_05", 0.1: "noise_0_10", 0.2: "noise_0_20", 0.5: "noise_0_50"}

    if not os.path.exists(folder_path): return

    for noise, name in experiments.items():
        found = next((f for f in os.listdir(folder_path) if name in f), None)
        if not found: continue
        
        gt, est = read_bag(os.path.join(folder_path, found))
        rmse, conv, stab = calculate_metrics(gt, est)
        t, err = get_error_series(gt, est)
        
        # B2 uses STABILITY instead of Convergence
        results.append({"Noise": noise, "RMSE (m)": rmse, "Stability (m)": stab})
        history.append((noise, t, err))

    df = pd.DataFrame(results).sort_values("Noise")
    # Pass 'Stability (m)' as the second metric to plot
    plot_task_results("B2", df, history, "Noise", metric2_col="Stability (m)")

def analyze_b3_sensor(folder_path):
    print("\n--- Processing Task B3 (Sensor Noise) ---")
    results = []
    history = []
    
    experiments = {0.05: "sensor_0_05", 0.1: "sensor_0_10", 0.3: "sensor_0_30", 0.5: "sensor_0_50", 1.0: "sensor_1_00"}

    if not os.path.exists(folder_path): return

    for noise, name in experiments.items():
        found = next((f for f in os.listdir(folder_path) if name in f), None)
        if not found: continue
        
        gt, est = read_bag(os.path.join(folder_path, found))
        rmse, conv, stab = calculate_metrics(gt, est)
        t, err = get_error_series(gt, est)
        
        # B3 uses Convergence Time (Default)
        results.append({"Noise": noise, "RMSE (m)": rmse, "Conv Time (s)": conv})
        history.append((noise, t, err))

    df = pd.DataFrame(results).sort_values("Noise")
    plot_task_results("B3", df, history, "Noise", metric2_col="Conv Time (s)")

if __name__ == "__main__":
    analyze_b1_particles(os.path.join(BASE_DIR, "B1"))
    analyze_b2_motion(os.path.join(BASE_DIR, "B2"))
    analyze_b3_sensor(os.path.join(BASE_DIR, "B3"))
