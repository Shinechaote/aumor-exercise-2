import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Use the high-level reader which handles deserialization automatically
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Folder structure expected:
# ./bags/B1_particles/100_particles.mcap
# ./bags/B1_particles/500_particles.mcap
# ...
BASE_DIR = "./bags"

# Topics
GT_TOPIC = "/robot_gt"      # Type: nav_msgs/Odometry
EST_TOPIC = "/robot_estimated" # Type: nav_msgs/Odometry or geometry_msgs/PoseWithCovarianceStamped
CLOUD_TOPIC = "/particle_cloud" # Type: sensor_msgs/PointCloud2 (Optional for convergence)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def get_yaw(q):
    """Calculates yaw from quaternion (x, y, z, w)."""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)

def calculate_rmse(gt_data, est_data):
    """
    Calculates Root Mean Square Error between Ground Truth and Estimate.
    Matches timestamps using nearest neighbor search.
    """
    if not gt_data or not est_data:
        return float('nan'), float('nan')

    gt_df = pd.DataFrame(gt_data, columns=['t', 'x', 'y', 'theta'])
    est_df = pd.DataFrame(est_data, columns=['t', 'x', 'y', 'theta'])

    # Sort by time
    gt_df = gt_df.sort_values('t')
    est_df = est_df.sort_values('t')

    squared_errors_pos = []
    squared_errors_ang = []

    # Simple nearest neighbor synchronization
    for _, est_row in est_df.iterrows():
        # Find closest GT timestamp
        t_est = est_row['t']
        closest_gt_idx = (gt_df['t'] - t_est).abs().idxmin()
        gt_row = gt_df.loc[closest_gt_idx]

        # Sync check: if time diff > 0.1s, skip (tracking lost/lag)
        if abs(gt_row['t'] - t_est) > 0.1: 
            continue

        # Position Error
        d_pos = (est_row['x'] - gt_row['x'])**2 + (est_row['y'] - gt_row['y'])**2
        squared_errors_pos.append(d_pos)

        # Angle Error (handle wrap around)
        d_theta = est_row['theta'] - gt_row['theta']
        d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
        squared_errors_ang.append(d_theta**2)

    if not squared_errors_pos:
        return float('nan'), float('nan')

    rmse_pos = np.sqrt(np.mean(squared_errors_pos))
    rmse_ang = np.sqrt(np.mean(squared_errors_ang))

    return rmse_pos, rmse_ang

def calculate_convergence_time(est_data, gt_data, error_threshold=0.5):
    """
    Finds the time (seconds from start) where error consistently drops below threshold.
    """
    if not gt_data or not est_data: return float('nan')
    
    start_time = gt_data[0][0]
    gt_df = pd.DataFrame(gt_data, columns=['t', 'x', 'y', 'theta']).sort_values('t')
    est_df = pd.DataFrame(est_data, columns=['t', 'x', 'y', 'theta']).sort_values('t')
    
    converged_at = None
    
    # Check window of last N samples to ensure stability
    window_size = 10 
    errors = []

    for _, est_row in est_df.iterrows():
        t_est = est_row['t']
        closest_gt_idx = (gt_df['t'] - t_est).abs().idxmin()
        gt_row = gt_df.loc[closest_gt_idx]

        dist = np.sqrt((est_row['x'] - gt_row['x'])**2 + (est_row['y'] - gt_row['y'])**2)
        errors.append(dist)
        
        if len(errors) > window_size:
            errors.pop(0)
            # If the max error in the window is below threshold, we have converged
            if max(errors) < error_threshold:
                converged_at = t_est
                break # First time it stabilized
    
    if converged_at:
        return converged_at - start_time
    return float('nan') # Never converged

def read_bag(bag_path):
    """
    Reads a bag file using AnyReader (works for both ROS1 and ROS2).
    Automatically handles deserialization.
    """
    gt_data = []
    est_data = []
    
    print(f"Reading: {bag_path}")
    bag_path_obj = Path(bag_path)
    
    # Create a typestore to hold message definitions
    typestore = get_typestore(Stores.ROS2_KILTED) 

    try:
        with AnyReader([bag_path_obj], default_typestore=typestore) as reader:
            for connection, timestamp, rawdata in reader.messages():
                # Automatic deserialization
                msg = reader.deserialize(rawdata, connection.msgtype)
                t_sec = timestamp * 1e-9

                if connection.topic == GT_TOPIC:
                    # Expecting nav_msgs/Odometry
                    p = msg.pose.pose.position
                    q = msg.pose.pose.orientation
                    gt_data.append([t_sec, p.x, p.y, get_yaw(q)])

                elif connection.topic == EST_TOPIC:
                    # Handles Odometry OR PoseWithCovarianceStamped
                    if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
                        # This structure fits both Odometry and PoseWithCovarianceStamped
                        p = msg.pose.pose.position
                        q = msg.pose.pose.orientation
                        est_data.append([t_sec, p.x, p.y, get_yaw(q)])
                        
    except Exception as e:
        print(f"Error reading {bag_path}: {e}")
        
    return gt_data, est_data
# -----------------------------------------------------------------------------
# MAIN ANALYSIS LOOP
# -----------------------------------------------------------------------------

def analyze_b1_particles(folder_path):
    print("\n--- Task B1: Particle Count Variation ---")
    results = []
    
    # Expected filenames format: "100.mcap", "500.mcap", etc.
    # OR map manually:
    experiments = {
        100: "particles_100", 
        500: "particles_500", 
        1000: "particles_1000", 
        2000: "particles_2000", 
        5000: "particles_5000"
    }

    for count, bag_name in experiments.items():
        # Find the full path (handle extensions likely .mcap or folder)
        full_path = None
        for f in os.listdir(folder_path):
            if bag_name in f:
                full_path = os.path.join(folder_path, f)
                break
        
        if not full_path:
            print(f"Warning: Bag for {count} particles not found.")
            continue

        gt, est = read_bag(full_path)
        rmse_pos, rmse_ang = calculate_rmse(gt, est)
        conv_time = calculate_convergence_time(est, gt)
        
        results.append({
            "Particle Count": count,
            "RMSE Position (m)": rmse_pos,
            "RMSE Angle (rad)": rmse_ang,
            "Convergence Time (s)": conv_time
        })

    df = pd.DataFrame(results).sort_values("Particle Count")
    print(df)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Particle Count (N)')
    ax1.set_ylabel('RMSE Position (m)', color=color)
    ax1.plot(df['Particle Count'], df['RMSE Position (m)'], marker='o', color=color, label='RMSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Convergence Time (s)', color=color)  # we already handled the x-label with ax1
    ax2.plot(df['Particle Count'], df['Convergence Time (s)'], marker='x', linestyle='--', color=color, label='Time')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Task B1: RMSE & Convergence vs Particle Count')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('B1_Result.png')
    plt.show()

def analyze_b2_motion(folder_path):
    print("\n--- Task B2: Motion Noise Variation ---")
    results = []
    
    # Map noise value (std dev) to filename
    experiments = {
        0.01: "noise_0_01",
        0.05: "noise_0_05",
        0.10: "noise_0_10",
        0.20: "noise_0_20",
        0.50: "noise_0_50"
    }

    for noise, bag_name in experiments.items():
        full_path = None
        for f in os.listdir(folder_path):
            if bag_name in f:
                full_path = os.path.join(folder_path, f)
                break
        if not full_path: continue

        gt, est = read_bag(full_path)
        rmse_pos, _ = calculate_rmse(gt, est)
        
        results.append({
            "Motion Noise Std": noise,
            "RMSE Position (m)": rmse_pos
        })

    df = pd.DataFrame(results).sort_values("Motion Noise Std")
    print(df)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="Motion Noise Std", y="RMSE Position (m)", marker='o')
    plt.title('Task B2: Localization Error vs Motion Noise')
    plt.xlabel('Motion Noise Std Dev (m)')
    plt.ylabel('RMSE (m)')
    plt.grid(True)
    plt.savefig('B2_Result.png')
    plt.show()

def analyze_b3_sensor(folder_path):
    print("\n--- Task B3: Sensor Noise Variation ---")
    results = []
    
    # Map noise value to filename
    experiments = {
        0.05: "sensor_0_05",
        0.10: "sensor_0_10",
        0.30: "sensor_0_30",
        0.50: "sensor_0_50",
        1.00: "sensor_1_00"
    }

    for noise, bag_name in experiments.items():
        full_path = None
        for f in os.listdir(folder_path):
            if bag_name in f:
                full_path = os.path.join(folder_path, f)
                break
        if not full_path: continue

        gt, est = read_bag(full_path)
        rmse_pos, _ = calculate_rmse(gt, est)
        
        results.append({
            "Sensor Noise Std": noise,
            "RMSE Position (m)": rmse_pos
        })

    df = pd.DataFrame(results).sort_values("Sensor Noise Std")
    print(df)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Sensor Noise Std", y="RMSE Position (m)", palette="viridis")
    plt.title('Task B3: Impact of Sensor Noise on Accuracy')
    plt.xlabel('Sensor Noise Std Dev (m)')
    plt.ylabel('RMSE (m)')
    plt.savefig('B3_Result.png')
    plt.show()

if __name__ == "__main__":
    # Update these paths to match your actual computer's folder structure
    # Example:
    # bags/
    #   B1/
    #     particles_100.mcap
    #   B2/ ...
    
    analyze_b1_particles(os.path.join(BASE_DIR, "B1"))
    analyze_b2_motion(os.path.join(BASE_DIR, "B2"))
    analyze_b3_sensor(os.path.join(BASE_DIR, "B3"))
