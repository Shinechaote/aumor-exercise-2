import numpy as np
import math
import struct
import os
from scipy.stats import norm

# ROS Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
NUM_PARTICLES = 100
# Motion: error per odometry step [x (m), y (m), theta (rad)]
MOTION_NOISE_STD = [0.05, 0.05, 0.05] 
# Sensor: error in observation [x (m), y (m), theta (rad)]
SENSOR_NOISE_STD = [0.1, 0.1, 0.1] 

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def normalize_angles(angles):
    return np.arctan2(np.sin(angles), np.cos(angles))

def particles_to_pointcloud2(particles, header):
    """
    Converts a numpy array of particles [x, y, theta] into a PointCloud2 message.
    """
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(particles)
    
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    
    msg.is_bigendian = False
    msg.point_step = 16 
    msg.row_step = msg.point_step * particles.shape[0]
    msg.is_dense = True
    
    buffer = []
    for p in particles:
        buffer.append(struct.pack('ffff', p[0], p[1], 0.0, p[2]))
        
    msg.data = b''.join(buffer)
    return msg

class MCLNode(Node):
    def __init__(self):
        super().__init__('mcl_node')

        self.landmarks_gt, self.map_bounds = self.load_landmarks()
        self.particles, self.weights = self.initialize_particles(NUM_PARTICLES)
        self.landmarks = [] 
        self.last_odom_pose = None

        # Publishers and Subscribers
        qos = QoSProfile(depth=10)
        
        self.pub_estim_pose = self.create_publisher(
            Odometry, 
            '/robot_estimated', 
            qos
        )
        
        self.pub_particle_cloud = self.create_publisher(
            PointCloud2, 
            '/particle_cloud', 
            qos
        )
        
        self.sub_odom = self.create_subscription(
            Odometry, 
            'robot_noisy', 
            self.odom_callback, 
            10
        )
        
        self.sub_landmarks = self.create_subscription(
            PointCloud2, 
            'landmarks_observed', 
            self.landmarks_callback, 
            10
        )
        
        self.get_logger().info("MCL Node Started. Waiting for odometry...")
        self.pub_debug_obs = self.create_publisher(PointCloud2, '/debug_expected_landmarks', 10)

    def publish_debug_observations(self, best_particle, observations, header):
        """
        Visualizes where the BEST particle thinks the OBSERVED landmarks are.
        Transform: Local (Sensor) -> Global (Map)
        """
        p_x, p_y, p_theta = best_particle
        
        debug_points = []
        
        for (obs_x, obs_y, obs_id) in observations:
            # 1. ROTATE (Local -> Global rotation)
            # Standard Rigid Body Transform:
            # x_global = x_local * cos(theta) - y_local * sin(theta)
            # y_global = x_local * sin(theta) + y_local * cos(theta)
            
            # Note: obs_x is forward, obs_y is left
            rot_x = obs_x * np.cos(p_theta) - obs_y * np.sin(p_theta)
            rot_y = obs_x * np.sin(p_theta) + obs_y * np.cos(p_theta)
            
            # 2. TRANSLATE (Add particle position)
            map_x = p_x + rot_x
            map_y = p_y + rot_y
            
            # Pack for PointCloud2 (x, y, z, intensity=id)
            debug_points.append([map_x, map_y, 0.0, float(obs_id)])
            
        # Convert to PointCloud2
        # (Reusing the manual packing for simplicity here to ensure no dependency errors)
        msg = PointCloud2()
        msg.header = header
        msg.header.frame_id = "map" # We are publishing into the MAP frame
        msg.height = 1
        msg.width = len(debug_points)
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.is_bigendian = False
        msg.point_step = 16 
        msg.row_step = 16 * len(debug_points)
        msg.is_dense = True
        
        buffer = []
        for p in debug_points:
            buffer.append(struct.pack('ffff', p[0], p[1], p[2], p[3]))
            
        msg.data = b''.join(buffer)
        
        # You need to create this publisher in __init__:
        # self.pub_debug_obs = self.create_publisher(PointCloud2, '/debug_obs_global', 10)
        self.pub_debug_obs.publish(msg)

    def load_landmarks(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, "landmarks.csv")
        try:
            csv_data = np.genfromtxt(file_path, delimiter=',')
            # Handle single line CSV case
            if csv_data.ndim == 1:
                csv_data = csv_data.reshape(1, -1)

            landmarks = {int(row[0]): (row[1], row[2]) for row in csv_data}
            
            all_x = csv_data[:, 1]
            all_y = csv_data[:, 2]
            bounds_x = [np.min(all_x) - 1.0, np.max(all_x) + 1.0]
            bounds_y = [np.min(all_y) - 1.0, np.max(all_y) + 1.0]
            return landmarks, (bounds_x, bounds_y)
        except Exception as e:
            self.get_logger().warn(f"Could not load landmarks.csv: {e}")
            return {}, ([-5, 5], [-5, 5])

    def initialize_particles(self, num):
        bounds_x, bounds_y = self.map_bounds
        x = np.random.uniform(bounds_x[0], bounds_x[1], (num, 1))
        y = np.random.uniform(bounds_y[0], bounds_y[1], (num, 1))
        theta = np.random.uniform(-np.pi, np.pi, (num, 1))
        
        particles = np.concatenate((x, y, theta), axis=1)
        weights = np.ones(num) / num
        return particles, weights

    def landmarks_callback(self, msg):
        parsed_landmarks = []
        point_step = msg.point_step
        data = msg.data
        offsets = {f.name: f.offset for f in msg.fields}
        
        if not all(k in offsets for k in ['x', 'y', 'id']):
            return

        num_points = msg.width * msg.height
        for i in range(num_points):
            base = i * point_step
            x = struct.unpack_from('f', data, base + offsets['x'])[0]
            y = struct.unpack_from('f', data, base + offsets['y'])[0]
            l_id = struct.unpack_from('I', data, base + offsets['id'])[0]
            parsed_landmarks.append([x, y, int(l_id)])
            
        self.landmarks = parsed_landmarks

    def publish_estimated_pose(self, header):
        mean_x = np.average(self.particles[:, 0], weights=self.weights)
        mean_y = np.average(self.particles[:, 1], weights=self.weights)
        sin_sum = np.sum(np.sin(self.particles[:, 2]) * self.weights)
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * self.weights)
        mean_theta = np.arctan2(sin_sum, cos_sum)

        msg = Odometry()
        msg.header = header
        msg.header.frame_id = "map"
        msg.child_frame_id = "robot_estimated"
        
        msg.pose.pose.position.x = mean_x
        msg.pose.pose.position.y = mean_y
        
        cy = math.cos(mean_theta * 0.5)
        sy = math.sin(mean_theta * 0.5)
        msg.pose.pose.orientation.w = cy
        msg.pose.pose.orientation.z = sy
        
        self.pub_estim_pose.publish(msg)

    def publish_particle_cloud(self, header):
        pc2_msg = particles_to_pointcloud2(self.particles, header)
        self.pub_particle_cloud.publish(pc2_msg)

    def odom_callback(self, msg):
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        curr_theta = normalize_angles(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2)))

        if self.last_odom_pose is None:
            self.last_odom_pose = (curr_x, curr_y, curr_theta)
            return

        prev_x, prev_y, prev_theta = self.last_odom_pose
        dx_global = curr_x - prev_x
        dy_global = curr_y - prev_y
        dtheta = normalize_angles(curr_theta - prev_theta)
        
        dx_local = dx_global * np.cos(prev_theta) + dy_global * np.sin(prev_theta)
        dy_local = -dx_global * np.sin(prev_theta) + dy_global * np.cos(prev_theta)

        # 1. Motion Update
        self.particles = self.motion_update(self.particles, [dx_local, dy_local, dtheta], MOTION_NOISE_STD)

        # 2. Measurement Update
        if self.landmarks:
            self.weights = self.measurement_update(
                self.particles, 
                self.weights, 
                self.landmarks, 
                self.landmarks_gt, 
                SENSOR_NOISE_STD
            )
            total_w = np.sum(self.weights)

            if total_w < 1e-20:
                self.particles, self.weights = self.initialize_particles(NUM_PARTICLES)
            else:
                self.weights = self.weights / total_w
            
            # 3. Resample (Adaptive)
            # Only resample if effective particle count is low
            n_eff = 1.0 / np.sum(self.weights ** 2)
            if n_eff < NUM_PARTICLES / 2.0:
                self.particles, self.weights = self.resample(self.particles, self.weights)

            best_particle_idx = np.argmax(self.weights)
            best_particle = self.particles[best_particle_idx]

        # Publish the observations as seen by this specific particle
        if self.landmarks:
             self.publish_debug_observations(best_particle, self.landmarks, msg.header)

        # 4. Publish Results
        self.publish_estimated_pose(msg.header)
        self.publish_particle_cloud(msg.header)
        
        self.last_odom_pose = (curr_x, curr_y, curr_theta)

    def motion_update(self, particles, control, noise_std):
        dx_l, dy_l, dtheta = control
        sig_x, sig_y, sig_theta = noise_std
        N = len(particles)

        noisy_dx = dx_l + np.random.normal(0, sig_x, N)
        noisy_dy = dy_l + np.random.normal(0, sig_y, N)
        noisy_dtheta = dtheta + np.random.normal(0, sig_theta, N)

        p_x = particles[:, 0]
        p_y = particles[:, 1]
        p_theta = particles[:, 2]

        new_x = p_x + (noisy_dx * np.cos(p_theta) - noisy_dy * np.sin(p_theta))
        new_y = p_y + (noisy_dx * np.sin(p_theta) + noisy_dy * np.cos(p_theta))
        new_theta = normalize_angles(p_theta + noisy_dtheta)

        return np.column_stack((new_x, new_y, new_theta))

    def measurement_update(self, particles, weights, landmarks_obs, landmarks_gt, noise_std):
        sig_x, sig_y, sig_theta = noise_std
        dist_x = norm(loc=0, scale=sig_x)
        dist_y = norm(loc=0, scale=sig_y)
        dist_theta = norm(loc=0, scale=sig_theta)

        p_x = particles[:, 0]
        p_y = particles[:, 1]
        p_theta = particles[:, 2]
        
        # Reset step likelihoods
        step_weights = np.ones_like(weights)

        for (obs_x, obs_y, obs_id) in landmarks_obs:
            if obs_id not in landmarks_gt:
                continue
                
            gt_x, gt_y = landmarks_gt[obs_id]

            # Expected Measurement
            dx_glob = gt_x - p_x
            dy_glob = gt_y - p_y
            
            exp_x = dx_glob * np.cos(p_theta) + dy_glob * np.sin(p_theta)
            exp_y = -dx_glob * np.sin(p_theta) + dy_glob * np.cos(p_theta)
            exp_theta = normalize_angles(np.arctan2(dy_glob, dx_glob) - p_theta)

            # Observed Measurement
            obs_theta = np.arctan2(obs_y, obs_x)

            # Errors
            err_x = obs_x - exp_x
            err_y = obs_y - exp_y
            err_theta = normalize_angles(obs_theta - exp_theta)

            # Likelihood
            lik = dist_x.pdf(err_x) * dist_y.pdf(err_y) * dist_theta.pdf(err_theta)
            step_weights *= lik + 1e-9

        # Update and Normalize
        weights *= step_weights
        return weights

    def resample(self, particles, weights):
        N = len(particles)
        new_particles = np.zeros_like(particles)
        r = np.random.uniform(0, 1.0/N)
        c = weights[0]
        i = 0
        for m in range(N):
            u = r + m * (1.0/N)
            while u > c:
                i = (i + 1) % N
                c += weights[i]
            new_particles[m] = particles[i]
        return new_particles, np.ones(N) / N

def main(args=None):
    rclpy.init(args=args)
    node = MCLNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
