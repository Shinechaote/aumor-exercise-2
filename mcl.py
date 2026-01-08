import numpy as np
import math
import struct
from scipy.stats import norm

# ROS Imports
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose
from sensor_msgs.msg import PointCloud2, PointField

NUM_PARTICLES = 100
MAP_X_BOUNDS = [-5, 5]
MAP_Y_BOUNDS = [-5, 5]

# Motion: error per odometry step [x (m), y (m), theta (rad)]
MOTION_NOISE_STD = [0.05, 0.05, 0.05] 
# Sensor: error in observation [x (m), y (m), theta (rad)]
SENSOR_NOISE_STD = [0.1, 0.1, 0.1] 

csv_data = np.genfromtxt("landmarks.csv", delimiter=',')
LANDMARKS_GT = {id: (x,y) for (id, x, y) in csv_data}

def normalize_angles(angles):
    """Normalizes angles to range [-pi, pi] for both scalars and arrays."""
    return np.arctan2(np.sin(angles), np.cos(angles))

class MCLNode(Node):
    def __init__(self):
        super().__init__('mcl_node')

        self.particles, self.weights = self.initialize_particles(NUM_PARTICLES)
        # Stores list of [x, y, theta, id]
        self.latest_landmarks = [] 
        # To calculate deltas
        self.last_odom_pose = None

        # Publishers and Subscribers
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, 'robot_estim', 10)
        self.pub_particles = self.create_publisher(PoseArray, 'particle_cloud', 10)
        
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

    def initialize_particles(self, num):
        """Spawns particles uniformly in the map bounds."""
        x = np.random.uniform(MAP_X_BOUNDS[0], MAP_X_BOUNDS[1], (num, 1))
        y = np.random.uniform(MAP_Y_BOUNDS[0], MAP_Y_BOUNDS[1], (num, 1))
        theta = np.random.uniform(-np.pi, np.pi, (num, 1))
        
        particles = np.concatenate((x, y, theta), axis=1)
        weights = np.ones(num) / num
        return particles, weights

    def landmarks_callback(self, msg):
        """
        Parses PointCloud2 to update the current observation.
        Expects fields: x, y, theta, id
        """
        parsed_landmarks = []
        
        # Helper to unpack PointCloud2 binary data
        # We define a simple generator for the specific structure
        
        # Calculate offsets based on fields (Simplified for readability)
        # Assuming float32 (4 bytes) for x, y, theta, id
        point_step = msg.point_step
        data = msg.data
        
        # Map field names to offsets
        offsets = {f.name: f.offset for f in msg.fields}
        required = ['x', 'y', 'theta', 'id']
        
        if not all(field in offsets for field in required):
            self.get_logger().warn("PointCloud2 missing required fields (x, y, theta, id)")
            return

        num_points = msg.width * msg.height
        
        for i in range(num_points):
            base = i * point_step
            # Unpack float32s
            x = struct.unpack_from('f', data, base + offsets['x'])[0]
            y = struct.unpack_from('f', data, base + offsets['y'])[0]
            theta = struct.unpack_from('f', data, base + offsets['theta'])[0]
            l_id = struct.unpack_from('f', data, base + offsets['id'])[0]
            
            parsed_landmarks.append([x, y, theta, int(l_id)])
            
        self.latest_landmarks = parsed_landmarks

    def odom_callback(self, msg):
        """
        Main Loop:
        1. Calculate odometry delta
        2. Motion Update (Predict)
        3. Measurement Update (Correct)
        4. Resample
        5. Publish
        """
        # Extract current odometry pose
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        curr_theta = normalize_angles(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2)))

        if self.last_odom_pose is None:
            self.last_odom_pose = (curr_x, curr_y, curr_theta)
            return

        # 1. Calculate Delta (Local Robot Frame)
        prev_x, prev_y, prev_theta = self.last_odom_pose
        
        # Global diff
        dx_global = curr_x - prev_x
        dy_global = curr_y - prev_y
        dtheta = normalize_angles(curr_theta - prev_theta)
        
        # Rotate global diff into local robot frame to get forward/sideways motion
        # This is the control input u_t
        dx_local = dx_global * np.cos(prev_theta) + dy_global * np.sin(prev_theta)
        dy_local = -dx_global * np.sin(prev_theta) + dy_global * np.cos(prev_theta)

        # 2. Motion Update
        self.particles = self.motion_update(self.particles, [dx_local, dy_local, dtheta], MOTION_NOISE_STD)

        # 3. Measurement Update (only if we have landmarks)
        if self.latest_landmarks:
            self.weights = self.measurement_update(
                self.particles, 
                self.weights, 
                self.latest_landmarks, 
                LANDMARKS_GT, 
                SENSOR_NOISE_STD
            )
            
            # 4. Resample (Adaptive: only if effective particles are low)
            # Simple check: if weights are too concentrated
            n_eff = 1.0 / np.sum(self.weights ** 2)
            if n_eff < NUM_PARTICLES / 2.0:
                self.particles, self.weights = self.resample(self.particles, self.weights)

        # 5. Publish Estimate
        self.publish_estimated_pose(msg.header)
        
        # Update last pose
        self.last_odom_pose = (curr_x, curr_y, curr_theta)

    def motion_update(self, particles, control, noise_std):
        """
        Applies motion model.
        control: [dx_local, dy_local, dtheta]
        """
        dx_l, dy_l, dtheta = control
        sig_x, sig_y, sig_theta = noise_std
        N = len(particles)

        # Add Noise to control input (Unique per particle)
        noisy_dx = dx_l + np.random.normal(0, sig_x, N)
        noisy_dy = dy_l + np.random.normal(0, sig_y, N)
        noisy_dtheta = dtheta + np.random.normal(0, sig_theta, N)

        p_x = particles[:, 0]
        p_y = particles[:, 1]
        p_theta = particles[:, 2]

        # Rotate local noisy motion into global frame based on particle's orientation
        # New X = Old X + (dx * cos(theta) - dy * sin(theta))
        new_x = p_x + (noisy_dx * np.cos(p_theta) - noisy_dy * np.sin(p_theta))
        new_y = p_y + (noisy_dx * np.sin(p_theta) + noisy_dy * np.cos(p_theta))
        new_theta = normalize_angles(p_theta + noisy_dtheta)

        return np.column_stack((new_x, new_y, new_theta))

    def measurement_update(self, particles, weights, landmarks_obs, landmarks_gt, noise_std):
        sig_x, sig_y, sig_theta = noise_std
        
        # Precompute distributions
        dist_x = norm(loc=0, scale=sig_x)
        dist_y = norm(loc=0, scale=sig_y)
        dist_theta = norm(loc=0, scale=sig_theta)

        p_x, p_y, p_theta = particles[:, 0], particles[:, 1], particles[:, 2]

        for (obs_x, obs_y, obs_theta, obs_id) in landmarks_obs:
            if obs_id not in landmarks_gt:
                continue
                
            gt_x, gt_y, gt_theta = landmarks_gt[obs_id]

            # Vectorized Expected Measurement
            dx_glob = gt_x - p_x
            dy_glob = gt_y - p_y
            
            exp_x = dx_glob * np.cos(p_theta) + dy_glob * np.sin(p_theta)
            exp_y = -dx_glob * np.sin(p_theta) + dy_glob * np.cos(p_theta)
            exp_theta = normalize_angles(gt_theta - p_theta)

            # Errors
            err_x = obs_x - exp_x
            err_y = obs_y - exp_y
            err_theta = normalize_angles(obs_theta - exp_theta)

            # Likelihood
            likelihood = dist_x.pdf(err_x) * dist_y.pdf(err_y) * dist_theta.pdf(err_theta)
            
            # Update weights
            weights *= likelihood

        # Normalize
        w_sum = np.sum(weights)
        if w_sum > 0:
            weights /= w_sum
        else:
            weights[:] = 1.0 / len(weights) # Reset if lost
            
        return weights

    def resample(self, particles, weights):
        N = len(particles)
        new_particles = np.zeros_like(particles)
        
        # Low Variance Resampling
        r = np.random.uniform(0, 1.0/N)
        c = weights[0]
        i = 0
        
        for m in range(N):
            u = r + m * (1.0/N)
            while u > c:
                i = (i + 1) % N
                c += weights[i]
            new_particles[m] = particles[i]
            
        new_weights = np.ones(N) / N
        return new_particles, new_weights

    def publish_estimated_pose(self, header):
        # Weighted mean estimate
        mean_x = np.average(self.particles[:, 0], weights=self.weights)
        mean_y = np.average(self.particles[:, 1], weights=self.weights)
        
        # Circular mean for angle
        sin_sum = np.sum(np.sin(self.particles[:, 2]) * self.weights)
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * self.weights)
        mean_theta = np.arctan2(sin_sum, cos_sum)

        # Create Message
        msg = PoseWithCovarianceStamped()
        msg.header = header
        msg.header.frame_id = "map" # Pose is in map frame
        
        msg.pose.pose.position.x = mean_x
        msg.pose.pose.position.y = mean_y
        msg.pose.pose.position.z = 0.0
        
        # Yaw to Quaternion
        cy = math.cos(mean_theta * 0.5)
        sy = math.sin(mean_theta * 0.5)
        msg.pose.pose.orientation.w = cy
        msg.pose.pose.orientation.z = sy
        
        self.pub_pose.publish(msg)

    def publish_particles(self, header):
        """Converts numpy particle array to PoseArray and publishes."""
        msg = PoseArray()
        msg.header = header
        msg.header.frame_id = "map" # Particles live in global map frame
        
        # We need to iterate to convert theta -> quaternion for each particle
        # Optimally, this can be vectorized too, but for <1000 particles a loop is fine
        for p in self.particles:
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            pose.position.z = 0.0
            
            # Simple yaw -> quaternion
            cy = math.cos(p[2] * 0.5)
            sy = math.sin(p[2] * 0.5)
            pose.orientation.w = cy
            pose.orientation.z = sy
            
            msg.poses.append(pose)
            
        self.pub_particles.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MCLNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
