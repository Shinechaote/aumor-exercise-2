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

# Motion: [x (m), y (m), theta (rad)]
MOTION_NOISE_STD = [0.05, 0.05, 0.05] 

# Sensor: [Range (m), Bearing (rad)]
# Increased these to be more forgiving
SENSOR_NOISE_STD = [0.5, 0.3] 

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def normalize_angles(angles):
    """Normalizes angles to range [-pi, pi]."""
    return np.arctan2(np.sin(angles), np.cos(angles))

def particles_to_pointcloud2(particles, header):
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
    msg.row_step = 16 * particles.shape[0]
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

        qos = QoSProfile(depth=10)
        
        # Publishers
        self.pub_estim_pose = self.create_publisher(Odometry, '/robot_estimated', qos)
        self.pub_particle_cloud = self.create_publisher(PointCloud2, '/particle_cloud', qos)
        self.pub_debug_landmarks = self.create_publisher(PointCloud2, '/debug_expected_landmarks', qos)
        
        # Subscribers
        self.sub_odom = self.create_subscription(Odometry, 'robot_noisy', self.odom_callback, 10)
        self.sub_landmarks = self.create_subscription(PointCloud2, 'landmarks_observed', self.landmarks_callback, 10)
        
        self.get_logger().info("MCL Node Started (Range-Bearing Model).")

    def load_landmarks(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, "landmarks.csv")
        try:
            csv_data = np.genfromtxt(file_path, delimiter=',')
            if csv_data.ndim == 1: csv_data = csv_data.reshape(1, -1)
            landmarks = {int(row[0]): (row[1], row[2]) for row in csv_data}
            
            all_x = csv_data[:, 1]
            all_y = csv_data[:, 2]
            bounds_x = [np.min(all_x) - 1.0, np.max(all_x) + 1.0]
            bounds_y = [np.min(all_y) - 1.0, np.max(all_y) + 1.0]
            return landmarks, (bounds_x, bounds_y)
        except Exception:
            return {}, ([-10, 10], [-10, 10])

    def initialize_particles(self, num):
        bx, by = self.map_bounds
        x = np.random.uniform(bx[0], bx[1], (num, 1))
        y = np.random.uniform(by[0], by[1], (num, 1))
        theta = np.random.uniform(-np.pi, np.pi, (num, 1))
        
        particles = np.concatenate((x, y, theta), axis=1)
        weights = np.ones(num) / num
        return particles, weights

    def landmarks_callback(self, msg):
        parsed = []
        point_step = msg.point_step
        data = msg.data
        offsets = {f.name: f.offset for f in msg.fields}
        
        if not all(k in offsets for k in ['x', 'y', 'id']): return

        num_points = msg.width * msg.height
        for i in range(num_points):
            base = i * point_step
            x = struct.unpack_from('f', data, base + offsets['x'])[0]
            y = struct.unpack_from('f', data, base + offsets['y'])[0]
            l_id = struct.unpack_from('f', data, base + offsets['id'])[0]
            parsed.append([x, y, int(l_id)])
        self.landmarks = parsed

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
        self.pub_particle_cloud.publish(particles_to_pointcloud2(self.particles, header))

    def publish_debug_observations(self, best_particle, observations, header):
        """Visualizes observations transformed to Map Frame based on best particle."""
        p_x, p_y, p_theta = best_particle
        debug_points = []
        
        for (obs_x, obs_y, obs_id) in observations:
            # Local (Robot) -> Global (Map)
            # 1. Rotate
            rot_x = obs_x * np.cos(p_theta) - obs_y * np.sin(p_theta)
            rot_y = obs_x * np.sin(p_theta) + obs_y * np.cos(p_theta)
            # 2. Translate
            map_x = p_x + rot_x
            map_y = p_y + rot_y
            debug_points.append([map_x, map_y, 0.0, float(obs_id)])
            
        # Manually pack pointcloud for debug
        msg = PointCloud2()
        msg.header = header
        msg.header.frame_id = "map"
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
        
        self.pub_debug_landmarks.publish(msg)

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
            self.weights = self.measurement_update_range_bearing(
                self.particles, 
                self.weights, 
                self.landmarks, 
                self.landmarks_gt, 
                SENSOR_NOISE_STD
            )
            
            # Normalize
            total_w = np.sum(self.weights)
            if total_w < 1.e-20:
                self.get_logger().warn("LOST: Respawning particles.")
                self.particles, self.weights = self.initialize_particles(NUM_PARTICLES)
            else:
                self.weights /= total_w

            # 3. Resample (Adaptive)
            n_eff = 1.0 / np.sum(self.weights ** 2)
            if n_eff < NUM_PARTICLES / 2.0:
                self.particles, self.weights = self.resample(self.particles, self.weights)
            
            # Debug Visual
            best_idx = np.argmax(self.weights)
            self.publish_debug_observations(self.particles[best_idx], self.landmarks, msg.header)

        # 4. Publish Results
        self.publish_estimated_pose(msg.header)
        self.publish_particle_cloud(msg.header)
        
        self.last_odom_pose = (curr_x, curr_y, curr_theta)

    def motion_update(self, particles, control, noise_std):
        dx_l, dy_l, dtheta = control
        sig_x, sig_y, sig_t = noise_std
        N = len(particles)

        # Randomized control input
        n_dx = dx_l + np.random.normal(0, sig_x, N)
        n_dy = dy_l + np.random.normal(0, sig_y, N)
        n_dtheta = dtheta + np.random.normal(0, sig_t, N)

        p_x, p_y, p_th = particles[:, 0], particles[:, 1], particles[:, 2]

        new_x = p_x + (n_dx * np.cos(p_th) - n_dy * np.sin(p_th))
        new_y = p_y + (n_dx * np.sin(p_th) + n_dy * np.cos(p_th))
        new_th = normalize_angles(p_th + n_dtheta)

        return np.column_stack((new_x, new_y, new_th))

    def measurement_update_range_bearing(self, particles, weights, obs, landmarks_gt, noise_std):
        sig_r, sig_phi = noise_std
        dist_r = norm(loc=0, scale=sig_r)
        dist_phi = norm(loc=0, scale=sig_phi)

        p_x = particles[:, 0]
        p_y = particles[:, 1]
        p_theta = particles[:, 2]

        # Reset weights to 1.0 for this step's calculation
        # (We multiply cumulatively for each landmark in this frame)
        step_weights = np.ones_like(weights)

        for (ox, oy, oid) in obs:
            if oid not in landmarks_gt: continue
            gt_x, gt_y = landmarks_gt[oid]

            # 1. Observed Range/Bearing
            obs_range = np.hypot(ox, oy)
            obs_bearing = np.arctan2(oy, ox)

            # 2. Expected Range/Bearing
            dx = gt_x - p_x
            dy = gt_y - p_y
            exp_range = np.hypot(dx, dy)
            # The critical angle calculation: Angle to landmark - Particle Heading
            exp_bearing = normalize_angles(np.arctan2(dy, dx) - p_theta)

            # 3. Errors
            err_r = obs_range - exp_range
            err_phi = normalize_angles(obs_bearing - exp_bearing)

            # 4. Likelihood
            # Adding +1.e-50 keeps the particle "alive" even if one landmark is weird.
            # This prevents the "All particles lost" scenario.
            lik = (dist_r.pdf(err_r) * dist_phi.pdf(err_phi)) + 1.e-50
            step_weights *= lik

        return step_weights

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
