import rclpy
from rclpy.node import Node
import random
import math
import numpy as np
import os
import struct

# ROS2 Nachrichten-Imports
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose, PoseWithCovarianceStamped

class MCLNode(Node):
    def __init__(self):
        super().__init__('mcl_node')

        # --- Variablen & Parameter ---
        self.num_particles = 300
        self.last_odom_pose = None 
        
        # --- Task A1: Initialisierung (Global) ---
        self.particles = self.init_particles(self.num_particles)
        
        # --- Subscriber ---
        self.sub_odom = self.create_subscription(
            Odometry, '/robot_noisy', self.odom_callback, 10)
        self.sub_landmarks = self.create_subscription(
            PointCloud2, '/landmarks_observed', self.landmarks_callback, 10)

        # --- Publisher ---
        self.part_pub = self.create_publisher(PoseArray, '/particle_cloud', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/estimated_pose', 10)
        
        # Timer für Visualisierung und Schätzung
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.get_logger().info("MCL Node gestartet - Warte auf Konvergenz...")

        # Pfad zu den Landmarken
        csv_path = os.path.expanduser('~/AMR/ex02_v1/src/fake_robot/data/landmarks.csv')
        try:
            data = np.genfromtxt(csv_path, delimiter=',')
            self.landmarks = {int(row[0]): (row[1], row[2]) for row in data}
            self.get_logger().info(f"{len(self.landmarks)} Landmarken geladen.")
        except Exception as e:
            self.get_logger().error(f"CSV Fehler: {e}")

    def init_particles(self, num):
        """ Task A1: Partikel global über den Raum verteilen """
        particles = []
        for i in range(num):
            particles.append({
                'x': random.uniform(-5.0, 5.0),
                'y': random.uniform(-5.0, 5.0),
                'theta': random.uniform(-math.pi, math.pi),
                'w': 1.0 / num
            })
        return particles

    def odom_callback(self, msg):
        """ Task A2: Motion Update (Relativbewegung) """
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        curr_theta = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))

        if self.last_odom_pose is None:
            self.last_odom_pose = (curr_x, curr_y, curr_theta)
            return

        prev_x, prev_y, prev_theta = self.last_odom_pose
        
        # 1. Delta in Weltkoordinaten berechnen
        diff_x = curr_x - prev_x
        diff_y = curr_y - prev_y
        dtheta = math.atan2(math.sin(curr_theta - prev_theta), math.cos(curr_theta - prev_theta))

        # 2. Delta in lokales System des Roboters (basierend auf der alten Ausrichtung) drehen
        dx_local = diff_x * math.cos(prev_theta) + diff_y * math.sin(prev_theta)
        dy_local = -diff_x * math.sin(prev_theta) + diff_y * math.cos(prev_theta)

        # 3. Auf Partikel anwenden
        for p in self.particles:
            # Task A2.2: Rauschen hinzufügen (Unsicherheit der Bewegung)
            noise_x = random.gauss(0, 0.05)
            noise_y = random.gauss(0, 0.05)
            noise_theta = random.gauss(0, 0.02)

            # Lokale Bewegung mit dem individuellen Winkel des Partikels in die Welt zurückdrehen
            world_dx = dx_local * math.cos(p['theta']) - dy_local * math.sin(p['theta'])
            world_dy = dx_local * math.sin(p['theta']) + dy_local * math.cos(p['theta'])

            p['x'] += world_dx + noise_x
            p['y'] += world_dy + noise_y
            p['theta'] += dtheta + noise_theta
            p['theta'] = math.atan2(math.sin(p['theta']), math.cos(p['theta']))
            
        self.last_odom_pose = (curr_x, curr_y, curr_theta)

    def landmarks_callback(self, msg):
        observed_landmarks = []
        point_step = msg.point_step 
        
        for i in range(msg.width * msg.height):
            offset = i * point_step
            try:
                # X und Y bleiben Floats
                lx = struct.unpack_from('f', msg.data, offset + 0)[0]
                ly = struct.unpack_from('f', msg.data, offset + 4)[0]
                
                # ID als uint32_t lesen (Format 'I')
                # Teste Offset 12, falls das nicht klappt, Offset 16
                lid = struct.unpack_from('I', msg.data, offset + 12)[0]
                
                if i == 0:
                    self.get_logger().info(f"ID-Check: x={lx:.2f}, y={ly:.2f}, ID={lid}")
                
                observed_landmarks.append((lx, ly, lid))
            except Exception as e:
                continue

        if observed_landmarks:
            self.update_weights(observed_landmarks)

    def update_weights(self, observed):
        """ Task A3: Measurement Update """
        sigma = 0.5 
        for p in self.particles:
            likelihood = 1.0
            for lx, ly, lid in observed:
                if lid in self.landmarks:
                    gt_x, gt_y = self.landmarks[lid]
                    
                    dx = gt_x - p['x']
                    dy = gt_y - p['y']
                    
                    tx = dx * math.cos(p['theta']) + dy * math.sin(p['theta'])
                    ty = -dx * math.sin(p['theta']) + dy * math.cos(p['theta'])
                    
                    dist_sq = (lx - tx)**2 + (ly - ty)**2
                    likelihood *= math.exp(-dist_sq / (2 * sigma**2))
            
            p['w'] = max(likelihood, 1e-300)

        total_w = sum(p['w'] for p in self.particles)
        if total_w > 0:
            for p in self.particles:
                p['w'] /= total_w
            self.resample()
            

    def resample(self):
        """ Task A4: Importance Resampling """
        weights = [p['w'] for p in self.particles]
        indices = np.random.choice(len(self.particles), size=self.num_particles, p=weights)
        
        new_particles = []
        for idx in indices:
            p = self.particles[idx]
            new_particles.append({
                'x': p['x'], 'y': p['y'], 'theta': p['theta'], 
                'w': 1.0 / self.num_particles
            })
        self.particles = new_particles

    def timer_callback(self):
        """ Task A5 & A6: Schätzung berechnen und visualisieren """
        self.publish_particles() # Task A6
        
        # Task A5: Schwerpunkt der Wolke berechnen (Estimated Pose)
        if self.particles:
            total_w = sum(p['w'] for p in self.particles)
            if total_w > 0:
                mean_x = sum(p['x'] * p['w'] for p in self.particles) / total_w
                mean_y = sum(p['y'] * p['w'] for p in self.particles) / total_w
                
                # Mittlerer Winkel über Vektor-Addition (verhindert Fehler bei -PI/PI Sprung)
                s_sum = sum(math.sin(p['theta']) * p['w'] for p in self.particles)
                c_sum = sum(math.cos(p['theta']) * p['w'] for p in self.particles)
                mean_theta = math.atan2(s_sum, c_sum)
                
                self.publish_estimate(mean_x, mean_y, mean_theta)

    def publish_estimate(self, x, y, theta):
        """ Task A5: Die geschätzte Pose als PoseWithCovarianceStamped senden """
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(theta / 2.0)
        self.pose_pub.publish(msg)

    def publish_particles(self):
        """ Task A6: Die gesamte Wolke als PoseArray für RViz senden """
        msg = PoseArray()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for p in self.particles:
            pose = Pose()
            pose.position.x = float(p['x'])
            pose.position.y = float(p['y'])
            pose.orientation.z = math.sin(p['theta'] / 2.0)
            pose.orientation.w = math.cos(p['theta'] / 2.0)
            msg.poses.append(pose)
        self.part_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MCLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()