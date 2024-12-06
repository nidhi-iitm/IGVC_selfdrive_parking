#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist,Pose,PointStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sklearn.cluster import DBSCAN
import numpy as np
import math
from nav2_simple_commander.robot_navigator import BasicNavigator
import numpy as np
from scipy import optimize
from tf2_ros import Buffer, TransformListener
import tf2_ros
import tf2_geometry_msgs
import struct
import tf_transformations

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.subscription=self.create_subscription(Odometry,'/odom',self.odom_callback,10)
        self.path_publisher = self.create_publisher(Path, 'plan', 10)
        self.publisher_ = self.create_publisher(PointCloud2, 'Points', 10)
        self.publisher_map = self.create_publisher(PointCloud2,'points_map', 10)
        self.nav_publisher = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self.first_message_received = False
        self.init_pose1 = PoseStamped()
        self.navigator = BasicNavigator()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.odom_receieved = False


    def get_quaternion_from_euler(self,roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]
    def odom_callback(self,msg):
            self.odom = msg
            self.odom_receieved = True
    def scan_callback(self, msg):
        
        if self.odom_receieved:
            # Convert polar to Cartesian coordinates
            new_coords = self.coord_change(msg.ranges, msg.angle_min, msg.angle_increment)
            
            # Prepare for transformation
            np_coords = np.array(new_coords)
            valid_coords = np_coords[np.isfinite(np_coords).all(axis=1)]
            points_3d = np.hstack((valid_coords, np.zeros((valid_coords.shape[0], 1))))
            
            # Get position and orientation from odometry
            position = np.array([self.odom.pose.pose.position.x, 
                                self.odom.pose.pose.position.y, 
                                self.odom.pose.pose.position.z])
            orientation = self.odom.pose.pose.orientation
            quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
            
            rotation_matrix = tf_transformations.quaternion_matrix(quaternion)[:3, :3]
            final_pose = []
            
            # Transform points to the map frame
            for i in points_3d:
                goal_in_map_frame = np.dot(rotation_matrix, i) + position
                final_pose.append(goal_in_map_frame)
            
            # Convert to pointcloud and publish
            pointcloud_msg = self.create_pointcloud2(final_pose)
            self.publisher_map.publish(pointcloud_msg)
            self.odom_receieved = False

            # Perform clustering if it's the first message received
            if not self.first_message_received:
                fin_coords = np.array([i[:2] for i in final_pose])  
                
                if len(fin_coords) > 0:
                    eps = 0.5
                    min_samples = 5 # Min no of points to be classified into a cluster
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples) 
                    clusters = dbscan.fit_predict(fin_coords)  #Perform DBSCAN clustering on the lidar points
                    unique_clusters = set(clusters)
                    centroids = []

                    for cluster_id in unique_clusters:
                        if cluster_id != -1:  # To filter out unclassified points

                            cluster_points = fin_coords[clusters == cluster_id]  # Use NumPy for boolean indexing instead of list

                            # Calculating the centre of the circle to which the points in the cluster belong to
                            def calc_radius(xc, yc, x, y):
                                return np.sqrt((x - xc)**2 + (y - yc)**2)

                            def residuals_circle(params, x, y):
                                xc, yc = params
                                ri = calc_radius(xc, yc, x, y)
                                return ri - np.mean(ri)

                            x = cluster_points[:, 0]
                            y = cluster_points[:, 1]

                            x0 = np.mean(x)
                            y0 = np.mean(y)

                            center_estimate = optimize.least_squares(residuals_circle, [x0, y0], args=(x, y)) # Using least squares to fit the circle
                            xc, yc = center_estimate.x
                            radius_estimate = np.mean(calc_radius(xc, yc, x, y)) # Calculating radius for debugging purposes.

                            # Output the estimated center and radius
                            centroid = np.array([xc, yc])
                            centroids.append(centroid) 
                            self.get_logger().info(f"Cluster {cluster_id}: Centroid - {centroid}")
                            self.get_logger().info(f"Cluster {cluster_id}: Radius - {radius_estimate}")

                    self.get_logger().info("Centroids found")

                    if centroids:
                        self.new_centroids = {}
                        for i in centroids:
                            x, y = i[0], i[1]
                            self.new_centroids[x**2 + y**2] = i
                        
                        l = list(self.new_centroids.keys())
                        l.sort()
                        
                        centroids = [self.new_centroids[l[0]], self.new_centroids[l[1]]] 
                        self.mean_centroid = 0.50 * centroids[1] + 0.50 * centroids[0]
                        self.final_parallel = 0.70*centroids[1] + 0.30*centroids[0] 
                        self.final_parallel1 = 0.50 * centroids[1] + 0.50 * centroids[0]
                        
                        self.get_logger().info(f"Mean Centroid: {self.mean_centroid}")
                        self.publish_trajectory(self.mean_centroid, centroids, centroids[1])
                    else:
                        self.get_logger().info(f"Centroids {centroids}")

                self.first_message_received = True


    def coord_change(self, ranges, angle_min, angle_increment):
        new_coords = []
        angle = angle_min
        for r in ranges:
            if not math.isfinite(r) or r == 0.0:
                angle += angle_increment
                continue
            #if r <= 5:
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            new_coords.append((x, y))
            angle += angle_increment
            # else:
            #     angle += angle_increment
        return new_coords

    def create_pointcloud2(self, points):
        """
        Create a PointCloud2 message from a list of points (Nx3 array)
        """
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"  # Frame of reference

        # Create the PointCloud2 fields (x, y, z)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Pack the point data into bytes
        point_data = []
        for point in points:
            point_data.append(struct.pack('fff', point[0], point[1], point[2]))

        point_data = b''.join(point_data)

        # Create the PointCloud2 message
        pointcloud = PointCloud2(
            header=header,
            height=1,
            width=len(points),
            fields=fields,
            is_bigendian=False,
            point_step=12,  # 3 floats (x, y, z) * 4 bytes each
            row_step=12 * len(points),
            data=point_data,
            is_dense=True
        )
        return pointcloud

    def publish_trajectory(self, mean_centroid,centroids,final_centroid):
        init_pos = [0.0, 0.0]
        fin_pos = mean_centroid
        th = math.atan2(fin_pos[1] - init_pos[1], fin_pos[0] - init_pos[0])
        self.get_logger().info(f"{th}")
        th = round(th, 2)
        self.get_logger().info(f"3")
        dist = math.dist(init_pos, fin_pos)
        self.get_logger().info(f"4")
        r = (dist / 4) / math.cos(th)

        path_msg = Path()
        path_msg.header.frame_id = "base_link"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        path_poses = []
        self.get_logger().info(f"5")
        inc1 = -0.1
        for t in np.arange(3.14,2*th ,inc1):
            x = init_pos[0] + r + r * math.cos(t)
            y = init_pos[1] + r * math.sin(t)
            z = 0.0

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            roll = 0.0
            pitch = 0.0
            yaw = np.round(math.atan((math.sin(t)-math.sin(t-inc1))/(math.cos(t)-math.cos(t-inc1))))  #to orient along the tangent to the curve
            qx, qy, qz, qw = self.get_quaternion_from_euler(roll, pitch, yaw)
            pose.pose.orientation.w = qw
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            path_msg.poses.append(pose)
            path_poses.append(pose)
        self.get_logger().info(f"6")
        inc2 = 0.1
        for t in np.arange(-3.14+2*th,0,inc2):
            x = fin_pos[0] - r + r * math.cos(t)
            y = fin_pos[1] + r * math.sin(t)
            z = 0.0

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            roll = 0.0
            pitch = 0.0
            yaw = (math.atan((math.sin(t)-math.sin(t-inc2))/(math.cos(t)-math.cos(t-inc2))))
            qx, qy, qz, qw = self.get_quaternion_from_euler(roll, pitch, yaw)
            pose.pose.orientation.w = qw
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            path_msg.poses.append(pose)
            path_poses.append(pose)

        for t in np.arange(np.round(fin_pos[1],4),np.round(self.final_parallel[1],4) ,0.1):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.y = t
            pose.pose.position.x = fin_pos[0]
            pose.pose.position.z = 0.0
            #pose.pose.orientation.w = 1.0
            qx, qy, qz, qw = self.get_quaternion_from_euler(0.0,0.0,1.57) #math.atan((centroids[1][1]-centroids[0][1] )/ (centroids[1][0]-centroids[0][0])))
            pose.pose.orientation.w = qw
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            path_msg.poses.append(pose)
            path_poses.append(pose)         
        self.path_publisher.publish(path_msg)
        self.get_logger().info(f"2")
        self.get_logger().info('Published a new trajectory.')
        self.navigator.goThroughPoses(path_poses)
        while not self.navigator.isTaskComplete():
            print("navigating")


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



