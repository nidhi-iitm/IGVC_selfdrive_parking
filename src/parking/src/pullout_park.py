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
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.stop_posn = 100.0
    def stop_robot(self):
        stop_msg = Twist()  #Passing a zero velocity command to stop the bot
        stop_msg.linear.x = 0.0
        stop_msg.linear.y = 0.0
        stop_msg.linear.z = 0.0
        stop_msg.angular.x = 0.0
        stop_msg.angular.y = 0.0
        stop_msg.angular.z = 0.0
        self.navigator.destroy_node()
        self.pub_cmd_vel.publish(stop_msg)
        self.get_logger().info('Robot stopped!')


    def get_quaternion_from_euler(self,roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]
    def odom_callback(self,msg):
            self.odom = msg
            self.odom_receieved = True
            posn = [self.odom.pose.pose.position.x, 
                                self.odom.pose.pose.position.y]
            tolerance = 0.1  
            if abs(posn[0] - self.stop_posn) < tolerance:
                while True:
                    self.stop_robot()
                    print('stopped')
    def scan_callback(self, msg):
        
        if self.odom_receieved:
            new_coords = self.coord_change(msg.ranges, msg.angle_min, msg.angle_increment)
            
            np_coords = np.array(new_coords)
            valid_coords = np_coords[np.isfinite(np_coords).all(axis=1)]
            points_3d = np.hstack((valid_coords, np.zeros((valid_coords.shape[0], 1))))
            
            position = np.array([self.odom.pose.pose.position.x, 
                                self.odom.pose.pose.position.y, 
                                self.odom.pose.pose.position.z])
            orientation = self.odom.pose.pose.orientation
            quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
            
            rotation_matrix = tf_transformations.quaternion_matrix(quaternion)[:3, :3]
            final_pose = []
            
            for i in points_3d:
                goal_in_map_frame = np.dot(rotation_matrix, i) + position
                final_pose.append(goal_in_map_frame)


            if not self.first_message_received:
                fin_coords = np.array([i[:2] for i in final_pose]) 
                
                if len(fin_coords) > 0:
                    eps = 0.5
                    min_samples = 5
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    clusters = dbscan.fit_predict(fin_coords)
                    unique_clusters = set(clusters)
                    centroids = []

                    for cluster_id in unique_clusters:
                        if cluster_id != -1:
                            cluster_points = fin_coords[clusters == cluster_id]

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

                            center_estimate = optimize.least_squares(residuals_circle, [x0, y0], args=(x, y))
                            xc, yc = center_estimate.x

                            radius_estimate = np.mean(calc_radius(xc, yc, x, y))

                            centroid = np.array([xc, yc])
                            centroids.append(centroid)
                            #self.get_logger().info(f"Cluster {cluster_id}: Centroid - {centroid}")
                            #self.get_logger().info(f"Cluster {cluster_id}: Radius - {radius_estimate}")

                    self.get_logger().info("Centroids found")

                    if centroids:
                        self.new_centroids = {}
                        for i in centroids:
                            x, y = i[0], i[1]
                            self.new_centroids[x**2 + y**2] = i
                        
                        l = list(self.new_centroids.keys())
                        l.sort()
                        
                        centroids = [self.new_centroids[l[0]]] 
                        self.mean_centroid = self.new_centroids[l[0]]
                        self.stop_posn = self.mean_centroid[0] + 0.8 #0.3m in front of cone + bot_length/2 approx
                        #visualising the stop point for debugging purposes
                        pointcloud_msg1 = self.create_pointcloud2(np.array([[self.stop_posn,self.mean_centroid[1],0]]))
                        self.publisher_map.publish(pointcloud_msg1)
                        self.publish_trajectory(self.mean_centroid)

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

    def publish_trajectory(self, mean_centroid):
        init_pos = [self.odom.pose.pose.position.x, 
                                self.odom.pose.pose.position.y]
        fin_pos = mean_centroid
        print('final_pose', fin_pos)
        x1 = fin_pos[0] 
        y1 = fin_pos[1]
        r = fin_pos[1]
        print('radius',y1)
        centre =[init_pos[0]-r, init_pos[1]]
        print('centre of circle', centre)

        path_msg = Path()
        path_msg.header.frame_id = "base_link"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        path_poses = []
  
        for t in np.arange(0, 1.57 ,0.05):
            x = centre[0] + r * math.cos(t)
            y = centre[1] + r * math.sin(t)
            z = 0.0

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            roll = 0.0
            pitch = 0.0
            yaw = np.round(math.atan((math.sin(t)-math.sin(t-0.01))/(math.cos(t)-math.cos(t-0.01))))
            qx, qy, qz, qw = self.get_quaternion_from_euler(roll, pitch, yaw)
            pose.pose.orientation.w = qw
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            path_msg.poses.append(pose)
            path_poses.append(pose)
        for t in np.arange(init_pos[0]-r, x1+0.4 ,-0.1):
            x = t
            y = fin_pos[1]
            z = 0.0
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            roll = 0.0
            pitch = 0.0
            yaw = 3.14
            qx, qy, qz, qw = self.get_quaternion_from_euler(roll, pitch, yaw)
            pose.pose.orientation.w = qw
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            path_msg.poses.append(pose)
            path_poses.append(pose)        
        self.get_logger().info(f"6")
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


