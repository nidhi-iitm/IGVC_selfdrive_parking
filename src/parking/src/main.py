#!/usr/bin/python3

import rclpy
from PIL import Image
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import torch
import pathlib
from ultralytics import YOLO
import math
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
import struct
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import tf_transformations

from tf_transformations import euler_from_quaternion

pi = 3.14
tilt_angle = 0.296706
camera_common_radius = 0.03
camera_subtended_angle = 1.91986
caster_pos_y = 0.45
cone_radius = 0.1
rot_angle = -1*(-pi/2 + camera_subtended_angle/2)

def cal_rel_cam(v, tilt_angle, rot_angle):
    rx = np.asarray(
            [[1, 0, 0],
            [0, np.cos(0), -np.sin(0)],
            [0, np.sin(0), np.cos(0)]]
        )
        
    ry = np.asarray(
        [[np.cos(tilt_angle), 0 ,np.sin(tilt_angle)],
        [0, 1, 0],
        [-np.sin(tilt_angle), 0, np.cos(tilt_angle)]]
    )

    rz = np.asarray(
        [[np.cos(rot_angle), -np.sin(rot_angle), 0],
        [np.sin(rot_angle), np.cos(rot_angle), 0],
        [0, 0, 1]]
    )
    
    y, z, x = v[0], v[1], v[2]
    v = np.dot(rx, np.array([x, y, z]))
    v = np.dot(ry, v)
    v = np.dot(rz, v)

    return v

class ConeDetectionNode(Node):
    def __init__(self):
        super().__init__('cone_detection')

        self.camera1_orientation = -rot_angle-pi/2
        self.camera2_orientation = -rot_angle

        self.rgb_camera1 = np.empty( shape=(0, 0) )
        self.rgb_camera2 = np.empty( shape=(0, 0) )

        self.new_centroids = {}
       
        self.model = YOLO("/home/nidhi/Abhiyaan/parallel_parking/src/parking/src/best.pt")
        self.classNames = ["cone"]
        
        self.subscription_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        self.subscription_cam1info = self.create_subscription(
            CameraInfo,
            # '/zed/zed_node/depth/depth_registered',
            # '/zed/zed_node/depth/camera_info',
            '/depth_camera1/camera_info',
            lambda msg: self.caminfo_callback(msg, camera_name = "camera1"),
            10)
        
        self.subscription_camera1_depth = self.create_subscription(
            Image,
            # '/zed/zed_node/depth/depth_registered',
            '/depth_camera1/depth/image_raw',
            lambda msg: self.depth_callback(msg, camera_name = "camera1"),
            10)
        
        
        self.subscription_camera1 = self.create_subscription(
            Image,
            # '/zed/zed_node/depth/depth_registered',
            # '/zed/zed_node/rgb/image_rect_color',
            '/camera1/image_raw',
            lambda msg: self.image_callback(msg, camera_name = "camera1"),
            10)
        
        self.subscription_cam1info = self.create_subscription(
            CameraInfo,
            # '/zed/zed_node/depth/depth_registered',
            # '/zed/zed_node/depth/camera_info',
            '/depth_short_1_camera/camera_info',
            lambda msg: self.caminfo_callback(msg, camera_name = "camera2"),
            10)
        
        self.subscription_camera1_depth = self.create_subscription(
            Image,
            # '/zed/zed_node/depth/depth_registered',
            '/depth_short_1_camera/depth/image_raw',
            lambda msg: self.depth_callback(msg, camera_name = "camera2"),
            10)
        
        self.subscription_camera1 = self.create_subscription(
            Image,
            # '/zed/zed_node/depth/depth_registered',
            # '/zed/zed_node/rgb/image_rect_color',
            '/short_1_camera/image_raw',
            lambda msg: self.image_callback(msg, camera_name = "camera2"),
            10)

        self.publisher_ = self.create_publisher(
            Image,
            '/cones',
            10)
            
        # self.detect_pub = self.create_publisher(Pose, 'destination_pose',10)
        self.detect_pub = self.create_publisher(Pose, 'destination_pose',10)
        self.detec_cone1_pub = self.create_publisher(Pose, 'mean_destination',10)
        self.pointcloud_ = self.create_publisher(PointCloud2, 'dest_pcl',10)
        self.caminfo1 = None
        self.depth_img1 = None

        self.caminfo2 = None
        self.depth_img2 = None

        self.bridge = CvBridge()
        
    def odom_callback(self, msg):
        self.x_pos = msg.pose.pose.position.x
        self.y_pos = msg.pose.pose.position.y
        self.z_pos = msg.pose.pose.position.z
        orientation = msg.pose.pose.orientation

        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.orientation = yaw
        
    def depth_callback(self, msg, camera_name):
        if camera_name == "camera1":
            self.depth_img1 = self.bridge.imgmsg_to_cv2(msg)
        else:
            self.depth_img2 = self.bridge.imgmsg_to_cv2(msg)
        
    def caminfo_callback(self, msg, camera_name):
        if camera_name == "camera1":
            self.caminfo1 = msg
        else:
            self.caminfo2 = msg

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

    def publish_coordinates(self, camera_name):
        l = list(self.new_centroids.keys())
        if len(l) < 2 or self.dual_camera == False:
            return
        
        l.sort()
        
        v = []

        for i in range(1,3):
            if self.new_centroids[l[-i]][-1] == "camera1":
                v1 = [-1*self.new_centroids[l[-i]][0], -1*self.new_centroids[l[-i]][1], self.new_centroids[l[-i]][2]]
                v.append([cal_rel_cam(v=v1, tilt_angle=tilt_angle, rot_angle=rot_angle), "camera1"])
            else:
                v2 = [-1*self.new_centroids[l[-i]][0], -1*self.new_centroids[l[-i]][1], self.new_centroids[l[-i]][2]]
                v.append([cal_rel_cam(v=v2, tilt_angle=tilt_angle, rot_angle=-1*rot_angle), "camera2"])

        new_v = [[0, 0],
                 [0, 0]]

        for i in [-1, -2]:
            x, y = v[i][0][0], v[i][0][1]

            if v[i][-1] == "camera1":
                x += camera_common_radius*math.sin(camera_subtended_angle/2)
                y += -caster_pos_y/2 + camera_common_radius*math.cos(camera_subtended_angle/2)
            else:
                x += -camera_common_radius*math.sin(camera_subtended_angle/2)
                y += -caster_pos_y/2 + camera_common_radius*math.cos(camera_subtended_angle/2)
            
            new_v[i][0] = self.x_pos + x*math.cos(self.orientation) - y*math.sin(self.orientation)
            new_v[i][1] = self.y_pos + x*math.sin(self.orientation) + y*math.cos(self.orientation)
            
            # ---------------------------------------------------------------------------------------------------
            #Code when we need to publish position of 2 cones on different topics
            
            # d = math.sqrt(np.square(new_v[i][0]-self.x_pos) + np.square(new_v[i][1]-self.y_pos))
            # t = (d+cone_radius)/d

            # new_v[i][0] = (1 - t) * self.x_pos + t * new_v[i][0]
            # new_v[i][1] = (1 - t) * self.y_pos + t * new_v[i][1]


        # msg1 = Pose()
        # msg1.position.x = new_v[0][0]
        # msg1.position.y = new_v[0][1]

        # msg1.position.z = 0.0
        # msg1.orientation.x = 0.0
        # msg1.orientation.y = 0.0
        # msg1.orientation.z = 0.0
        # msg1.orientation.w = 1.0

        # msg2 = Pose()
        # msg2.position.x = new_v[1][0]
        # msg2.position.y = new_v[1][1]

        # msg2.position.z = 0.0
        # msg2.orientation.x = 0.0
        # msg2.orientation.y = 0.0
        # msg2.orientation.z = 0.0
        # msg2.orientation.w = 1.0

        # self.detec_cone1_pub.publish(msg1)
        # self.detec_cone2_pub.publish(msg2)
            
        # ---------------------------------------------------------------------------------------
        
        # Code when we need to publish position of mean of cones
        
        mean_centroid = [0.50*(new_v[0][0]+new_v[1][0]), 0.50*(new_v[0][1]+new_v[1][1])]

        d = math.sqrt(np.square(mean_centroid[0]-self.x_pos) + np.square(mean_centroid[1]-self.y_pos))
        t = (d+cone_radius)/d

        mean_centroid[0] = (1 - t) * self.x_pos + t * mean_centroid[0]
        mean_centroid[1] = (1 - t) * self.y_pos + t * mean_centroid[1]

        self.get_logger().info(f"Mean Centroid: {mean_centroid}")

        msg = Pose()
        msg.position.x = mean_centroid[0]
        msg.position.y = mean_centroid[1]

        msg.position.z = 0.0
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.orientation.w = 1.0

        self.detect_pub.publish(msg)
        pcl_msg = self.create_pointcloud2(np.array([[mean_centroid[0],mean_centroid[1],0.0]]))
        self.publisher_.publish(pcl_msg)
        # -------------------------------------------------------------------------------------------------
        
        return

    def display_image(self):
        if(self.dual_camera == True and self.rgb_camera1.shape[-1] == 3 and self.rgb_camera2.shape[-1] == 3):
            height, width, channels = self.rgb_camera1.shape

            # Define the thickness of the vertical black line
            line_thickness = 10  # Adjust thickness as needed

            # Create a black vertical line with the same height and desired thickness
            # The line is a numpy array filled with zeros (black)
            black_line = np.zeros((height, line_thickness, channels), dtype=np.uint8)

            numpy_horizontal_concat = np.concatenate((self.rgb_camera1, black_line, self.rgb_camera2), axis=1)


            # Create a blank space above the images to place the text
            text_space_height = 50  # Height of the space for text
            text_space = np.zeros((text_space_height, numpy_horizontal_concat.shape[1], channels), dtype=np.uint8)

            # Add text to the text_space
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)  # White text
            thickness = 2

            # Calculate the position of "Image 1" and "Image 2"
            # Text will be centered horizontally in each half
            text_size_image1 = cv.getTextSize("Left Camera", font, font_scale, thickness)[0]
            text_x_image1 = (width - text_size_image1[0]) // 2  # Centering text for Image 1
            text_y = text_space_height // 2 + text_size_image1[1] // 2  # Center vertically in the text space

            text_size_image2 = cv.getTextSize("Right Camera", font, font_scale, thickness)[0]
            text_x_image2 = width + line_thickness + (width - text_size_image2[0]) // 2  # Centering text for Image 2

            # Put the text on the blank space above the images
            cv.putText(text_space, "Left Camera", (text_x_image1, text_y), font, font_scale, color, thickness)
            cv.putText(text_space, "Right Camera", (text_x_image2, text_y), font, font_scale, color, thickness)

            # Concatenate the text space on top of the images
            final_image = np.concatenate((text_space, numpy_horizontal_concat), axis=0)

            cv.imshow("IMAGE", final_image)
            cv.waitKey(1)
        
        return

    def cone_detection(self, rgb, camera_name):
        # Add parameter settings or modifications as needed
        results = self.model(rgb, verbose = False)

        cl_box = []

        if camera_name == "camera1":
            self.new_centroids = {}
            self.dual_camera = False

            camera_factor = self.caminfo1.k[-1]
            camera_cx = self.caminfo1.k[2]
            camera_cy = self.caminfo1.k[5]
            camera_fx = self.caminfo1.k[0]
            camera_fy = self.caminfo1.k[4]

            depth_image = self.depth_img1
        
        else:
            self.dual_camera = True

            camera_factor = self.caminfo2.k[-1]
            camera_cx = self.caminfo2.k[2]
            camera_cy = self.caminfo2.k[5]
            camera_fx = self.caminfo2.k[0]
            camera_fy = self.caminfo2.k[4]

            depth_image = self.depth_img2
        
        for r in results:
            boxes = r.boxes
    
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                cl_box.append(((x1+x2)/2, (y1+y2)/2))
                confidence = math.ceil((box.conf[0]*100))/100
                
                if(confidence < 0.6):
                    continue
                    
                cv.rectangle(rgb, (x1, y1), (x2, y2), (255, 0, 255), 3)

                u = int((x1+x2)/2)
                v = int((y1+y2)/2)

                z = depth_image[v][u] / camera_factor
                x = (u - camera_cx) * z / camera_fx
                y = (v - camera_cy) * z / camera_fy

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100

                # class name
                cls = int(box.cls[0])
                coordinates=[]
                coordinates.append([confidence,x,y,z])

                # cone details
                org = [x1, y1]
                font = cv.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
    
                cv.putText(rgb, self.classNames[cls], org, font, fontScale, color, thickness)
        
                if coordinates:
                    if not coordinates[0]:
                        continue
                    
                    for i in coordinates:
                        confidence, x, y, z = i[0], i[1], i[2], i[3]
                        self.new_centroids[confidence] = [i[1], i[2], i[3], camera_name]

        self.publish_coordinates(camera_name)

        if camera_name == "camera1":
            self.rgb_camera1 = rgb
        else:
            self.rgb_camera2 = rgb

        self.display_image()
        
        return rgb

    def image_callback(self, data, camera_name):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return
        
        if camera_name == "camera1":
            if self.depth_img1 is not None and self.caminfo1 is not None:
                final = self.cone_detection(cv_image, camera_name = camera_name)

                try:
                    self.publisher_.publish(self.bridge.cv2_to_imgmsg(final))
                except CvBridgeError as e:
                    self.get_logger().error(str(e))
        else:
            if self.depth_img2 is not None and self.caminfo2 is not None:
                final = self.cone_detection(cv_image, camera_name = camera_name)

                try:
                    self.publisher_.publish(self.bridge.cv2_to_imgmsg(final))
                except CvBridgeError as e:
                    self.get_logger().error(str(e))

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()