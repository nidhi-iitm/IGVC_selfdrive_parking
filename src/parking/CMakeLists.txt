cmake_minimum_required(VERSION 3.5)
project(parking)

# Default to C++14 if not set
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)  
find_package(Python3 REQUIRED)


# Add the Python files as executables
install(PROGRAMS
  src/parallel_park.py
  src/posn_publisher_zed.py
  src/pullin_park.py
  src/pullout_park.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
