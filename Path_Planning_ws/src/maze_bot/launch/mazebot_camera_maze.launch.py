#Shishir Khanal
#12/24/2014
#Launch file to run maze in Gazebo

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from scripts import GazeboRosPaths

def generate_launch_description():
    package_share_dir = get_package_share_directory("maze_bot")
    urdf_file = os.path.join(package_share_dir, "urdf", "maze_bot.urdf")
    world_file = os.path.join(package_share_dir, "worlds", "maze_1.world")
    
    model_path, plugin_path, media_path = GazeboRosPaths.get_paths()

    env = {
        "GAZEBO_MODEL_PATH": model_path,
        "GAZEBO_PLUGIN_PATH": plugin_path,
        "GAZEBO_RESOURCE_PATH": media_path,
    }
    
    return LaunchDescription([

        ExecuteProcess(
            cmd=["gazebo", "--verbose", world_file, "-s", "libgazebo_ros_factory.so",],
            output="screen",
            additional_env=env,
        ),

        Node(
            package="gazebo_ros",
            node_executable="spawn_entity.py",
            arguments=["-entity", "maze_bot", "-b", "-file", urdf_file]),

        Node(
            package="robot_state_publisher",
            node_executable="robot_state_publisher",
            output="screen",
            arguments=[urdf_file]),

    ])
