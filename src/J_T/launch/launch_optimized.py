from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():

    
    # Get package share directory
    pkg_dir = os.path.join(os.path.expanduser('~'), 'camera')
    config_dir = os.path.join(pkg_dir, 'config')
    
    # Camera nodes with homography files
    camera_nodes = []
    for cam_idx in [0, 2, 4, 6]:
        homography_file = os.path.join(config_dir, f'homography_camera_{cam_idx}.npy')
        
        node = Node(
            package='J_T',
            executable='stereo_stitch_node',
            name=f'stereo_stitch_camera_{cam_idx}',
            namespace='stereo_cameras',
            output='log',
            arguments=[str(cam_idx), homography_file],
            emulate_tty=True
        )
        camera_nodes.append(node)
    
    # Combined view node
    combined_node = Node(
        package='J_T',
        executable='combined_view',
        name='combined_view',
        output='screen',
        emulate_tty=True
    )
    
    return LaunchDescription(camera_nodes + [combined_node])