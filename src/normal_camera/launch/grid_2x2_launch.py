from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription

def generate_launch_description():
    return LaunchDescription([
        # Camera 1 Publisher (Index 0)
        Node(
            package='normal_camera',
            executable='camera_0_publisher',
            name='camera_publisher_1',
            parameters=[{
                'camera_index': 0,
                'topic_name': '/camera1/image',
                'frame_rate': 30,
                'width': 640,
                'height': 480,
                'show_preview': True
            }],
            output='screen'
        ),
        
        # Camera 2 Publisher (Index 2)
        Node(
            package='normal_camera',
            executable='camera_2_publisher',
            name='camera_publisher_2',
            parameters=[{
                'camera_index': 2,
                'topic_name': '/camera2/image',
                'frame_rate': 30,
                'width': 640,
                'height': 480,
                'show_preview': True
            }],
            output='screen'
        ),
        
        # Camera 3 Publisher (Index 4)
        Node(
            package='normal_camera',
            executable='camera_4_publisher',
            name='camera_publisher_3',
            parameters=[{
                'camera_index': 4,
                'topic_name': '/camera3/image',
                'frame_rate': 30,
                'width': 640,
                'height': 480,
                'show_preview': True
            }],
            output='screen'
        ),
        
        # Camera 4 Publisher (Index 6)
        Node(
            package='normal_camera',
            executable='camera_6_publisher',
            name='camera_publisher_4',
            parameters=[{
                'camera_index': 6,
                'topic_name': '/camera4/image',
                'frame_rate': 30,
                'width': 640,
                'height': 480,
                'show_preview': True
            }],
            output='screen'
        ),
        
        # Grid Viewer
        Node(
            package='normal_camera',
            executable='grid_viewer',
            name='grid_viewer',
            parameters=[{
                'grid_rows': 2,
                'grid_cols': 2,
                'cell_width': 640,
                'cell_height': 480,
                'window_name': 'Camera Grid 2x2',
                'camera_topics': [
                    '/camera1/image',
                    '/camera2/image',
                    '/camera3/image',
                    '/camera4/image'
                ]
            }],
            output='screen'
        ),
    ])