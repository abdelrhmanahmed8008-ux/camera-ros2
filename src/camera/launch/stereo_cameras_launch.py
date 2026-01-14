from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """
    Launch 4 stereo camera nodes + combined view in a SINGLE window (2x2 grid)
    
    This launch file:
    1. Starts all 4 camera stitching nodes (they run in background)
    2. Starts a combined view node that displays all 4 feeds in one window
    
    The individual camera windows are NOT shown - only the combined view!
    """
    
    # Declare launch arguments
    n_features_arg = DeclareLaunchArgument(
        'n_features',
        default_value='3000',
        description='Number of ORB features to detect'
    )
    
    match_threshold_arg = DeclareLaunchArgument(
        'match_threshold',
        default_value='0.75',
        description='Feature matching threshold (0.0-1.0)'
    )
    
    # Get launch configurations
    n_features = LaunchConfiguration('n_features')
    match_threshold = LaunchConfiguration('match_threshold')
    
    # Define the 4 camera nodes (running in background, no individual windows shown)
    camera_0_node = Node(
        package='camera',
        executable='stereo_stitch_0',
        name='stereo_stitch_camera_0',
        namespace='stereo_cameras',
        output='log',  # Changed from 'screen' to 'log' to reduce terminal spam
        emulate_tty=True,
    )
    
    camera_2_node = Node(
        package='camera',
        executable='stereo_stitch_2',
        name='stereo_stitch_camera_2',
        namespace='stereo_cameras',
        output='log',
        emulate_tty=True,
    )
    
    camera_4_node = Node(
        package='camera',
        executable='stereo_stitch_4',
        name='stereo_stitch_camera_4',
        namespace='stereo_cameras',
        output='log',
        emulate_tty=True,
    )
    
    camera_6_node = Node(
        package='camera',
        executable='stereo_stitch_6',
        name='stereo_stitch_camera_6',
        namespace='stereo_cameras',
        output='log',
        emulate_tty=True,
    )
    
    # Combined view node - displays all 4 cameras in one window
    # Starts immediately with cameras
    combined_view_node = Node(
        package='camera',
        executable='combined_view',
        name='combined_camera_view',
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        n_features_arg,
        match_threshold_arg,
        camera_0_node,
        camera_2_node,
        camera_4_node,
        camera_6_node,
        combined_view_node,
    ])