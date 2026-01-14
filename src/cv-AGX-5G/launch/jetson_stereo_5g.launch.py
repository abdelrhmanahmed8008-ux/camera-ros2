#!/usr/bin/env python3
"""
Launch file for Jetson Thor AGX with 5G network integration
Optimized for low latency and high performance
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    """
    Launch configuration for Jetson stereo camera system with 5G
    
    Components:
    1. Four stereo camera stitching nodes (hardware-accelerated)
    2. Combined view display node
    3. 5G network bridge for remote streaming
    4. Optional performance monitoring
    """
    
    # ========== Launch Arguments ==========
    
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
    
    use_compressed_arg = DeclareLaunchArgument(
        'use_compressed',
        default_value='true',
        description='Use compressed image topics for 5G streaming'
    )
    
    jpeg_quality_arg = DeclareLaunchArgument(
        'jpeg_quality',
        default_value='85',
        description='JPEG compression quality (1-100)'
    )
    
    enable_5g_arg = DeclareLaunchArgument(
        'enable_5g',
        default_value='true',
        description='Enable 5G network bridge'
    )
    
    remote_host_arg = DeclareLaunchArgument(
        'remote_host',
        default_value='0.0.0.0',
        description='Remote host for 5G streaming (0.0.0.0 = all interfaces)'
    )
    
    stream_port_arg = DeclareLaunchArgument(
        'stream_port',
        default_value='5000',
        description='Port for video streaming'
    )
    
    control_port_arg = DeclareLaunchArgument(
        'control_port',
        default_value='5001',
        description='Port for control commands'
    )
    
    enable_display_arg = DeclareLaunchArgument(
        'enable_display',
        default_value='true',
        description='Enable local display (combined view)'
    )
    
    enable_monitoring_arg = DeclareLaunchArgument(
        'enable_monitoring',
        default_value='false',
        description='Enable system performance monitoring'
    )
    
    max_cpu_usage_arg = DeclareLaunchArgument(
        'max_cpu_usage',
        default_value='80',
        description='Maximum CPU usage percentage per core'
    )
    
    # Get launch configurations
    n_features = LaunchConfiguration('n_features')
    match_threshold = LaunchConfiguration('match_threshold')
    use_compressed = LaunchConfiguration('use_compressed')
    jpeg_quality = LaunchConfiguration('jpeg_quality')
    enable_5g = LaunchConfiguration('enable_5g')
    enable_display = LaunchConfiguration('enable_display')
    enable_monitoring = LaunchConfiguration('enable_monitoring')
    remote_host = LaunchConfiguration('remote_host')
    stream_port = LaunchConfiguration('stream_port')
    control_port = LaunchConfiguration('control_port')
    
    # ========== Camera Nodes (Jetson-optimized) ==========
    
    # Camera 0 (front-left)
    camera_0_node = Node(
        package='camera',
        executable='jetson_stereo_stitch',
        name='stereo_stitch_camera_0',
        namespace='stereo_cameras',
        output='log',
        parameters=[{
            'camera_index': 0,
            'n_features': n_features,
            'match_threshold': match_threshold,
            'use_compressed': use_compressed,
            'jpeg_quality': jpeg_quality
        }],
        emulate_tty=True,
    )
    
    # Camera 2 (front-right)
    camera_2_node = Node(
        package='camera',
        executable='jetson_stereo_stitch',
        name='stereo_stitch_camera_2',
        namespace='stereo_cameras',
        output='log',
        parameters=[{
            'camera_index': 2,
            'n_features': n_features,
            'match_threshold': match_threshold,
            'use_compressed': use_compressed,
            'jpeg_quality': jpeg_quality
        }],
        emulate_tty=True,
    )
    
    # Camera 4 (rear-left)
    camera_4_node = Node(
        package='camera',
        executable='jetson_stereo_stitch',
        name='stereo_stitch_camera_4',
        namespace='stereo_cameras',
        output='log',
        parameters=[{
            'camera_index': 4,
            'n_features': n_features,
            'match_threshold': match_threshold,
            'use_compressed': use_compressed,
            'jpeg_quality': jpeg_quality
        }],
        emulate_tty=True,
    )
    
    # Camera 6 (rear-right)
    camera_6_node = Node(
        package='camera',
        executable='jetson_stereo_stitch',
        name='stereo_stitch_camera_6',
        namespace='stereo_cameras',
        output='log',
        parameters=[{
            'camera_index': 6,
            'n_features': n_features,
            'match_threshold': match_threshold,
            'use_compressed': use_compressed,
            'jpeg_quality': jpeg_quality
        }],
        emulate_tty=True,
    )
    
    # ========== Combined Display Node ==========
    
    combined_view_node = Node(
        package='camera',
        executable='combined_view',
        name='combined_camera_view',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(enable_display)
    )
    
    # ========== 5G Network Bridge ==========
    
    fiveG_bridge_node = Node(
        package='camera',
        executable='fiveG_network_bridge',
        name='fiveG_network_bridge',
        output='screen',
        parameters=[{
            'remote_host': remote_host,
            'stream_port': stream_port,
            'control_port': control_port,
            'enable_webrtc': False,
            'adaptive_quality': True,
            'num_cameras': 4
        }],
        emulate_tty=True,
        condition=IfCondition(enable_5g)
    )
    
    # ========== Performance Monitoring (Optional) ==========
    
    # Jetson stats monitoring
    jetson_stats_node = ExecuteProcess(
        cmd=['jtop', '--loop'],
        output='screen',
        condition=IfCondition(enable_monitoring)
    )
    
    # CPU frequency governor setup for performance
    cpu_performance_setup = ExecuteProcess(
        cmd=['bash', '-c', 
             'echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'],
        output='log',
        shell=True
    )
    
    # Enable Jetson power mode (MAXN)
    jetson_power_mode = ExecuteProcess(
        cmd=['sudo', 'nvpmodel', '-m', '0'],  # Mode 0 = MAXN
        output='log'
    )
    
    # ========== Launch Description ==========
    
    return LaunchDescription([
        # Arguments
        n_features_arg,
        match_threshold_arg,
        use_compressed_arg,
        jpeg_quality_arg,
        enable_5g_arg,
        remote_host_arg,
        stream_port_arg,
        control_port_arg,
        enable_display_arg,
        enable_monitoring_arg,
        max_cpu_usage_arg,
        
        # System configuration
        jetson_power_mode,
        cpu_performance_setup,
        
        # Camera nodes
        camera_0_node,
        camera_2_node,
        camera_4_node,
        camera_6_node,
        
        # Display
        combined_view_node,
        
        # 5G Network
        TimerAction(
            period=2.0,  # Wait 2 seconds for cameras to initialize
            actions=[fiveG_bridge_node]
        ),
        
        # Monitoring (optional)
        jetson_stats_node,
    ])