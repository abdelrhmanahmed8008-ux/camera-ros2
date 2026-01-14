#!/usr/bin/env python3
"""
5G Network Integration Node for Jetson Thor AGX
Provides remote streaming, control, and monitoring over 5G
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
import json
import socket
import threading
import time
from collections import deque
import cv2
from cv_bridge import CvBridge

# For WebRTC streaming (optional)
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from av import VideoFrame
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("Warning: WebRTC not available. Install aiortc for WebRTC support")


class NetworkMetrics:
    """Track network performance metrics"""
    
    def __init__(self, window_size=100):
        self.latency_buffer = deque(maxlen=window_size)
        self.bandwidth_buffer = deque(maxlen=window_size)
        self.packet_loss = 0
        self.total_packets = 0
        self.last_update = time.time()
        
    def update_latency(self, latency_ms):
        self.latency_buffer.append(latency_ms)
        
    def update_bandwidth(self, bytes_sent, duration):
        if duration > 0:
            bandwidth_mbps = (bytes_sent * 8) / (duration * 1e6)
            self.bandwidth_buffer.append(bandwidth_mbps)
    
    def update_packet_loss(self, received, total):
        self.total_packets = total
        self.packet_loss = max(0, total - received)
    
    def get_avg_latency(self):
        if not self.latency_buffer:
            return 0
        return sum(self.latency_buffer) / len(self.latency_buffer)
    
    def get_avg_bandwidth(self):
        if not self.bandwidth_buffer:
            return 0
        return sum(self.bandwidth_buffer) / len(self.bandwidth_buffer)
    
    def get_packet_loss_rate(self):
        if self.total_packets == 0:
            return 0
        return (self.packet_loss / self.total_packets) * 100
    
    def get_summary(self):
        return {
            'avg_latency_ms': round(self.get_avg_latency(), 2),
            'avg_bandwidth_mbps': round(self.get_avg_bandwidth(), 2),
            'packet_loss_rate': round(self.get_packet_loss_rate(), 2),
            'total_packets': self.total_packets
        }


class AdaptiveQualityController:
    """Adaptive quality control based on network conditions"""
    
    def __init__(self):
        self.quality_levels = {
            'high': {'jpeg_quality': 95, 'resolution_scale': 1.0, 'fps': 30},
            'medium': {'jpeg_quality': 80, 'resolution_scale': 0.75, 'fps': 20},
            'low': {'jpeg_quality': 60, 'resolution_scale': 0.5, 'fps': 15},
            'minimal': {'jpeg_quality': 40, 'resolution_scale': 0.25, 'fps': 10}
        }
        
        self.current_level = 'high'
        self.last_adjustment = time.time()
        self.adjustment_cooldown = 2.0  # seconds
    
    def adjust_quality(self, metrics):
        """Adjust quality based on network metrics"""
        current_time = time.time()
        
        # Don't adjust too frequently
        if current_time - self.last_adjustment < self.adjustment_cooldown:
            return self.current_level
        
        latency = metrics.get_avg_latency()
        bandwidth = metrics.get_avg_bandwidth()
        packet_loss = metrics.get_packet_loss_rate()
        
        # Decision logic
        if latency > 200 or packet_loss > 5 or bandwidth < 5:
            new_level = 'minimal'
        elif latency > 100 or packet_loss > 2 or bandwidth < 15:
            new_level = 'low'
        elif latency > 50 or packet_loss > 1 or bandwidth < 30:
            new_level = 'medium'
        else:
            new_level = 'high'
        
        if new_level != self.current_level:
            self.current_level = new_level
            self.last_adjustment = current_time
            print(f"Quality adjusted to: {new_level} (latency: {latency:.1f}ms, bandwidth: {bandwidth:.1f}Mbps)")
        
        return self.current_level
    
    def get_settings(self):
        return self.quality_levels[self.current_level]


class FiveGNetworkBridge(Node):
    """5G Network bridge for remote streaming and control"""
    
    def __init__(self):
        super().__init__('fiveG_network_bridge')
        
        # Declare parameters
        self.declare_parameter('remote_host', '0.0.0.0')  # Listen on all interfaces
        self.declare_parameter('stream_port', 5000)
        self.declare_parameter('control_port', 5001)
        self.declare_parameter('enable_webrtc', False)
        self.declare_parameter('adaptive_quality', True)
        self.declare_parameter('num_cameras', 4)
        
        # Get parameters
        self.remote_host = self.get_parameter('remote_host').value
        self.stream_port = self.get_parameter('stream_port').value
        self.control_port = self.get_parameter('control_port').value
        self.enable_webrtc = self.get_parameter('enable_webrtc').value
        self.adaptive_quality = self.get_parameter('adaptive_quality').value
        self.num_cameras = self.get_parameter('num_cameras').value
        
        # Network components
        self.metrics = NetworkMetrics()
        self.quality_controller = AdaptiveQualityController() if self.adaptive_quality else None
        
        # Bridge for image conversion
        self.bridge = CvBridge()
        
        # Storage for latest images from each camera
        self.latest_images = {}
        self.image_timestamps = {}
        
        # QoS profile optimized for 5G
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Lower latency
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # Only keep latest
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Subscribe to all camera compressed topics
        self.subscribers = {}
        camera_indices = [0, 2, 4, 6] if self.num_cameras == 4 else range(self.num_cameras)
        
        for cam_idx in camera_indices:
            topic = f'/stereo_cameras/camera_{cam_idx}/panorama/compressed'
            self.subscribers[cam_idx] = self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, idx=cam_idx: self.image_callback(msg, idx),
                qos_profile
            )
            self.get_logger().info(f'Subscribed to {topic}')
        
        # Publisher for network status
        self.status_publisher = self.create_publisher(
            String,
            '/fiveG/network_status',
            10
        )
        
        # Start network servers
        self.stream_server_thread = None
        self.control_server_thread = None
        self.clients = []
        self.running = True
        
        self.start_servers()
        
        # Status update timer
        self.status_timer = self.create_timer(1.0, self.publish_status)
        
        self.get_logger().info('✓ 5G Network Bridge initialized')
        self.get_logger().info(f'Stream server: {self.remote_host}:{self.stream_port}')
        self.get_logger().info(f'Control server: {self.remote_host}:{self.control_port}')
    
    def image_callback(self, msg, camera_idx):
        """Receive compressed images from cameras"""
        self.latest_images[camera_idx] = msg.data
        self.image_timestamps[camera_idx] = time.time()
        
        # Update metrics
        self.metrics.total_packets += 1
    
    def start_servers(self):
        """Start streaming and control servers"""
        # Start streaming server
        self.stream_server_thread = threading.Thread(
            target=self.stream_server,
            daemon=True
        )
        self.stream_server_thread.start()
        
        # Start control server
        self.control_server_thread = threading.Thread(
            target=self.control_server,
            daemon=True
        )
        self.control_server_thread.start()
    
    def stream_server(self):
        """TCP server for streaming video data"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.remote_host, self.stream_port))
            server_socket.listen(5)
            self.get_logger().info(f'Stream server listening on port {self.stream_port}')
            
            while self.running:
                server_socket.settimeout(1.0)
                try:
                    client_socket, address = server_socket.accept()
                    self.get_logger().info(f'Client connected: {address}')
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_stream_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        self.get_logger().error(f'Stream server error: {str(e)}')
                    
        finally:
            server_socket.close()
    
    def handle_stream_client(self, client_socket, address):
        """Handle individual streaming client"""
        try:
            while self.running:
                # Get quality settings
                if self.quality_controller:
                    settings = self.quality_controller.get_settings()
                    jpeg_quality = settings['jpeg_quality']
                else:
                    jpeg_quality = 85
                
                # Prepare frame data
                frame_data = {
                    'timestamp': time.time(),
                    'cameras': {}
                }
                
                # Collect all camera images
                for cam_idx, img_data in self.latest_images.items():
                    if img_data:
                        frame_data['cameras'][cam_idx] = {
                            'data': img_data.hex() if isinstance(img_data, bytes) else img_data,
                            'timestamp': self.image_timestamps.get(cam_idx, 0)
                        }
                
                # Serialize and send
                json_data = json.dumps(frame_data)
                message = json_data.encode('utf-8')
                
                # Send length header first
                length = len(message)
                client_socket.sendall(length.to_bytes(4, byteorder='big'))
                
                # Send data
                start_time = time.time()
                client_socket.sendall(message)
                duration = time.time() - start_time
                
                # Update bandwidth metrics
                self.metrics.update_bandwidth(len(message), duration)
                
                # Control frame rate
                time.sleep(1.0 / 30)  # 30 FPS max
                
        except Exception as e:
            self.get_logger().warn(f'Client {address} disconnected: {str(e)}')
        finally:
            client_socket.close()
    
    def control_server(self):
        """TCP server for receiving control commands"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.remote_host, self.control_port))
            server_socket.listen(5)
            self.get_logger().info(f'Control server listening on port {self.control_port}')
            
            while self.running:
                server_socket.settimeout(1.0)
                try:
                    client_socket, address = server_socket.accept()
                    self.get_logger().info(f'Control client connected: {address}')
                    
                    # Handle control commands
                    while self.running:
                        data = client_socket.recv(4096)
                        if not data:
                            break
                        
                        try:
                            command = json.loads(data.decode('utf-8'))
                            self.handle_control_command(command)
                            
                            # Send acknowledgment
                            response = {'status': 'ok', 'command': command.get('type')}
                            client_socket.sendall(json.dumps(response).encode('utf-8'))
                            
                        except Exception as e:
                            self.get_logger().error(f'Control command error: {str(e)}')
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        self.get_logger().error(f'Control server error: {str(e)}')
                        
        finally:
            server_socket.close()
    
    def handle_control_command(self, command):
        """Process control commands from remote client"""
        cmd_type = command.get('type')
        
        if cmd_type == 'set_quality':
            quality = command.get('quality', 'medium')
            if self.quality_controller:
                self.quality_controller.current_level = quality
                self.get_logger().info(f'Quality set to: {quality}')
        
        elif cmd_type == 'request_snapshot':
            camera_id = command.get('camera_id', 0)
            self.get_logger().info(f'Snapshot requested for camera {camera_id}')
            # Could trigger snapshot save here
        
        elif cmd_type == 'ping':
            # Measure round-trip latency
            send_time = command.get('timestamp', 0)
            if send_time > 0:
                latency = (time.time() - send_time) * 1000
                self.metrics.update_latency(latency)
        
        else:
            self.get_logger().warn(f'Unknown command type: {cmd_type}')
    
    def publish_status(self):
        """Publish network status periodically"""
        # Update quality based on metrics
        if self.quality_controller:
            self.quality_controller.adjust_quality(self.metrics)
        
        # Publish status
        status = {
            'timestamp': time.time(),
            'metrics': self.metrics.get_summary(),
            'quality': self.quality_controller.current_level if self.quality_controller else 'fixed',
            'active_cameras': len(self.latest_images),
            'uptime': time.time() - self.get_clock().now().nanoseconds / 1e9
        }
        
        msg = String()
        msg.data = json.dumps(status)
        self.status_publisher.publish(msg)
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.stream_server_thread:
            self.stream_server_thread.join(timeout=2.0)
        if self.control_server_thread:
            self.control_server_thread.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    
    node = FiveGNetworkBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()