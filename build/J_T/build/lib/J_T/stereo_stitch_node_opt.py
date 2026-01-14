import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from CUDA_stereo_stitcher import CUDAStereoStitcher, StitchConfig, split_stereo_frame


class StereoStitchNodeOptimized(Node):
    """Hardware-accelerated stereo stitch node"""
    
    def __init__(self, camera_index: int, homography_path: str):
        super().__init__(f'stereo_stitch_camera_{camera_index}')
        
        self.camera_index = camera_index
        
        # Load homography
        homography = np.load(homography_path)
        self.get_logger().info(f'Loaded homography from {homography_path}')
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera {camera_index}')
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'Camera {camera_index}: {actual_w}x{actual_h}')
        
        # Initialize stitcher
        config = StitchConfig(
            homography=homography,
            output_width=2048,
            output_height=720,
            blend_width=50,
            use_gpu=True
        )
        self.stitcher = CUDAStereoStitcher(config)
        
        # ROS publisher
        self.publisher = self.create_publisher(
            Image,
            f'/stereo_cameras/camera_{camera_index}/panorama',
            10
        )
        self.bridge = CvBridge()
        
        # Performance monitoring
        self.frame_times = []
        self.last_stats = time.time()
        
        # Timer
        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)
        
        self.get_logger().info(f'Stereo node initialized (GPU-accelerated)')
    
    def process_frame(self):
        """Process frame"""
        t_start = time.time()
        
        ret, stereo_frame = self.cap.read()
        if not ret:
            return
        
        left, right = split_stereo_frame(stereo_frame)
        panorama = self.stitcher.stitch(left, right)
        
        try:
            msg = self.bridge.cv2_to_imgmsg(panorama, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = f'camera_{self.camera_index}'
            self.publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Publish failed: {e}')
        
        t_end = time.time()
        self.frame_times.append(t_end - t_start)
        
        if t_end - self.last_stats > 5.0:
            avg_time = np.mean(self.frame_times) * 1000
            fps = 1.0 / np.mean(self.frame_times)
            self.get_logger().info(
                f'Cam {self.camera_index}: {fps:.1f} FPS, {avg_time:.1f}ms'
            )
            self.frame_times.clear()
            self.last_stats = t_end
    
    def cleanup(self):
        """Cleanup"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'stitcher'):
            self.stitcher.cleanup()


def main(args=None):
    rclpy.init(args=args)
    
    if len(sys.argv) < 3:
        print('Usage: stereo_stitch_node_optimized.py <camera_index> <homography_path>')
        return
    
    camera_index = int(sys.argv[1])
    homography_path = sys.argv[2]
    
    node = StereoStitchNodeOptimized(camera_index, homography_path)
    
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