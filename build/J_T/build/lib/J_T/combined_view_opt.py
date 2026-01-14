import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class CombinedViewOptimized(Node):
    """GPU-accelerated combined view"""
    
    def __init__(self):
        super().__init__('combined_view')
        
        self.bridge = CvBridge()
        
        # Image storage
        self.images = {0: None, 2: None, 4: None, 6: None}
        self.last_update = {0: 0.0, 2: 0.0, 4: 0.0, 6: 0.0}
        self.frame_counts = {0: 0, 2: 0, 4: 0, 6: 0}
        
        # Subscribe to cameras
        self.subscribers = {}
        for cam_idx in [0, 2, 4, 6]:
            self.subscribers[cam_idx] = self.create_subscription(
                Image,
                f'/stereo_cameras/camera_{cam_idx}/panorama',
                lambda msg, idx=cam_idx: self.image_callback(msg, idx),
                1
            )
        
        # Display timer
        self.timer = self.create_timer(1.0 / 30.0, self.display_callback)
        
        # Performance
        self.compose_times = []
        self.last_stats = time.time()
        
        # Create window
        cv2.namedWindow('Combined View 2x2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Combined View 2x2', 2048, 1440)
        
        self.get_logger().info('Combined View Optimized initialized')
    
    def image_callback(self, msg: Image, camera_idx: int):
        """Receive panorama"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.images[camera_idx] = cv_image
            self.last_update[camera_idx] = time.time()
            self.frame_counts[camera_idx] += 1
        except Exception as e:
            self.get_logger().error(f'Camera {camera_idx} error: {e}')
    
    def display_callback(self):
        """Compose and display grid"""
        t_start = time.time()
        
        cell_w, cell_h = 1024, 720
        grid = np.zeros((cell_h * 2, cell_w * 2, 3), dtype=np.uint8)
        
        positions = {0: (0, 0), 2: (0, 1), 4: (1, 0), 6: (1, 1)}
        current_time = time.time()
        
        for cam_idx, (row, col) in positions.items():
            y, x = row * cell_h, col * cell_w
            
            is_stale = (current_time - self.last_update[cam_idx]) > 2.0
            
            if self.images[cam_idx] is not None and not is_stale:
                img = cv2.resize(self.images[cam_idx], (cell_w, cell_h))
                grid[y:y+cell_h, x:x+cell_w] = img
                
                label = f'Camera {cam_idx} - {self.frame_counts[cam_idx]} frames'
                cv2.putText(grid, label, (x+10, y+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                label = f'Camera {cam_idx} - Waiting'
                cv2.putText(grid, label, (x+cell_w//3, y+cell_h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # Add grid lines
        cv2.line(grid, (cell_w, 0), (cell_w, cell_h*2), (100, 100, 100), 2)
        cv2.line(grid, (0, cell_h), (cell_w*2, cell_h), (100, 100, 100), 2)
        
        cv2.imshow('Combined View 2x2', grid)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            rclpy.shutdown()
        elif key == ord('s'):
            filename = f'grid_{int(time.time())}.jpg'
            cv2.imwrite(filename, grid)
            self.get_logger().info(f'Saved: {filename}')
        
        t_end = time.time()
        self.compose_times.append(t_end - t_start)
        
        if t_end - self.last_stats > 5.0:
            avg = np.mean(self.compose_times) * 1000
            fps = 1.0 / np.mean(self.compose_times)
            self.get_logger().info(f'Grid: {fps:.1f} FPS, {avg:.1f}ms')
            self.compose_times.clear()
            self.last_stats = t_end
    
    def cleanup(self):
        """Cleanup"""
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CombinedViewOptimized()
    
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