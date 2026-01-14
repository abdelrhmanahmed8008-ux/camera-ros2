#!/usr/bin/env python3
"""
Combined view node - displays all 4 camera panoramas in a single 2x2 grid window
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time



class CombinedCameraView(Node):
    def __init__(self):
        super().__init__('combined_camera_view')
        
        self.bridge = CvBridge()
        
        # Storage for the 4 camera images
        self.images = {
            0: None,
            2: None,
            4: None,
            6: None
        }
        
        # Timestamps to detect stale images
        self.last_update = {
            0: time.time(),
            2: time.time(),
            4: time.time(),
            6: time.time()
        }
        
        # Camera statistics
        self.camera_stats = {
            0: {'fps': 0, 'frame_count': 0, 'last_time': time.time()},
            2: {'fps': 0, 'frame_count': 0, 'last_time': time.time()},
            4: {'fps': 0, 'frame_count': 0, 'last_time': time.time()},
            6: {'fps': 0, 'frame_count': 0, 'last_time': time.time()},
        }
        
        # Subscribe to all 4 camera panorama topics with larger queue
        self.subscribers = {}
        for cam_idx in [0, 2, 4, 6]:
            self.subscribers[cam_idx] = self.create_subscription(
                Image,
                f'/stereo_cameras/camera_{cam_idx}/panorama',
                lambda msg, idx=cam_idx: self.image_callback(msg, idx),
                1  # Reduced queue size for lower latency
            )
        
        # Timer for display update (faster refresh)
        self.timer = self.create_timer(0.016, self.display_callback)  # ~60 Hz instead of 30 Hz
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0
        
        self.get_logger().info('Combined Camera View Node initialized')
        self.get_logger().info('Subscribing to 4 camera panorama topics')
        
        # Create the combined window (larger size)
        cv2.namedWindow('Combined Camera View - 2x2 Grid', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Combined Camera View - 2x2 Grid', 2560, 1440)  # Increased from 1920x1080
    
    def image_callback(self, msg, camera_idx):
        """Callback for receiving panorama images"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.images[camera_idx] = cv_image
            self.last_update[camera_idx] = time.time()
            
            # Update FPS for this camera
            stats = self.camera_stats[camera_idx]
            stats['frame_count'] += 1
            
            if stats['frame_count'] >= 10:
                elapsed = time.time() - stats['last_time']
                stats['fps'] = stats['frame_count'] / elapsed
                stats['frame_count'] = 0
                stats['last_time'] = time.time()
                
        except Exception as e:
            self.get_logger().error(f'Failed to convert image from camera {camera_idx}: {str(e)}')
    
    def create_placeholder(self, width, height, camera_idx):
        """Create a placeholder image when camera feed is not available"""
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        text = f"Camera {camera_idx}"
        status = "Waiting for feed..."
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(placeholder, text, (width//2 - 100, height//2 - 20),
                   font, 1.5, (100, 100, 100), 2)
        cv2.putText(placeholder, status, (width//2 - 150, height//2 + 20),
                   font, 0.8, (80, 80, 80), 1)
        
        return placeholder
    
    def resize_image(self, img, target_width, target_height):
        """Resize image to fit in grid cell while maintaining aspect ratio"""
        if img is None:
            return self.create_placeholder(target_width, target_height, 0)
        
        h, w = img.shape[:2]
        
        # Resize to fill the entire cell (no black borders)
        # This will crop if aspect ratios don't match
        scale = max(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop to fit exactly
        y_start = (new_h - target_height) // 2
        x_start = (new_w - target_width) // 2
        cropped = resized[y_start:y_start+target_height, x_start:x_start+target_width]
        
        return cropped
    
    def add_label(self, img, camera_idx, is_stale=False):
        """Add camera label and detailed status to image"""
        
        if is_stale:
            # Stale/disconnected camera
            label = f"Camera {camera_idx}"
            status = "DISCONNECTED"
            color = (0, 0, 255)  # Red
            
            # Draw semi-transparent background
            overlay = img.copy()
            cv2.rectangle(overlay, (5, 5), (350, 75), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            
            cv2.putText(img, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(img, status, (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Active camera - show detailed info
            stats = self.camera_stats[camera_idx]
            fps = stats['fps']
            
            # Camera label and status
            label = f"Camera {camera_idx}"
            info = f"{fps:.1f} FPS | ACTIVE"
            
            color = (0, 255, 0)  # Green
            
            # Draw semi-transparent background for text
            overlay = img.copy()
            cv2.rectangle(overlay, (5, 5), (350, 75), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            
            # Add text with drop shadow for better visibility
            # Shadow
            cv2.putText(img, label, (12, 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(img, info, (12, 62),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            # Main text
            cv2.putText(img, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(img, info, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img
    
    def display_callback(self):
        """Create and display the combined 2x2 grid view"""
        
        # Calculate FPS
        self.fps_counter += 1
        if self.fps_counter >= 30:
            elapsed = time.time() - self.fps_start
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start = time.time()
        
        # Target size for each quadrant - larger display
        grid_width = 2560   # Increased from 1920
        grid_height = 1440  # Increased from 1080
        cell_width = grid_width // 2
        cell_height = grid_height // 2
        
        # Check for stale images (no update in 2 seconds)
        current_time = time.time()
        stale_threshold = 2.0
        
        # Prepare the 4 quadrants
        quadrants = {}
        camera_positions = {
            0: 'top-left',
            2: 'top-right',
            4: 'bottom-left',
            6: 'bottom-right'
        }
        
        for cam_idx in [0, 2, 4, 6]:
            is_stale = (current_time - self.last_update[cam_idx]) > stale_threshold
            
            if self.images[cam_idx] is not None and not is_stale:
                # Resize and add label
                img = self.resize_image(self.images[cam_idx], cell_width, cell_height)
                img = self.add_label(img, cam_idx, is_stale=False)
            else:
                # Create placeholder
                img = self.create_placeholder(cell_width, cell_height, cam_idx)
            
            quadrants[cam_idx] = img
        
        # Create the 2x2 grid
        # Top row: Camera 0 (left) | Camera 2 (right)
        top_row = np.hstack([quadrants[0], quadrants[2]])
        
        # Bottom row: Camera 4 (left) | Camera 6 (right)
        bottom_row = np.hstack([quadrants[4], quadrants[6]])
        
        # Combine rows
        combined = np.vstack([top_row, bottom_row])
        
        # Add overall info text
        info_text = f"Combined View | FPS: {self.current_fps:.1f} | Press 'q' to quit, 's' to save"
        cv2.putText(combined, info_text, (grid_width//2 - 300, grid_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add grid lines
        # Vertical line
        cv2.line(combined, (cell_width, 0), (cell_width, grid_height), 
                (100, 100, 100), 2)
        # Horizontal line
        cv2.line(combined, (0, cell_height), (grid_width, cell_height), 
                (100, 100, 100), 2)
        
        # Display
        cv2.imshow('Combined Camera View - 2x2 Grid', combined)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Quit requested')
            rclpy.shutdown()
        elif key == ord('s'):
            filename = f'combined_view_{int(time.time())}.jpg'
            cv2.imwrite(filename, combined)
            self.get_logger().info(f'Saved: {filename}')
    
    def cleanup(self):
        """Release resources"""
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CombinedCameraView()
    
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