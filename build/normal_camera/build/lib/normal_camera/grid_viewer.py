import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from functools import partial

class GridViewer(Node):
    def __init__(self):
        super().__init__('grid_viewer')
        
        # Declare parameters
        self.declare_parameter('grid_rows', 2)
        self.declare_parameter('grid_cols', 2)
        self.declare_parameter('cell_width', 640)
        self.declare_parameter('cell_height', 480)
        self.declare_parameter('window_name', 'Camera Grid Viewer')
        self.declare_parameter('camera_topics', [
            '/camera1/image',
            '/camera2/image',
            '/camera3/image',
            '/camera4/image'
        ])
        
        # Get parameters
        self.grid_rows = self.get_parameter('grid_rows').value
        self.grid_cols = self.get_parameter('grid_cols').value
        self.cell_width = self.get_parameter('cell_width').value
        self.cell_height = self.get_parameter('cell_height').value
        self.window_name = self.get_parameter('window_name').value
        topics = self.get_parameter('camera_topics').value
        
        # Initialize
        self.bridge = CvBridge()
        total_cameras = self.grid_rows * self.grid_cols
        self.frames = [None] * total_cameras
        self.frame_count = [0] * total_cameras  # Track frames received
        self.subscribers = []
        
        # Create subscribers with proper callback binding
        for i in range(min(total_cameras, len(topics))):
            callback = partial(self.image_callback, index=i)
            sub = self.create_subscription(
                Image,
                topics[i],
                callback,
                10
            )
            self.subscribers.append(sub)
            self.get_logger().info(f'Subscribed to {topics[i]} for position {i}')
        
        # Create display timer (30 Hz)
        self.timer = self.create_timer(0.033, self.display_grid)
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.get_logger().info(
            f'Grid viewer started: {self.grid_rows}x{self.grid_cols} grid'
        )
        
        # Status timer to log frame reception
        self.status_timer = self.create_timer(2.0, self.print_status)
    
    def image_callback(self, msg, index):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frames[index] = cv2.resize(
                cv_image, 
                (self.cell_width, self.cell_height)
            )
            self.frame_count[index] += 1
            
            # Log first frame received
            if self.frame_count[index] == 1:
                self.get_logger().info(f'First frame received for camera {index + 1}')
        except Exception as e:
            self.get_logger().error(f'Error processing image for camera {index}: {e}')
    
    def print_status(self):
        """Print status of received frames"""
        status = ', '.join([f'Cam{i+1}:{self.frame_count[i]}' for i in range(len(self.frame_count))])
        self.get_logger().info(f'Frame counts: {status}')
    
    def display_grid(self):
        grid_width = self.grid_cols * self.cell_width
        grid_height = self.grid_rows * self.cell_height
        grid = np.full((grid_height, grid_width, 3), 50, dtype=np.uint8)
        
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                idx = i * self.grid_cols + j
                x = j * self.cell_width
                y = i * self.cell_height
                
                if idx < len(self.frames) and self.frames[idx] is not None:
                    try:
                        grid[y:y+self.cell_height, x:x+self.cell_width] = self.frames[idx]
                        
                        # Map display position to camera index (0,2,4,6)
                        camera_index = idx * 2
                        # Add camera label with frame count
                        label = f'Camera {camera_index} [{self.frame_count[idx]}]'
                        cv2.putText(grid, label, (x + 10, y + 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    except Exception as e:
                        self.get_logger().error(f'Error displaying camera {idx}: {e}')
                else:
                    # Draw placeholder
                    camera_index = idx * 2
                    label = f'Waiting - Camera {camera_index}'
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = x + (self.cell_width - text_size[0]) // 2
                    text_y = y + (self.cell_height + text_size[1]) // 2
                    cv2.putText(grid, label, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        cv2.imshow(self.window_name, grid)
        cv2.waitKey(1)
    
    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GridViewer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()