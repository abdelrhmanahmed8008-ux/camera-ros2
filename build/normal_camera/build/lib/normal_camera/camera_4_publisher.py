import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_4_publisher')
        
        # Declare parameters
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('topic_name', 'camera/image')
        self.declare_parameter('frame_rate', 30)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('show_preview', True)
        
        # Get parameters
        camera_index = self.get_parameter('camera_index').value
        topic_name = self.get_parameter('topic_name').value
        frame_rate = self.get_parameter('frame_rate').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        self.show_preview = self.get_parameter('show_preview').value
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, frame_rate)
        
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera {camera_index}')
            return
        
        # Create preview window if enabled
      #  self.window_name = f'Camera {camera_index}'
        #if self.show_preview:
          # cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
          #  cv2.resizeWindow(self.window_name, width, height)
        
        # Create publisher
        self.publisher = self.create_publisher(Image, topic_name, 10)
        self.bridge = CvBridge()
        
        # Create timer
        self.timer = self.create_timer(1.0 / frame_rate, self.timer_callback)
        
        self.get_logger().info(
            f'Camera publisher started: camera_index={camera_index}, '
            f'topic={topic_name}, fps={frame_rate}, preview={self.show_preview}'
        )
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return
        
        # Show preview window if enabled
        #if self.show_preview:
            #cv2.imshow(self.window_name, frame)
           # cv2.waitKey(1)
        
        # Convert to ROS message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        
        self.publisher.publish(msg)
    
    def destroy_node(self):
        self.cap.release()
        if self.show_preview:
            cv2.destroyWindow(self.window_name)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()