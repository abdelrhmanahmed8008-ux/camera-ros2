#!/usr/bin/env python3
"""
Jetson-optimized stereo stitching node with hardware acceleration
Supports CUDA acceleration, VPI (Vision Programming Interface), and GStreamer
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from collections import deque

# Try to import Jetson-specific libraries
try:
    import jetson.inference
    import jetson.utils
    JETSON_AVAILABLE = True
except ImportError:
    JETSON_AVAILABLE = False
    print("Warning: Jetson libraries not available, using CPU fallback")

# Try to import VPI for hardware acceleration
try:
    import vpi
    VPI_AVAILABLE = True
except ImportError:
    VPI_AVAILABLE = False
    print("Warning: VPI not available, using OpenCV")


def split_stereo_frame(frame):
    """Split stereo camera frame into left and right"""
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]


class JetsonStereoStitcher:
    """Jetson-optimized stereo image stitching with hardware acceleration"""
    
    def __init__(self, n_features=3000, match_threshold=0.7, use_cuda=True):
        self.use_cuda = use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        # Initialize feature detector (CUDA-accelerated if available)
        if self.use_cuda:
            print("✓ CUDA enabled for feature detection")
            self.orb = cv2.cuda.ORB_create(
                nfeatures=n_features,
                scaleFactor=1.2,
                nlevels=8,  # Reduced for speed
                edgeThreshold=15
            )
        else:
            print("✓ Using CPU for feature detection")
            self.orb = cv2.ORB_create(
                nfeatures=n_features,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15
            )
        
        # Feature matcher
        if self.use_cuda:
            self.matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.match_threshold = match_threshold
        
        # CUDA streams for async operations
        if self.use_cuda:
            self.stream = cv2.cuda.Stream()
        
        # VPI context for hardware-accelerated operations
        self.vpi_context = None
        if VPI_AVAILABLE:
            try:
                self.vpi_context = vpi.Backend.CUDA
                print("✓ VPI CUDA backend enabled")
            except:
                print("Warning: VPI CUDA backend not available")
        
        # Homography stabilization
        self.homography_buffer = deque(maxlen=10)
        self.stable_homography = None
        self.is_calibrated = False
        self.calibration_frames = 0
        self.calibration_threshold = 30
        
        # Fixed output parameters
        self.fixed_output_size = None
        self.fixed_translation = None
        
        # CUDA memory pools for reuse
        if self.use_cuda:
            self.gpu_gray1 = None
            self.gpu_gray2 = None

    def preprocess_image_cuda(self, img, gpu_mat=None):
        """Hardware-accelerated image preprocessing"""
        if self.use_cuda:
            # Upload to GPU if needed
            if gpu_mat is None:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
            else:
                gpu_mat.upload(img)
                gpu_img = gpu_mat
            
            # Convert to grayscale on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY, stream=self.stream)
            
            # Apply CLAHE on GPU
            clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gpu_gray = clahe.apply(gpu_gray, stream=self.stream)
            
            # Gaussian blur on GPU
            gpu_gray = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC1, cv2.CV_8UC1, (3, 3), 0
            ).apply(gpu_gray, stream=self.stream)
            
            return gpu_gray
        else:
            # CPU fallback
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            return gray

    def detect_and_compute(self, img):
        """Detect and compute features with hardware acceleration"""
        if self.use_cuda:
            # Preprocess on GPU
            gpu_gray = self.preprocess_image_cuda(img)
            
            # Detect keypoints and compute descriptors on GPU
            gpu_keypoints, gpu_descriptors = self.orb.detectAndComputeAsync(
                gpu_gray, None, stream=self.stream
            )
            
            # Wait for GPU operations to complete
            self.stream.waitForCompletion()
            
            # Download results
            keypoints = self.orb.convert(gpu_keypoints)
            descriptors = gpu_descriptors.download()
            
            return keypoints, descriptors
        else:
            # CPU fallback
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Match features with hardware acceleration"""
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        
        if self.use_cuda:
            # Upload descriptors to GPU
            gpu_desc1 = cv2.cuda_GpuMat()
            gpu_desc2 = cv2.cuda_GpuMat()
            gpu_desc1.upload(desc1)
            gpu_desc2.upload(desc2)
            
            # Match on GPU
            gpu_matches = self.matcher.knnMatch(gpu_desc1, gpu_desc2, k=2)
            matches = gpu_matches
        else:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Filter matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches

    def find_homography_ransac(self, kp1, kp2, matches, min_matches=20):
        """Find homography with RANSAC"""
        if len(matches) < min_matches:
            return None, None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC,
            ransacReprojThreshold=2.0, maxIters=2000, confidence=0.995
        )
        
        return H, mask

    def is_valid_homography(self, H, max_scale=1.3, min_scale=0.7):
        """Validate homography matrix"""
        if H is None:
            return False
        
        H = H / H[2, 2]
        scale_x = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
        scale_y = np.sqrt(H[0, 1]**2 + H[1, 1]**2)
        
        if scale_x > max_scale or scale_y > max_scale:
            return False
        if scale_x < min_scale or scale_y < min_scale:
            return False
        if abs(H[2, 0]) > 0.0005 or abs(H[2, 1]) > 0.0005:
            return False
        
        return True

    def smooth_homography(self, H):
        """Smooth homography over time for stability"""
        if not self.is_valid_homography(H):
            return self.stable_homography
        
        self.homography_buffer.append(H)
        self.calibration_frames += 1
        
        if self.calibration_frames < self.calibration_threshold:
            if len(self.homography_buffer) >= 5:
                H_array = np.array(list(self.homography_buffer))
                median_H = np.median(H_array, axis=0)
                return median_H
            return H
        
        if not self.is_calibrated:
            H_array = np.array(list(self.homography_buffer))
            self.stable_homography = np.median(H_array, axis=0)
            self.is_calibrated = True
            print("✓ Calibration complete!")
        
        if self.stable_homography is not None:
            smoothed = 0.95 * self.stable_homography + 0.05 * H
            return smoothed
        
        return H

    def compute_fixed_output_params(self, img1, img2, H):
        """Compute fixed output canvas size and translation"""
        if self.fixed_output_size is not None:
            return self.fixed_output_size, self.fixed_translation
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners_img1, H)
        
        corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        all_corners = np.concatenate((warped_corners, corners_img2), axis=0)
        
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 10)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 10)
        
        self.fixed_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
        self.fixed_output_size = (x_max - x_min, y_max - y_min)
        
        return self.fixed_output_size, self.fixed_translation

    def warp_and_blend_cuda(self, img1, img2, H):
        """Hardware-accelerated warping and blending"""
        h2, w2 = img2.shape[:2]
        output_size, translation = self.compute_fixed_output_params(img1, img2, H)
        
        if self.use_cuda:
            # Upload images to GPU
            gpu_img1 = cv2.cuda_GpuMat()
            gpu_img1.upload(img1)
            
            # Warp on GPU
            warp_matrix = translation.dot(H)
            gpu_warped = cv2.cuda.warpPerspective(
                gpu_img1, warp_matrix, output_size,
                flags=cv2.INTER_LINEAR,
                stream=self.stream
            )
            
            # Download result
            self.stream.waitForCompletion()
            warped_img1 = gpu_warped.download()
        else:
            # CPU fallback
            warped_img1 = cv2.warpPerspective(
                img1, translation.dot(H), output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
        
        # Create canvas and blend
        canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        
        y_start = max(0, int(-translation[1, 2]))
        y_end = min(output_size[1], int(y_start + h2))
        x_start = max(0, int(-translation[0, 2]))
        x_end = min(output_size[0], int(x_start + w2))
        
        canvas[y_start:y_end, x_start:x_end] = img2[:y_end-y_start, :x_end-x_start]
        
        # Simple alpha blending in overlap region
        mask1 = (warped_img1.sum(axis=2) > 0).astype(np.uint8)
        mask2 = (canvas.sum(axis=2) > 0).astype(np.uint8)
        overlap = np.logical_and(mask1 > 0, mask2 > 0)
        
        result = np.where(mask1[:,:,None] > 0, warped_img1, canvas)
        result[overlap] = ((warped_img1[overlap].astype(np.float32) * 0.5 + 
                           canvas[overlap].astype(np.float32) * 0.5)).astype(np.uint8)
        
        return result

    def stitch(self, img1, img2):
        """Main stitching pipeline"""
        # Feature detection and matching
        kp1, desc1 = self.detect_and_compute(img1)
        kp2, desc2 = self.detect_and_compute(img2)
        
        matches = self.match_features(desc1, desc2)
        
        if len(matches) < 20:
            if self.stable_homography is not None:
                panorama = self.warp_and_blend_cuda(img1, img2, self.stable_homography)
                return panorama, 0, self.is_calibrated
            return None, 0, self.is_calibrated
        
        H, mask = self.find_homography_ransac(kp1, kp2, matches)
        
        if H is None or not self.is_valid_homography(H):
            if self.stable_homography is not None:
                panorama = self.warp_and_blend_cuda(img1, img2, self.stable_homography)
                return panorama, 0, self.is_calibrated
            return None, 0, self.is_calibrated
        
        H_stable = self.smooth_homography(H)
        inliers = int(np.sum(mask)) if mask is not None else 0
        panorama = self.warp_and_blend_cuda(img1, img2, H_stable)
        
        return panorama, inliers, self.is_calibrated


class JetsonStereoStitchNode(Node):
    """ROS2 node for Jetson stereo stitching with 5G streaming"""
    
    def __init__(self, camera_index=0, n_features=3000, match_threshold=0.75):
        super().__init__('jetson_stereo_stitch_node')
        
        # Declare parameters
        self.declare_parameter('camera_index', camera_index)
        self.declare_parameter('n_features', n_features)
        self.declare_parameter('match_threshold', match_threshold)
        self.declare_parameter('use_compressed', True)  # For 5G streaming
        self.declare_parameter('jpeg_quality', 85)  # Compression quality
        
        # Get parameters
        self.camera_index = self.get_parameter('camera_index').value
        n_features = self.get_parameter('n_features').value
        match_threshold = self.get_parameter('match_threshold').value
        self.use_compressed = self.get_parameter('use_compressed').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        
        # Initialize camera with GStreamer pipeline for lower latency
        gst_pipeline = self.create_gstreamer_pipeline(self.camera_index)
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open camera {camera_index}')
            return
        
        self.get_logger().info(f'✓ Camera {camera_index} opened with GStreamer')
        
        # Initialize Jetson-optimized stitcher
        self.stitcher = JetsonStereoStitcher(
            n_features=n_features,
            match_threshold=match_threshold,
            use_cuda=True
        )
        
        # Publishers - both raw and compressed for flexibility
        self.publisher_raw = self.create_publisher(
            Image, 
            f'/stereo_cameras/camera_{camera_index}/panorama',
            10
        )
        
        self.publisher_compressed = self.create_publisher(
            CompressedImage,
            f'/stereo_cameras/camera_{camera_index}/panorama/compressed',
            10
        )
        
        self.bridge = CvBridge()
        
        # Timer for processing frames
        self.timer = self.create_timer(0.033, self.timer_callback)  # 30 Hz
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0
        self.frame_count = 0
        self.avg_processing_time = 0
        
        self.get_logger().info(f'✓ Jetson Stereo Stitch Node initialized for camera {camera_index}')
        self.get_logger().info('Calibrating... Keep camera steady for 30 frames')

    def create_gstreamer_pipeline(self, camera_index):
        """Create optimized GStreamer pipeline for Jetson"""
        # Using v4l2src with hardware acceleration
        pipeline = (
            f'v4l2src device=/dev/video{camera_index} ! '
            'video/x-raw,width=1280,height=720,framerate=30/1 ! '
            'videoconvert ! '
            'video/x-raw,format=BGR ! '
            'appsink drop=1'
        )
        return pipeline

    def timer_callback(self):
        """Main processing loop"""
        ret, stereo_frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame")
            return
        
        # Split stereo frame
        left, right = split_stereo_frame(stereo_frame)
        
        # Stitch images
        start_time = time.time()
        panorama, inliers, is_calibrated = self.stitcher.stitch(left, right)
        processing_time = time.time() - start_time
        
        # Update average processing time
        self.avg_processing_time = 0.9 * self.avg_processing_time + 0.1 * processing_time
        
        # Calculate FPS
        self.fps_counter += 1
        if self.fps_counter >= 10:
            elapsed = time.time() - self.fps_start
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start = time.time()
        
        # Publish results
        if panorama is not None:
            try:
                # Publish raw image
                msg_raw = self.bridge.cv2_to_imgmsg(panorama, encoding='bgr8')
                msg_raw.header.stamp = self.get_clock().now().to_msg()
                msg_raw.header.frame_id = f'camera_{self.camera_index}'
                self.publisher_raw.publish(msg_raw)
                
                # Publish compressed image for 5G streaming
                if self.use_compressed:
                    msg_compressed = CompressedImage()
                    msg_compressed.header = msg_raw.header
                    msg_compressed.format = "jpeg"
                    
                    # Encode with specified quality
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                    _, buffer = cv2.imencode('.jpg', panorama, encode_param)
                    msg_compressed.data = buffer.tobytes()
                    
                    self.publisher_compressed.publish(msg_compressed)
                
                # Log performance metrics periodically
                if self.frame_count % 100 == 0:
                    status = "CALIBRATED" if is_calibrated else f"CALIBRATING {self.stitcher.calibration_frames}/30"
                    self.get_logger().info(
                        f'Camera {self.camera_index}: {self.current_fps:.1f} FPS | '
                        f'{self.avg_processing_time*1000:.1f}ms | {inliers} inliers | {status}'
                    )
                
            except Exception as e:
                self.get_logger().error(f'Failed to publish: {str(e)}')
        
        self.frame_count += 1

    def cleanup(self):
        """Release resources"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


def main(args=None):
    rclpy.init(args=args)
    
    # Get camera index from command line or parameter
    import sys
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    node = JetsonStereoStitchNode(camera_index=camera_idx)
    
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