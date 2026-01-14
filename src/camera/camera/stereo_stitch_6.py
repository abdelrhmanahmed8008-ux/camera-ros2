import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import time
from collections import deque


def split_stereo_frame(frame):
    """Split stereo camera frame into left and right"""
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]


class StereoStitcher:
    """Stereo image stitching class"""
    def __init__(self, n_features=5000, match_threshold=0.7):
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=1.2,
            nlevels=12,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31
        )
        
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.match_threshold = match_threshold
        
        # Homography stabilization
        self.homography_buffer = deque(maxlen=10)
        self.stable_homography = None
        self.is_calibrated = False
        self.calibration_frames = 0
        self.calibration_threshold = 30
        
        # Fixed output parameters
        self.fixed_output_size = None
        self.fixed_translation = None

    def detect_and_compute(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Add CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.equalizeHist(gray)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def find_homography_ransac(self, kp1, kp2, matches, min_matches=20):
        if len(matches) < min_matches:
            return None, None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, 
            ransacReprojThreshold=2.0, maxIters=3000, confidence=0.999
        )
        
        return H, mask
    
    def is_valid_homography(self, H, max_scale=1.3, min_scale=0.7):
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
    
    def find_seam_vertical(self, img1, img2, mask1, mask2):
        overlap = np.logical_and(mask1 > 0, mask2 > 0).astype(np.uint8)
        
        if np.sum(overlap) < 100:
            return None
        
        rows, cols = np.where(overlap > 0)
        if len(rows) == 0:
            return None
        
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        
        overlap_width = x_max - x_min
        overlap_height = y_max - y_min
        
        if overlap_width < 10 or overlap_height < 10:
            return None
        
        diff = cv2.absdiff(img1[y_min:y_max, x_min:x_max], img2[y_min:y_max, x_min:x_max])
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        grad_x = cv2.Sobel(diff_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(diff_gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        cost = diff_gray + gradient * 0.5
        
        h, w = cost.shape
        dp = np.zeros_like(cost)
        dp[0, :] = cost[0, :]
        
        for i in range(1, h):
            for j in range(w):
                prev_costs = []
                if j > 0:
                    prev_costs.append(dp[i-1, j-1])
                prev_costs.append(dp[i-1, j])
                if j < w - 1:
                    prev_costs.append(dp[i-1, j+1])
                dp[i, j] = cost[i, j] + min(prev_costs)
        
        seam = np.zeros(h, dtype=np.int32)
        seam[-1] = np.argmin(dp[-1, :])
        
        for i in range(h-2, -1, -1):
            prev_col = seam[i+1]
            search_range = [prev_col]
            if prev_col > 0:
                search_range.append(prev_col - 1)
            if prev_col < w - 1:
                search_range.append(prev_col + 1)
            seam[i] = min(search_range, key=lambda c: dp[i, c])
        
        seam_global = np.zeros((overlap_height, 2), dtype=np.int32)
        for i in range(overlap_height):
            seam_global[i] = [y_min + i, x_min + seam[i]]
        
        return seam_global
    
    def warp_and_blend(self, img1, img2, H):
        h2, w2 = img2.shape[:2]
        output_size, translation = self.compute_fixed_output_params(img1, img2, H)
        
        warped_img1 = cv2.warpPerspective(img1, translation.dot(H), output_size,
                                  flags=cv2.INTER_CUBIC,  # Better quality
                                  borderMode=cv2.BORDER_CONSTANT)
        
        canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        
        y_start = max(0, int(-translation[1, 2]))
        y_end = min(output_size[1], int(y_start + h2))
        x_start = max(0, int(-translation[0, 2]))
        x_end = min(output_size[0], int(x_start + w2))
        
        canvas[y_start:y_end, x_start:x_end] = img2[:y_end-y_start, :x_end-x_start]
        
        mask1 = (warped_img1.sum(axis=2) > 0).astype(np.uint8)
        mask2 = (canvas.sum(axis=2) > 0).astype(np.uint8)
        
        seam = self.find_seam_vertical(warped_img1, canvas, mask1, mask2)
        
        if seam is not None:
            seam_mask = np.zeros(output_size[::-1], dtype=np.uint8)
            for y, x in seam:
                seam_mask[y, :x] = 1
            
            seam_mask_float = seam_mask.astype(np.float32)
            feather_mask = cv2.GaussianBlur(seam_mask_float, (5, 5), 0)
            feather_mask = np.dstack([feather_mask] * 3)
            
            overlap = np.logical_and(mask1 > 0, mask2 > 0).astype(np.uint8)
            overlap_3d = np.dstack([overlap] * 3)
            
            result = np.where(overlap_3d > 0,
                            (warped_img1.astype(np.float32) * feather_mask + 
                             canvas.astype(np.float32) * (1 - feather_mask)).astype(np.uint8),
                            np.where(mask1[:,:,None] > 0, warped_img1, canvas))
        else:
            result = np.where(mask1[:,:,None] > 0, warped_img1, canvas)
        
        return result
    
    def stitch(self, img1, img2):
        kp1, desc1 = self.detect_and_compute(img1)
        kp2, desc2 = self.detect_and_compute(img2)
        
        matches = self.match_features(desc1, desc2)
        
        if len(matches) < 20:
            if self.stable_homography is not None:
                panorama = self.warp_and_blend(img1, img2, self.stable_homography)
                return panorama, 0, self.is_calibrated
            return None, 0, self.is_calibrated
        
        H, mask = self.find_homography_ransac(kp1, kp2, matches)
        
        if H is None or not self.is_valid_homography(H):
            if self.stable_homography is not None:
                panorama = self.warp_and_blend(img1, img2, self.stable_homography)
                return panorama, 0, self.is_calibrated
            return None, 0, self.is_calibrated
        
        H_stable = self.smooth_homography(H)
        inliers = int(np.sum(mask)) if mask is not None else 0
        panorama = self.warp_and_blend(img1, img2, H_stable)
        
        return panorama, inliers, self.is_calibrated
    


class StereoStitchNode(Node):
    def __init__(self, camera_index=6, n_features=3000, match_threshold=0.75):
        super().__init__('stereo_stitch_node')
        self.camera_index = camera_index
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open camera {camera_index}')
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'Camera {camera_index} opened: {actual_width}x{actual_height}')
        
        # Initialize stitcher
        self.stitcher = StereoStitcher(n_features=n_features, match_threshold=match_threshold)
        
        # Publisher for stitched image
        self.publisher = self.create_publisher(Image, f'camera_{camera_index}/panorama', 10)
        self.bridge = CvBridge()
        
        # Timer for processing frames
        self.timer = self.create_timer(0.033, self.timer_callback)  # ~30 Hz
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0
        self.frame_count = 0
        
        self.get_logger().info(f'Stereo Stitch Node initialized for camera {camera_index}')
        self.get_logger().info('Calibrating... Keep camera steady for 30 frames')

    def timer_callback(self):
        ret, stereo_frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame")
            return
        
        # Split stereo frame into left and right
        left, right = split_stereo_frame(stereo_frame)
        
        # Stitch images
        start_time = time.time()
        panorama, inliers, is_calibrated = self.stitcher.stitch(left, right)
        processing_time = time.time() - start_time
        
        # Calculate FPS
        self.fps_counter += 1
        if self.fps_counter >= 10:
            elapsed = time.time() - self.fps_start
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start = time.time()
        
        # Display and publish
        if panorama is not None:
            display = panorama.copy()
            
            # Add status text
            if is_calibrated:
                status = "CALIBRATED"
                color = (0, 255, 0)
            else:
                status = f"CALIBRATING {self.stitcher.calibration_frames}/30"
                color = (0, 165, 255)
            
            info = f"Camera_{self.camera_index} | {self.current_fps:.1f} FPS | {processing_time*1000:.0f}ms | {inliers} inliers | {status}"
            cv2.putText(display, info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            #cv2.imshow(f'Panorama Camera {self.camera_index}', display)
            
            # Publish ROS message
            try:
                msg = self.bridge.cv2_to_imgmsg(panorama, encoding='bgr8')
                self.publisher.publish(msg)
            except Exception as e:
                self.get_logger().error(f'Failed to publish image: {str(e)}')
        else:
            # Show raw stereo if stitching fails
            combined = np.hstack((left, right))
            cv2.putText(combined, "Initializing...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            #cv2.imshow(f'Panorama Camera {self.camera_index}', combined)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and panorama is not None:
            filename = f'Camera_{self.camera_index}_panorama_{self.frame_count:04d}.jpg'
            cv2.imwrite(filename, panorama)
            self.get_logger().info(f'Saved: {filename}')
        elif key == ord('r'):
            self.get_logger().info('Recalibrating...')
            self.stitcher.is_calibrated = False
            self.stitcher.calibration_frames = 0
            self.stitcher.homography_buffer.clear()
            self.stitcher.stable_homography = None
            self.stitcher.fixed_output_size = None
            self.stitcher.fixed_translation = None
        
        self.frame_count += 1

    def cleanup(self):
        """Release resources"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
  
    node = StereoStitchNode(camera_index=6)
    
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