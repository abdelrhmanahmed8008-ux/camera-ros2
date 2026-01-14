import cv2
import numpy as np  
import sys


def split_stereo(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]


def calibrate_camera(camera_index: int, output_path: str):
    """Calibrate stereo camera and save homography"""
    
    print(f"Calibrating camera {camera_index}...")
    print("Keep the camera steady and pointed at a textured scene")
    print("Press SPACE to capture calibration frame, ESC to quit")
    
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    orb = cv2.ORB_create(nfeatures=5000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        left, right = split_stereo(frame)
        
        cv2.imshow('Calibration - Press SPACE', np.hstack([left, right]))
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("Calibration cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return
        
        if key == 32:  # SPACE
            print("Computing homography...")
            
            # Detect features
            gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            
            kp1, des1 = orb.detectAndCompute(gray_l, None)
            kp2, des2 = orb.detectAndCompute(gray_r, None)
            
            # Match features
            matches = bf.knnMatch(des1, des2, k=2)
            
            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
            
            print(f"Found {len(good)} good matches")
            
            if len(good) < 50:
                print("Not enough matches, try again")
                continue
            
            # Compute homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                print("Homography computation failed, try again")
                continue
            
            print(f"Homography computed with {mask.sum()} inliers")
            
            # Save homography
            np.save(output_path, H)
            print(f"Saved homography to {output_path}")
            
            # Show result
            h, w = left.shape[:2]
            warped = cv2.warpPerspective(left, H, (w, h))
            result = cv2.addWeighted(warped, 0.5, right, 0.5, 0)
            
            cv2.imshow('Calibration Result', result)
            cv2.waitKey(2000)
            
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Calibration complete!")


def main():
    if len(sys.argv) < 3:
        print("Usage: calibrate_homography.py <camera_index> <output_file>")
        print("Example: calibrate_homography.py 0 homography_camera_0.npy")
        return
    
    camera_index = int(sys.argv[1])
    output_path = sys.argv[2]
    
    calibrate_camera(camera_index, output_path)


if __name__ == '__main__':
    main()