
import numpy as np
import cv2
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available, falling back to CPU")

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StitchConfig:
    """Configuration for stereo stitching"""
    homography: np.ndarray
    output_width: int
    output_height: int
    blend_width: int = 50
    use_gpu: bool = True


class CUDAStereoStitcher:
    """Hardware-accelerated stereo stitcher for Jetson Thor"""
    
    def __init__(self, config: StitchConfig):
        self.config = config
        self.use_gpu = config.use_gpu and CUPY_AVAILABLE
        
        if self.use_gpu:
            self.H_gpu = cp.asarray(config.homography, dtype=cp.float32)
            self._create_blend_mask_gpu()
        else:
            self.H_cpu = config.homography.astype(np.float32)
            self._create_blend_mask_cpu()
        
        print(f"Stereo Stitcher: GPU={'enabled' if self.use_gpu else 'disabled'}")
    
    def _create_blend_mask_gpu(self):
        """Create feathering mask on GPU"""
        w, h = self.config.output_width, self.config.output_height
        blend = self.config.blend_width
        
        x = cp.arange(w, dtype=cp.float32)
        alpha = cp.ones((h, w), dtype=cp.float32)
        
        mask_left = cp.clip(x / blend, 0, 1)
        alpha *= mask_left[None, :]
        
        mask_right = cp.clip((w - x) / blend, 0, 1)
        alpha *= mask_right[None, :]
        
        self.blend_mask_gpu = alpha[:, :, None]
    
    def _create_blend_mask_cpu(self):
        """Create feathering mask on CPU"""
        w, h = self.config.output_width, self.config.output_height
        blend = self.config.blend_width
        
        x = np.arange(w, dtype=np.float32)
        alpha = np.ones((h, w), dtype=np.float32)
        
        mask_left = np.clip(x / blend, 0, 1)
        alpha *= mask_left[None, :]
        
        mask_right = np.clip((w - x) / blend, 0, 1)
        alpha *= mask_right[None, :]
        
        self.blend_mask_cpu = alpha[:, :, None]
    
    def stitch(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Stitch stereo pair"""
        if self.use_gpu:
            return self._stitch_gpu(left, right)
        else:
            return self._stitch_cpu(left, right)
    
    def _stitch_gpu(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """GPU-accelerated stitching"""
        # Warp left image
        warped_left = cv2.warpPerspective(
            left,
            self.config.homography,
            (self.config.output_width, self.config.output_height),
            flags=cv2.INTER_LINEAR
        )
        
        # Upload to GPU
        warped_gpu = cp.asarray(warped_left, dtype=cp.uint8)
        right_gpu = cp.asarray(right, dtype=cp.uint8)
        
        # Resize right if needed
        if right_gpu.shape[:2] != (self.config.output_height, self.config.output_width):
            right_cpu = cv2.resize(right, (self.config.output_width, self.config.output_height))
            right_gpu = cp.asarray(right_cpu, dtype=cp.uint8)
        
        # Blend on GPU
        warped_f = warped_gpu.astype(cp.float32)
        right_f = right_gpu.astype(cp.float32)
        
        valid_warped = (warped_f.sum(axis=2, keepdims=True) > 0).astype(cp.float32)
        valid_right = (right_f.sum(axis=2, keepdims=True) > 0).astype(cp.float32)
        
        overlap = valid_warped * valid_right
        alpha = self.blend_mask_gpu * overlap + (1 - overlap) * valid_warped
        
        result_gpu = (alpha * warped_f + (1 - alpha) * right_f).astype(cp.uint8)
        
        return cp.asnumpy(result_gpu)
    
    def _stitch_cpu(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """CPU fallback stitching"""
        warped_left = cv2.warpPerspective(
            left,
            self.config.homography,
            (self.config.output_width, self.config.output_height)
        )
        
        if right.shape[:2] != (self.config.output_height, self.config.output_width):
            right = cv2.resize(right, (self.config.output_width, self.config.output_height))
        
        warped_f = warped_left.astype(np.float32)
        right_f = right.astype(np.float32)
        
        valid_warped = (warped_f.sum(axis=2, keepdims=True) > 0).astype(np.float32)
        valid_right = (right_f.sum(axis=2, keepdims=True) > 0).astype(np.float32)
        
        overlap = valid_warped * valid_right
        alpha = self.blend_mask_cpu * overlap + (1 - overlap) * valid_warped
        
        result = (alpha * warped_f + (1 - alpha) * right_f).astype(np.uint8)
        
        return result
    
    def cleanup(self):
        """Release GPU resources"""
        if self.use_gpu and CUPY_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()


def split_stereo_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split side-by-side stereo frame"""
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]