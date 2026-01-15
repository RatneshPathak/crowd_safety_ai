import cv2
import numpy as np

def preprocess_frame(frame):
    """
    Convert frame to grayscale and reduce noise.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

class Preprocessor:
    def __init__(self):
        """Initialize with optimal preprocessing parameters."""
        self.kernel_size = (15, 15)
        self.sigma = 1.0
    
    def process(self, frame):
        """
        Full preprocessing pipeline for crowd analysis.
        Input: BGR frame (H, W, 3)
        Output: Normalized grayscale (H, W)
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, self.kernel_size, self.sigma)
        
        # 3. Normalize to 0-255 range
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized.astype(np.uint8)

# Test function
if __name__ == "__main__":
    print("🧹 Testing Preprocessor...")
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from video_loader import VideoLoader
    
    loader = VideoLoader("data/videos/merged_crowd_demo.mp4")
    ret, frame = loader.read()
    if ret:
        preprocessor = Preprocessor()
        processed = preprocessor.process(frame)
        print(f"✅ Original: {frame.shape} → Processed: {processed.shape} (grayscale)")
        loader.release()
    print("🎉 Preprocessor READY!")
