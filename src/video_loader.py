import cv2
import os
import numpy as np

class VideoLoader:
    def __init__(self, video_path, resize_width=640, resize_height=480, frame_skip=1):
        self.video_path = video_path
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.frame_skip = frame_skip

        # Handle both file path AND webcam (0)
        if isinstance(video_path, int) or os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)
        else:
            raise FileNotFoundError(f"[ERROR] Video file not found: {self.video_path}")

        if not self.cap.isOpened():
            raise IOError(f"[ERROR] Cannot open video: {video_path}")

        self.frame_count = 0
        print(f"✅ VideoLoader initialized: {video_path}")

    def read(self):
        """
        Reads the next valid frame based on frame skipping.
        Returns:
            ret (bool): Whether frame was read
            frame (np.ndarray): Processed frame
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                return False, None

            self.frame_count += 1

            if self.frame_count % self.frame_skip != 0:
                continue

            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            return True, frame

    def release(self):
        self.cap.release()

if __name__ == "__main__":
    print("🚀 Testing VideoLoader class...")
    
    # Test 1: Check without video (safe)
    try:
        loader = VideoLoader("data/videos/merged_crowd_demo.mp4")
        print("✅ Class initializes OK")
    except FileNotFoundError as e:
        print(f"ℹ️  {e} - NORMAL without video file")
    
    # Test 2: WEBCAM TEST (WORKS IMMEDIATELY)
    print("\n🎬 WEBCAM TEST...")
    try:
        loader = VideoLoader(0, frame_skip=5)  # 0 = webcam
        ret, frame = loader.read()
        if ret:
            print(f"✅ WEBCAM WORKS: Frame shape {frame.shape}")
            print(f"✅ Frame count: {loader.frame_count}")
        else:
            print("ℹ️ No frames from webcam")
        loader.release()
    except Exception as e:
        print(f"ℹ️ No webcam: {e}")
    
    print("\n🎉 VideoLoader class READY FOR HACKATHON!")