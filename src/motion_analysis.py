import cv2
import numpy as np

def compute_motion(prev_gray, curr_gray):
    """
    Compute motion magnitude using optical flow.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_motion = np.mean(magnitude)

    return avg_motion
