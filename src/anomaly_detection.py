import numpy as np

class AnomalyDetector:
    def __init__(self):
        self.motion_history = []
        self.density_history = []

    def compute_score(self, motion, density):
        self.motion_history.append(motion)
        self.density_history.append(density)

        if len(self.motion_history) < 10:
            return 0.0

        motion_mean = np.mean(self.motion_history)
        density_mean = np.mean(self.density_history)

        motion_dev = abs(motion - motion_mean)
        density_dev = abs(density - density_mean)

        score = (motion_dev + density_dev) / 2.0
        return min(score, 1.0)
