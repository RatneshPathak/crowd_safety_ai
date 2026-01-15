import numpy as np

def estimate_density(gray_frame, rows, cols):
    """
    Estimate crowd density using pixel intensity variations.
    """
    h, w = gray_frame.shape
    grid_h = h // rows
    grid_w = w // cols

    density_map = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            cell = gray_frame[
                i * grid_h:(i + 1) * grid_h,
                j * grid_w:(j + 1) * grid_w
            ]
            density_map[i, j] = np.mean(cell)

    # Normalize
    density_map = density_map / 255.0
    return density_map
