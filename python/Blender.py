import numpy as np

def src_dest_weighted_average(
    img1,
    img2,
    measure: float = 0.5) -> np.array:
    """Measure is assummed to be in [0,1]"""

    return np.asarray(img1 * measure + img2 * (1 - measure), dtype = np.uint8)
    # return addWeighted(img1, measure, img2, 1-measure, 0)
