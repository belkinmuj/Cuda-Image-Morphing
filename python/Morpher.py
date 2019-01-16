import numpy as np


import Morpher as BL
import Morpher as BW
import Morpher as BB 

def morph(
  src_img: np.ndarray,
  src_points: np.array,
  dest_img: np.ndarray,
  dest_points: np.array,
  num_frames: int = 20,
  fps = 10,
  alpha = False) -> np.ndarray:
  """
    Generates frames for morphing. Src to dest.
  """
  size = src_img.shape
  # clipping [1,fps] to avoid exceptions
  stall_frames = np.clip(int(fps*0.15), 1, fps)  

  num_frames -= (stall_frames * 2)

  # Generate (num_frames)
  for percent in np.linspace(1, 0, num = num_frames):
    """
      The loop iterates over {num_frames} equal intervals.
    """
    points = BL.src_dest_average_points(src_points, dest_points, percent)
    src_face = BW.warp_image(src_img, src_points, points, size)
    end_face = BW.warp_image(dest_img, dest_points, points, size)
    average_face = BB.src_dest_weighted_average(src_face, end_face, percent)
    yield average_face