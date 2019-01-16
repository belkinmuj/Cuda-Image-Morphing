
import numpy as np
def src_dest_average_points(
    start_points,
    end_points,
    measure: float = 0.5) -> np.array:
  """ Generated using measure param.
      Measure is assumed to be in [0,1]
  """
  return np.asarray(start_points*measure + end_points*(1-measure), np.uint8)
