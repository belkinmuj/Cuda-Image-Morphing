import numpy as np

"""
  Delaunay triangulation is a triangulation DT(P)
  such that no point in P is inside the circum-hypersphere of
  any d-simplex in DT(P).
"""
from scipy.spatial import Delaunay
from scipy.interpolate import interp2d

def grid_coordinates(points: np.array, dtype = np.uint16) -> np.array:
  """
    Builds a grid.
    Squared region min -> max.
  """
  xmin = np.min(points[:, 0])
  xmax = np.max(points[:, 0]) + 1
  ymin = np.min(points[:, 1])
  ymax = np.max(points[:, 1]) + 1
  return np.asarray([(x, y) for y in range(ymin, ymax)
                     for x in range(xmin, xmax)], dtype = dtype)


def bilinear_interpolate(img, coords: np.array) -> np.array:
  """
    Bilinear interpolation of each pixel of img on a dest shape.
  :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
  """
  int_coords = np.int32(coords)
  x0, y0 = int_coords
  dx, dy = coords - int_coords

  # V2.
  # f = interp2d(img)
  # f(xcoords, ycoords)

  # 4 Neighour pixels
  q11 = img[y0, x0]
  q21 = img[y0, x0+1]
  q12 = img[y0+1, x0]
  q22 = img[y0+1, x0+1]

  btm = q21.T * dx + q11.T * (1 - dx)
  top = q22.T * dx + q12.T * (1 - dx)
  inter_pixel = top * dy + btm * (1 - dy)

  return inter_pixel.T



def process_warp(src_img, result_img: np.zeros,
                tri_affines: np.matrix, dst_points: np.array,
                delaunay) -> None:
  """
  Warp each triangle from the src_image only within the
  ROI of the destination image (points in dst_points).
  Changes an image.
  """
  roi_coords = grid_coordinates(dst_points)
  # indices to vertices. -1 if pixel is not in any triangle
  roi_tri_indices = delaunay.find_simplex(roi_coords)

  for simplex in enumerate(delaunay.simplices):
    coords = roi_coords[roi_tri_indices == simplex[0]]
    num_coords = len(coords)
    out_coords = np.dot(tri_affines[simplex[0]],
                        np.vstack((coords.T, np.ones(num_coords))))
    x, y = coords.T
    result_img[y, x] = bilinear_interpolate(src_img, out_coords)

  return None

def warp_image(src_img, src_points: list,
    dest_points: list,
    dest_shape: tuple, dtype = np.uint16) -> np.ndarray:
  # Resultant image will not have an alpha channel
  src_img = src_img[:, :, :3]

  result_img = np.zeros(dest_shape, dtype)

  # Generate delaunay triangles
  delaunay = Delaunay(dest_points)
  # Generate affine matrice
  tri_affines = list(
    triangular_affine_matrices(delaunay.simplices, src_points, dest_points)
    )

  tri_affines = np.asarray(tri_affines)

  process_warp(src_img, result_img, tri_affines, dest_points, delaunay)

  return result_img

def triangular_affine_matrices(vertices: list,
  src_points: list,
  dest_points: list) -> np.matrix:
  
  """
  Calculate the affine transformation matrix for each
  triangle (x,y) vertex from dest_points to src_points
  """

  ones = [1] * 3
  for tri_indices in vertices:
    src_tri = np.vstack((src_points[tri_indices, :].T, ones))
    dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
    # Inv computes inversion matrix
    mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
    yield mat

