import cv2
import numpy as np
from skimage import transform as trans


FFHQ_KPS_512 = np.array(
    [
        [196.473, 235.520],
        [316.672, 235.520],
        [256.512, 310.272],
        [208.384, 385.024],
        [304.640, 385.024],
    ],
    dtype=np.float32,
)


def get_similarity_matrix(src_kps, dst_kps):
    """
    Calculate equations for grafting (Protects the face from squirming 100%)
    """
    try:
        tform = trans.SimilarityTransform.from_estimate(src_kps, dst_kps)
    except AttributeError:
        tform = trans.SimilarityTransform()
        tform.estimate(src_kps, dst_kps)
    return tform.params[0:2, :]
