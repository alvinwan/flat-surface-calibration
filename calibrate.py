import numpy as np
import cv2


def compute_homography_ims(original: np.array, observed: np.array) -> np.array:
    """Compute the transformation needed to right the observed image.

    The current code is adapted to checkerboard originals.

    :param original: original image projected in h x w x 3
    :param observed: the projection observed in h x w x 3
    :return: homography matrix 3 x 3
    """
    # termination critera
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # convert images and find corners
    ori_im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    obs_im = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY)
    ori_ret, ori_pts3d = cv2.findChessboardCorners(ori_im, (7, 6), None)
    obs_ret, obs_pts3d = cv2.findChessboardCorners(obs_im, (7, 6), None)

    if ori_ret and obs_ret:
        # convert 3d corners to 2d corners
        ori_pts2d = cv2.cornerSubPix(ori_im, ori_pts3d, (11, 11), (-1, -1), crit)
        obs_pts2d = cv2.cornerSubPix(obs_im, obs_pts3d, (11, 11), (-1, -1), crit)
        ori_pts2d = ori_pts2d.squeeze(1)
        obs_pts2d = obs_pts2d.squeeze(1)
        return compute_homography_pts(ori_pts2d, obs_pts2d)


def compute_homography_pts(pts1: np.array, pts2: np.array) -> np.array:
    """Compute homography between two sets of points.

    Gracefully handles sets with different numbers of points, by ignoring
    extras in the larger of the two sets.

    :param pts1: First set of points n x 2
    :param pts2: Second set of points n x 2
    :return: homography matrix 3 x 3
    """
    A = []
    b = pts2.reshape(-1, 1)
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        A.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2])
        A.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2])
    A = np.array(A)
    H_8 = np.linalg.pinv(A.T.dot(A)).dot(A.T).dot(b)
    return np.vstack((H_8, 1)).reshape(3, 3)


if __name__ == '__main__':
    im1 = cv2.imread('images/calib_radial.jpg')
    im2 = cv2.imread('images/calib_result.jpg')
    H = compute_homography_ims(im1, im2)
    print(H)