from skimage.transform import warp
import numpy as np
import cv2


def generate_checkerboard_image(
        width: int=3,
        height: int=3,
        cell_width: int=100,
        cell_height: int=90,
        as_rgb: bool=True
    ) -> np.array:
    """Generate image of a checkerboard."""
    white = np.zeros((cell_height, cell_width))
    black = white + 1
    row_odd = np.hstack([white, black] * width)
    row_even = np.hstack([black, white] * width)
    channel = np.vstack([row_odd, row_even] * height).astype(np.uint8)
    if as_rgb:
        return np.transpose(np.stack([channel] * 3), (1, 2, 0)) * 255
    return channel


def compute_adjusted_im(original: np.array, observed: np.array) -> np.array:
    """Computes adjusted image to project, so that the projection is righted.

    Note: In retrospect, this function is largely impractical. This alignment
    assumes that your camera is facing the plane head on, which may not
    always be the case.

    :param original: original image projected in h x w x 3
    :param observed: the projection observed in h x w x 3
    :return: adjusted image h x w x 3
    """
    H, _, _ = compute_homography_ims(observed, original)
    assert H is not None, 'Observed image does not contain checkerboard. Are' \
        ' you sure your computer is pointing in the right direction?'
    return warp(original, H)


def computed_projected_points(points: np.array, H: np.array) -> np.array:
    """Projects points using homography.

    :param points: set of points to project n x 2
    :param H: homography to apply 3 x 3
    :return: transformed points n x 2
    """
    p = np.hstack((points, np.ones((points.shape[0], 1))))
    Hp = p.dot(H.T)
    Hp[:, 0] = Hp[:, 0] / Hp[:, 2]
    Hp[:, 1] = Hp[:, 1] / Hp[:, 2]
    return Hp[:, :2]


def compute_homography_ims(original: np.array, observed: np.array) -> np.array:
    """Compute the transformation needed to right the observed image.

    The current code is adapted to checkerboard originals.

    :param original: original image projected in h x w x 3
    :param observed: the projection observed in h x w x 3
    :return: homography matrix 3 x 3, original points, observed points
    """
    # termination criteria
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # convert images and find corners
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    observed = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY)
    ori_ret, ori_pts3d = cv2.findChessboardCorners(original, (5, 5))
    obs_ret, obs_pts3d = cv2.findChessboardCorners(observed, (5, 5))

    if ori_ret and obs_ret:
        # convert 3d corners to 2d corners
        ori_pts2d = cv2.cornerSubPix(original, ori_pts3d, (11, 11), (-1, -1), crit)
        obs_pts2d = cv2.cornerSubPix(observed, obs_pts3d, (11, 11), (-1, -1), crit)
        ori_pts2d = ori_pts2d.squeeze(1)
        obs_pts2d = obs_pts2d.squeeze(1)
        return compute_homography_pts(ori_pts2d, obs_pts2d), ori_pts2d, obs_pts2d
    return None, None, None


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
    im1 = cv2.imread('images/screenshot-small.png')
    im2 = cv2.imread('images/checkerboard.png')
    H, _, _ = compute_homography_ims(im1, im2)
    print(H)

    original = generate_checkerboard_image()
    while True:
        cv2.imshow('frame', original)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break