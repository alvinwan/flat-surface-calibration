from calibrate import generate_checkerboard_image
from calibrate import compute_homography_ims
from skimage.transform import warp
import numpy as np
import cv2


def is_key(key, character: str) -> bool:
    return key & 0xFF == ord(character)


def extract_image_crop(im: np.array, points: np.array) -> np.array:
    """Take bounding box for points, and extract that portion of the image."""
    xmin, xmax = int(min(points[:, 0])), int(max(points[:, 0]))
    ymin, ymax = int(min(points[:, 1])), int(max(points[:, 1]))
    return im[ymin: ymax, xmin: xmax]


def main():
    """Utility for calibrating the camera and projector."""
    cap = cv2.VideoCapture(0)
    original = generate_checkerboard_image()

    while True:
        ret, observed = cap.read()
        H, obs_pts, ori_pts = compute_homography_ims(observed, original)

        cv2.drawChessboardCorners(observed, (5, 5), obs_pts, H is not None)
        cv2.imshow('observed', observed)

        if H is not None:
            crop = extract_image_crop(observed, obs_pts)
            # crop = warp(crop, H)

            # hacky fix - can't extract true corners
            crop = cv2.resize(crop, (400, 360))
            mock = original.copy()
            mock[90: 450, 100: 500] = crop
        else:
            mock = original
        cv2.imshow('original', mock)
        if is_key(cv2.waitKey(500), 'q'):
            break


if __name__ == '__main__':
    main()