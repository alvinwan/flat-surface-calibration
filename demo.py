from calibrate import generate_checkerboard_image
from calibrate import compute_homography_ims
from skimage.transform import warp
import cv2


def is_key(key, character: str) -> bool:
    return key & 0xFF == ord(character)


def main():
    cap = cv2.VideoCapture(0)
    original = extracted = generate_checkerboard_image()
    height, width, _ = original.shape
    H = obs_pts = None

    while True:
        ret, observed = cap.read()
        if H is None:
            H, ori_pts, obs_pts = compute_homography_ims(original, observed)
            print('Calibrated!' if H is not None else 'Calibrating...')
        else:
            extracted = warp(observed, H)[:height, :width]
        cv2.drawChessboardCorners(observed, (5, 5), obs_pts, H is not None)
        cv2.imshow('observed', observed)
        cv2.imshow('original', extracted)
        if is_key(cv2.waitKey(1), 'q'):
            break


if __name__ == '__main__':
    main()