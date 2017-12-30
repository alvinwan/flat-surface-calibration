from calibrate import generate_checkerboard_image
from calibrate import extract_projected_screen
import cv2


def is_key(key, character: str) -> bool:
    return key & 0xFF == ord(character)


def main():
    """Utility for calibrating the camera and projector."""
    cap = cv2.VideoCapture(0)
    original = generate_checkerboard_image()

    while True:
        ret, observed = cap.read()
        extracted = extract_projected_screen(original, observed, debug=True)
        cv2.imshow('observed', observed)
        cv2.imshow('original', extracted)
        if is_key(cv2.waitKey(1), 'q'):
            break


if __name__ == '__main__':
    main()