# Flat Surface Calibration
Compute calibration for projection on flat surface, to sidestep affine
distortions. Why would we want this? Say we have an image I on our
computer. Project this image, and say we now place our finger on the
projection. We wish to correspond our finger, on the projected image,
with the actual image on our computer. This repository allows us to do
that.

Specifically, we compute a homography. Take projection I' of 
image I. For any point p' on I', this repository allows you to find the
corresponding point p on I.

Below is a demo that uses this "flat surface calibration". We take the
camera input, and extract the projected image. Then, we transform the
projected image, so that its orientation matches the original. Finally,
we project the transformed, projected image. Notice that 
the projection "flickers" between a perfectly sharp image and a 
slightly pale image. The pale image is a projection of the extracted, 
projected image! For this demo, we simply use the two functions:

```
original = generate_checkerboard_image()
...
extracted = extract_projected_screen(original, observed, debug=True)
```

![flat-surface-calibration](https://user-images.githubusercontent.com/2068077/34454204-347dab6a-ed1b-11e7-9870-9574fd2a570f.gif)

## Installation

1. (Recommended) [Setup a Python virtual environment](https://www.digitalocean.com/community/tutorials/common-python-tools-using-virtualenv-installing-with-pip-and-managing-packages#a-thorough-virtualenv-how-to) with Python 3.6.

2. Install all Python dependencies.

```
pip install -r requirements.txt
```

> For Ubuntu users, install `libsm6` for OpenCV to work:
> ```
> apt install libsm6
> ```

Installation is complete. To get started, launch the demo.

```
python demo.py
```

