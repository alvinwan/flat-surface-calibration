# Flat Surface Calibration
Compute calibration for projection on flat surface, to sidestep affine
distortions. Why would we want this? Say we project an image and that 
we now place our finger on the projection. We wish to correspond our 
finger, on the projected image, with the actual image on our computer. 
This repository allows us to do that.

Below are two demos that uses this "flat surface calibration":
1. We take the camera input.
2. Extract the projected checkerboard.
3. We warp the extracted, projected checkerboard, so that it matches the original checkerboard.
Finally, we project the new extracted checkerboard. In the first demo, notice
we alternate between the original and a pale version (the projection 
of the projected image!) and the original. For the second demo, the
projection of my hand matches my own hand exactly. (The projection of the
projection begins to lose quality, but that's a problem beyond the
scope of this repository.) For the first, we use just two functions:

```
original = generate_checkerboard_image()
...
extracted = extract_projected_screen(original, observed, debug=True)
```

![flat-surface-calibration](https://user-images.githubusercontent.com/2068077/34454204-347dab6a-ed1b-11e7-9870-9574fd2a570f.gif)
![flat-surface-calibration-hand](https://user-images.githubusercontent.com/2068077/34454469-0b254150-ed21-11e7-9b49-49d44074d294.gif)


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

For the calibration functions available, see `calibrate.py`'s `high-level functions` section. For basic usage of these high-level functions, see `demo_basic.py`.