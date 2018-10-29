"""
example for randomness customization.
"""

import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage import transform as sktf

import fusewarp as fw
from fusewarp import transform as ft

img0 = skimage.data.rocket()
height, width = img0.shape[:2]

# you can use callable as transformer argument.
translate = ft.Translate(lambda: np.random.normal(0, 0.2), 0)

# you can use infinite generators as transformer argument
def repeat(n, a):
    i = 0
    while True:
        yield a*(i % n)
        i = i+1
theta = repeat(3, 30)  # repeat 0,30,60
rotate = ft.Rotate(theta)

# check results
warp = fw.FuseWarp([rotate, translate])
for i in range(12):
    warp.next_sample((width, height))
    transform = warp.get_transform()
    w, h = warp.get_size()

    img1 = sktf.warp(img0, transform.inverse, output_shape=(h, w))
    plt.imshow(img1)
    plt.show()
