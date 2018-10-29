"""
basic keypoint transformations
NOTE: Typically, opencv or tensorflow is faster than skimage. See example_performance.py.
"""

import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage import transform as sktf

import fusewarp as fw
from fusewarp import transform as ft

img0 = skimage.data.rocket()
keypoints0 = np.array([[323, 128], [202, 126], [447, 126], [198, 217], [451, 217]], dtype=np.float32)
height, width = img0.shape[:2]

# reference
plt.imshow(img0)
plt.suptitle("reference")
plt.scatter(keypoints0[:, 0], keypoints0[:, 1])
plt.show()

warp = fw.FuseWarp([
    ft.Scale((0.8, 1.2), resize=True),
    ft.Rotate((-10, 10)),
    ft.FlipLR(),
    ft.FrameRandom(300, 300)])

for i in range(10):
    warp.next_sample((width, height))
    transform = warp.get_transform()
    w, h = warp.get_size()

    img1 = sktf.warp(img0, transform.inverse, output_shape=(h, w))
    keypoints1 = transform(keypoints0).astype(np.int32)
    # remove outer keypoints
    keypoints1 = keypoints1[keypoints1[:, 0] >= 0]
    keypoints1 = keypoints1[keypoints1[:, 0] < w]
    keypoints1 = keypoints1[keypoints1[:, 1] >= 0]
    keypoints1 = keypoints1[keypoints1[:, 1] < h]

    plt.imshow(img1)
    plt.scatter(keypoints1[:, 0], keypoints1[:, 1])
    plt.show()