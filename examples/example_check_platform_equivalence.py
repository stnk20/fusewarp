import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage import transform as sktf
import cv2
import tensorflow as tf

import fusewarp as fw
from fusewarp import transform as ft

img0 = skimage.data.rocket()
keypoints0 = np.array([[323, 128], [202, 126], [447, 126], [198, 217], [451, 217]], dtype=np.float32)
height, width = img0.shape[:2]

# check all homography transform which does not change shape
warp = fw.FuseWarp([
    ft.Translate((-0.2, 0.2), (-0.2, 0.2)),
    ft.Rotate((0, 360)),
    ft.Scale((0.2, 2), (0.2, 2)),
    ft.FlipLR(),
    ft.FlipUD(),
    ft.Shear((-0.2, 0.2), (-0.2, 0.2)),
    ft.Perspective((0, 360), (0, 1))])

for i in range(10):
    warp.next_sample((width, height))
    transform = warp.get_transform()
    matrix = warp.get_matrix()
    w, h = warp.get_size()

    # skimage
    img_skimage = sktf.warp(img0, transform.inverse, output_shape=(h, w))
    # opencv
    img_opencv = cv2.warpPerspective(img0, matrix, (w, h))
    # tensorflow
    x = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    m = tf.placeholder(dtype=tf.float32, shape=[8])
    y = tf.contrib.image.transform(
        x, m, interpolation="BILINEAR", output_shape=(h, w))
    matrix_inv = np.linalg.inv(matrix)
    matrix_f = (matrix_inv/matrix_inv[2, 2]).flatten()[:8]
    with tf.Session() as sess:
        img_tensorflow = sess.run(y, feed_dict={x: img0, m: matrix_f})

    keypoints1 = transform(keypoints0).astype(np.int32)
    keypoints1 = keypoints1[keypoints1[:, 0] >= 0]
    keypoints1 = keypoints1[keypoints1[:, 0] < w]
    keypoints1 = keypoints1[keypoints1[:, 1] >= 0]
    keypoints1 = keypoints1[keypoints1[:, 1] < h]

    fig = plt.figure(figsize=(100,100))
    for i, img in enumerate([img_skimage, img_opencv, img_tensorflow]):
        fig.add_subplot(1, 3, i+1)
        plt.imshow(img)
        plt.scatter(keypoints1[:, 0], keypoints1[:, 1])
    plt.show()
