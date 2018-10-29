"""
check performance of tensorflow(cpu/gpu), opencv, skimage

Results in my laptop ( Core(TM) i7-7700HQ CPU @ 2.80GHz / Geforce GTX1050 GPU ) :

tensorflow-cpu
elapsed seconds per 100 images:  1.1677992343902588

tensorflow-gpu
elapsed seconds per 100 images:  0.14475488662719727

opencv
elapsed seconds per 100 images:  0.11846280097961426

skimage
elapsed seconds per 100 images:  2.898721694946289
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import transform as sktf
import fusewarp as fw
from fusewarp import transform as ft

import time

img = skimage.data.rocket()
# check all homography transform which does not change shape
warp = fw.FuseWarp([
    ft.Translate((-0.5, 0.5), (-0.5, 0.5)),
    ft.Rotate((0, 360)),
    ft.Scale((0.5, 2), (0.5, 2)),
    ft.FlipLR(),
    ft.FlipUD(),
    ft.Shear((-0.5, 0.5), (-0.5, 0.5)),
    ft.Perspective((0, 360), (0, 1))])


def transform_tensorflow(img, warp):
    import tensorflow as tf
    height, width = img.shape[:2]

    x = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    m = tf.placeholder(dtype=tf.float32, shape=[8])
    y = tf.contrib.image.transform(
        x, m, interpolation="BILINEAR", output_shape=(height, width))

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        t0 = time.time()
        for i in range(100):
            warp.next_sample((width, height))
            matrix = warp.get_matrix()
            w, h = warp.get_size()
            matrix_inv = np.linalg.inv(matrix)
            matrix_f = (matrix_inv/matrix_inv[2, 2]).flatten()[:8]
            img1 = sess.run(y, feed_dict={x: img, m: matrix_f})
        t1 = time.time()
        print()
        print("tensorflow-cpu")
        print("elapsed seconds per 100 images: ", t1-t0)
        print()

    time.sleep(3)

    with tf.Session() as sess:
        t0 = time.time()
        for i in range(100):
            warp.next_sample((width, height))
            matrix = warp.get_matrix()
            w, h = warp.get_size()
            matrix_inv = np.linalg.inv(matrix)
            matrix_f = (matrix_inv/matrix_inv[2, 2]).flatten()[:8]
            img1 = sess.run(y, feed_dict={x: img, m: matrix_f})
        t1 = time.time()
        print()
        print("tensorflow-gpu")
        print("elapsed seconds per 100 images: ", t1-t0)
        print()


def transform_opencv(img, warp):
    import cv2
    height, width = img.shape[:2]

    t0 = time.time()
    for i in range(100):
        warp.next_sample((width, height))
        matrix = warp.get_matrix()
        w, h = warp.get_size()
        img1 = cv2.warpPerspective(img, matrix, (w, h))
    t1 = time.time()
    print()
    print("opencv")
    print("elapsed seconds per 100 images: ", t1-t0)


def transform_skimage(img, warp):
    height, width = img.shape[:2]

    t0 = time.time()
    for i in range(100):
        warp.next_sample((width, height))
        transform = warp.get_transform()
        w, h = warp.get_size()
        img1 = sktf.warp(img, transform.inverse, output_shape=(h, w))
    t1 = time.time()
    print()
    print("skimage")
    print("elapsed seconds per 100 images: ", t1-t0)


transform_tensorflow(img, warp)
time.sleep(3)
transform_opencv(img, warp)
time.sleep(3)
transform_skimage(img, warp)
