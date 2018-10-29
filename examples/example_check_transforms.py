"""
check transformations
"""
import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage import transform as sktf

import fusewarp as fw
from fusewarp import transform as ft

def check_transforom(description, warp, rows=4, cols=4):
    img0 = skimage.data.rocket()
    keypoints0 = np.array([[323, 128], [202, 126], [447, 126], [198, 217], [451, 217]], dtype=np.float32)
    height, width = img0.shape[:2]

    fig = plt.figure(figsize=(100,100))
    plt.suptitle(description)
    for i in range(rows*cols):
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

        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img1)
        plt.scatter(keypoints1[:, 0], keypoints1[:, 1])
    plt.show()

conditions = [
    ["Translate(0.2, 0)", fw.FuseWarp([ ft.Translate(0.2, 0) ])],
    ["Translate(0, 0.2)", fw.FuseWarp([ ft.Translate(0, 0.2) ])],
    ["Translate((-0.2, 0.2), (-0.2, 0.2))", fw.FuseWarp([ ft.Translate((-0.2, 0.2), (-0.2, 0.2)) ])],
    ["Rotate(30)", fw.FuseWarp([ ft.Rotate(30) ])],
    ["Rotate((-30, 30))", fw.FuseWarp([ ft.Rotate((-30, 30)) ])],
    ["Scale(1.5)", fw.FuseWarp([ ft.Scale(1.5) ])],
    ["Scale(0.5, 1.5)", fw.FuseWarp([ ft.Scale(0.5, 1.5) ])],
    ["Scale((0.5, 1.5))", fw.FuseWarp([ ft.Scale((0.5, 1.5)) ])],
    ["Scale((0.5, 1.5), sy=None, resize=True)", fw.FuseWarp([ ft.Scale((0.5, 1.5), sy=None, resize=True) ])],
    ["FlipLR()", fw.FuseWarp([ ft.FlipLR() ])],
    ["FlipUD()", fw.FuseWarp([ ft.FlipUD() ])],
    ["Shear(0.2, 0)", fw.FuseWarp([ ft.Shear(0.2, 0) ])],
    ["Shear(0, 0.2)", fw.FuseWarp([ ft.Shear(0, 0.2) ])],
    ["Shear((-0.2, 0.2), (-0.2, 0.2))", fw.FuseWarp([ ft.Shear((-0.2, 0.2), (-0.2, 0.2)) ])],
    ["Perspective(90, 0.5)", fw.FuseWarp([ ft.Perspective(90, 0.5) ])],
    ["Perspective((-180, 180), (0, 0.5))", fw.FuseWarp([ ft.Perspective((-180, 180), (0, 0.5)) ])],
    ["ShiftBorder(-0.2, 0, relative_pos=True)", fw.FuseWarp([ ft.ShiftBorder(-0.2, 0, relative_pos=True) ])],
    ["ShiftBorder( 0.2, 0, relative_pos=True)", fw.FuseWarp([ ft.ShiftBorder( 0.2, 0, relative_pos=True) ])],
    ["ShiftBorder(0, -0.2, relative_pos=True)", fw.FuseWarp([ ft.ShiftBorder(0, -0.2, relative_pos=True) ])],
    ["ShiftBorder(0,  0.2, relative_pos=True)", fw.FuseWarp([ ft.ShiftBorder(0,  0.2, relative_pos=True) ])],
    ["TrimBorder((0, 0.2), (0, 0.2))", fw.FuseWarp([ ft.TrimBorder((0, 0.2), (0, 0.2)) ])],
    ["PadBorder((0, 0.2), (0, 0.2))", fw.FuseWarp([ ft.PadBorder((0, 0.2), (0, 0.2)) ])],
    ["Frame(300, 200, 0.1, 0.1)", fw.FuseWarp([ ft.Frame(300, 200, 0.1, 0.1) ])],
    ["FrameRandom(300, 200)", fw.FuseWarp([ ft.FrameRandom(300, 200) ])],
    ["FrameCenter(300, 200)", fw.FuseWarp([ ft.FrameCenter(300, 200) ])],
    ["Frame(300, 600, 0.1, 0.1)", fw.FuseWarp([ ft.Frame(300, 600, 0.1, 0.1) ])],
    ["FrameRandom(300, 600)", fw.FuseWarp([ ft.FrameRandom(300, 600) ])],
    ["FrameCenter(300, 600)", fw.FuseWarp([ ft.FrameCenter(300, 600) ])],
    ["Resize(200, 200)", fw.FuseWarp([ ft.Resize(200, 200) ])],
    ["Resize(200, 200, keep_aspect_ratio=False)", fw.FuseWarp([ ft.Resize(200, 200, keep_aspect_ratio=False) ])],
    # below cases are relatively slow, because these are non homography transformation.
    ["GridPerturbation(0.5)", fw.FuseWarp([ ft.GridPerturbation(0.5) ])],
    ["GridPerturbation((-0.1, 0.1))", fw.FuseWarp([ ft.GridPerturbation((-0.1, 0.1)) ])],
    ["GridPerturbation((-0.1, 0.1), num_grid=(3,3))", fw.FuseWarp([ ft.GridPerturbation((-0.1, 0.1), num_grid=(3,3)) ])],
]

for description, warp in conditions:
    check_transforom(description, warp)