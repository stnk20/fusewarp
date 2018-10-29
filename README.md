# fusewarp

fusewarp is a helper library for image data augmentation.

fusewarp only manages transformation of image coordinates and size, not processes transformation itself.

fusewarp is designed for:
* faster image transformation
* easier handling of keypoints or bounding boxes

## Features

fusewarp is efficient as follows:

* It is not necessary to process pixels outside the output range, faster processing.

* fusewarp can "fuse" multiple transformations,
    - Pixel interpolation is only once, faster processing and better image quality (depending on interpolation method).
    - You can write a pipeline without worrying about processing speed, your code can be clearer and more flexible.

* Although based on skimage, it is easy to use with other libraries such as OpenCV, Tensorflow (CPU/GPU), etc.

## Supported transformations

* Translate
* Rotate
* Scale
* FlipLR
* FlipUD
* Shear
* Perspective
* ShiftBorder
* TrimBorder
* PadBorder
* Frame
* FrameRandom ("random cropping")
* FrameCenter ("center cropping")
* Resize
* GridPerturbation

## Randomness handling
Randomness handling of fuseaug is borrowed from [imgaug](https://github.com/aleju/imgaug) API (not exactly same):

* number : deterministic
* tuple of 2 numbers : uniform random
* list of numbers : random choice from elements
* function/generator/iterator : use returned number (ex. random functions or infinite generator)

## Installation

```
pip install git+https://github.com/stnk20/fusewarp
```

Dependency:

* numpy
* skimage

## Example usage

```
## build pipeline
warp = fw.FuseWarp([
    ft.Scale((0.8, 1.2), resize=True),
    ft.Rotate((-10, 10)),
    ft.FlipLR(),
    ft.FrameRandom(300, 300)])

## do random sampling in image augmentation loop
for i in range(10):
    warp.next_sample((src_width, src_height))
    matrix = warp.get_matrix() 
    dst_width, dst_height = warp.get_size()

    ## you can transform images with any library that support homography transformation. for example, opencv, skimage, tensorflow, etc.
    dst_img = cv2.warpPerspective(src_img, matrix, (dst_width, dst_height))

    ## map keypoints
    ## format: ndarray, shape = (N,2)
    dst_keypoints = transform(src_keypoints)

```

## Note

* There is almost no range cheking of parameters in this module, so please take care of them. Especially for operations that change image size.
* Subpixel level accuracy may be lost.
* General transformation other than Homography transformation is supported experimentally.

## Extendability

You can extend fusewarp by creating a subclass of TransformSampler class.
Please try to read fusewarp's code for implementation details.(I wrote it as simple as possible).
