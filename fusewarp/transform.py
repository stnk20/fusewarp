import random
import numpy as np
import skimage.transform as sktf

class NumberSampler(object):
    """Generate callable sampler.
    If you pass callablel/generator/sampler, please take care to keep returned numbers in tractable range.

    Parameters
    ----------
    x : number / (number,number) / [number,number,...] / callable object / generator / iterator
        Treated as below (number means int/float):
        * number : deterministic
        * tuple of 2 numbers : random sampling from uniform range [a,b)
        * list of numbers : random sampling from each elements
        * callable : use number returned by x() (function call)
        * generator/iterator : use number returned by next(x)
    """
    def __init__(self, x):
        if callable(x):
            self.sampler = x
        elif isinstance(x, tuple):
            self.sampler = lambda: x[0]+(x[1]-x[0])*random.random()
        elif isinstance(x, list):
            self.sampler = lambda: random.choice(x)
        elif isinstance(x, (float, int)):
            self.sampler = lambda: x
        elif hasattr(x, "__next__"):
            self.sampler = lambda: next(x)
        else:
            raise ValueError("invalid type: x must be int/float/tuple/list/function/generator/iterator")

    def __call__(self):
        return self.sampler()

class TransformSampler(object):
    """
    Base calss for transform samplers.
    """
    def __init__(self):
        pass

    def get_next(self, size):
        """Sample some parameters to generate next transformations (for coordinate and size), return them.

        Parameters
        ----------
        size : (int,int) or (float,float)
            Size of image passed to this sampler.

        Returns
        ----------
        transforms : ( skimage.transform.GeometrictTransform, function((float,float))->(float,float) )
        """
        raise NotImplementedError()

class Translate(TransformSampler):
    """Translate images.

    Parameters
    ----------
    x : acceptable types of NumberSampler()
        x component of offset vector.
    y : acceptable types of NumberSampler()
        y component of offset vector.
    relative_pos : bool
        If True, x/y is interpreted as relative value to width/height.
        If False, x/y is interpreted as pixel.
    """
    def __init__(self, x, y, relative_pos=True):
        self.x_s = NumberSampler(x)
        self.y_s = NumberSampler(y)
        self.relative_pos = relative_pos

    def get_next(self, size):
        x = self.x_s()
        y = self.y_s()
        if self.relative_pos:
            x *= size[0]
            y *= size[1]
        return sktf.AffineTransform(translation=(x, y)), lambda s: s

class Rotate(TransformSampler):
    """Rotate images.

    Parameters
    ----------
    theta : acceptable types of NumberSampler()
        Rotation angle (degrees).
    """
    def __init__(self, theta):
        self.theta_s = NumberSampler(theta)

    def get_next(self, size):
        theta = np.deg2rad(self.theta_s())
        return sktf.AffineTransform(rotation=theta), lambda s: s

class Scale(TransformSampler):
    """Scale images.

    Parameters
    ----------
    sx : acceptable types of NumberSampler()
        x component of scale vector. sx>1 means enlarge.
    sy : acceptable types of NumberSampler() or None
        y component of scale vector. sy>1 means enlarge.
        If None, sy = sx.
    resize : bool
        If True, size is changed to scaled image.
        If False, size is not changed.
        INFO: if you need specify *absolute* size of output image, use Resize class.
    """
    def __init__(self, sx, sy=None, resize=False):
        self.sx_s = NumberSampler(sx)
        if sy is None:
            self.sy_s = None
        else:
            self.sy_s = NumberSampler(sy)
        self.resize = resize

    def get_next(self, size):
        if self.sy_s is None:
            sx = self.sx_s()
            sy = sx
        else:
            sx = self.sx_s()
            sy = self.sy_s()
        if self.resize:
            size_transform = lambda s: (sx*s[0], sy*s[1])
        else:
            size_transform = lambda s: s
        return sktf.AffineTransform(scale=(sx, sy)), size_transform

class FlipLR(Scale):
    """Flip left-right."""
    def __init__(self):
        super().__init__([-1, 1], 1)

class FlipUD(Scale):
    """Flip up-down."""
    def __init__(self):
        super().__init__(1, [-1, 1])

class Shear(TransformSampler):
    """Shear images.
    The transformation matrix is below (not same with skimage).
        [[ 1  ix   0]
         [iy   1   0]
         [ 0   0   1]]

    Parameters
    ----------
        ix : acceptable types of NumberSampler()
            x component of shear intensity.
        iy : acceptable types of NumberSampler()
            y component of shear intensity.
    """
    def __init__(self, ix, iy):
        self.ix_s = NumberSampler(ix)
        self.iy_s = NumberSampler(iy)
    def get_next(self, size):
        ix = self.ix_s()
        iy = self.iy_s()
        M = np.array([[1, ix, 0], [iy, 1, 0], [0, 0, 1]], dtype=np.float32)
        return sktf.AffineTransform(matrix=M), lambda s: s

class Perspective(TransformSampler):
    """Emphasis perspective.
    The transformation matrix is:
        [[ 1   0   0]
         [ 0   1   0]
         [ a   b   1]]
        where,
        c = 1/max(w,h)
        a = c*intensity*cos(theta)
        b = c*intensity*sin(theta)

    Parameters
    ----------
        theta : acceptable types of NumberSampler()
            Direction of perspective farside (degrees).
        intensity : acceptable types of NumberSampler()
            Intensity of perspective, typical range is about 0 to 2.
    """
    def __init__(self, theta, intensity):
        self.t_s = NumberSampler(theta)
        self.i_s = NumberSampler(intensity)

    def get_next(self, size):
        theta = np.deg2rad(self.t_s())
        intensity = self.i_s()
        c = 1/max(size)
        a = c*intensity*np.cos(theta)
        b = c*intensity*np.sin(theta)
        M = np.array([[1, 0, 0], [0, 1, 0], [a, b, 1]], dtype=np.float32)
        return sktf.ProjectiveTransform(matrix=M), lambda s: s

class ShiftBorder(TransformSampler):
    """Pad or trim each border of images (bx>0 or by>0 means padding, and bx<0 or by<0 means trimming).

    Parameters
    ----------
        bx : acceptable types of NumberSampler()
            x component of padding/trimming.
        by : acceptable types of NumberSampler()
            y component of padding/trimming.
        relative_pos : bool
            If True, bx/by is interpreted as relative value to width/height.
        sign : None or +1 or -1
            If None, do nothing.
            If +1, bx/by is clipped to positive.
            If -1, bx/by is negated, then clipped to negative.
    """
    def __init__(self, bx, by, relative_pos, sign=None):
        self.bx_s = NumberSampler(bx)
        self.by_s = NumberSampler(by)
        self.relative_pos = relative_pos
        self.sign = sign

    def get_next(self, size):
        bx = self.bx_s()
        by = self.by_s()
        if self.relative_pos:
            bx *= size[0]
            by *= size[1]
        if self.sign is not None:
            if self.sign > 0:
                bx = max(0, bx)
                by = max(0, by)
            elif self.sign < 0:
                bx = min(0, -bx)
                by = min(0, -by)
            else:
                pass
        return sktf.AffineTransform(), lambda s: (s[0]+bx*2, s[1]+by*2)

class TrimBorder(ShiftBorder):
    """Trim each border of images.
    Note that this transformation does not touch the pixel values, so TrimBorder then PadBorder yields identity transformation.
    Wrapper subclass of ShiftBorder.

    Parameters
    ----------
        bx : acceptable types of NumberSampler()
            x component of trimming.
        by : acceptable types of NumberSampler()
            y component of trimming.
        relative_pos : bool
            If True, bx/by is interpreted as relative value to width/height.
    """
    def __init__(self, bx, by, relative_pos=True):
        super().__init__(bx, by, sign=-1, relative_pos=relative_pos)

class PadBorder(ShiftBorder):
    """Pad each border of images.
    Wrapper subclass of ShiftBorder.

    Parameters
    ----------
        bx : acceptable types of NumberSampler()
            x component of padding.
        by : acceptable types of NumberSampler()
            y component of padding.
        relative_pos : bool
            If True, bx/by is interpreted as relative value to width/height.
    """
    def __init__(self, bx, by, relative_pos=True):
        super().__init__(bx, by, sign=1, relative_pos=relative_pos)

class Frame(TransformSampler):
    """Frame images with specified offset and size.
    This transformation is equivalent or similar to "set ROI" or "cropping".
    Note that the input images can be smaller than the output images, they are automatically padded.
    Parameters
    ----------
        x : acceptable types of NumberSampler()
            x component of offset.
        y : acceptable types of NumberSampler()
            y component of offset.
        w : int/float
            width of output image.
        h : int/float
            height of output image.
        relative_pos : bool
            If True, bx/by is interpreted as relative value to width/height.
        valid_region : bool
            If True, -0.5<bx<0.5/-0.5<by<0.5 is mapped to valid x/y region (normalization factors vary with input size and output size)
            If False, bx/by is simply scaled by inverse of width/height (same as Translate with relative_pos=True).
            This parameter is effective only if relative_pos=True.
    """
    def __init__(self, w, h, x, y, relative_pos=True, valid_region=True):
        self.w_s = NumberSampler(w)
        self.h_s = NumberSampler(h)
        self.x_s = NumberSampler(x)
        self.y_s = NumberSampler(y)
        self.relative_pos = relative_pos
        self.valid_region = valid_region

    def get_next(self, size):
        w = self.w_s()
        h = self.h_s()
        x = self.x_s()
        y = self.y_s()
        if self.relative_pos:
            if self.valid_region:
                x *= size[0]-w
                y *= size[1]-h
            else:
                x *= size[0]
                y *= size[1]
        return sktf.AffineTransform(translation=(x, y)), lambda s: (w, h)

class FrameRandom(Frame):
    """Frame images with specified size, similar to a "random cropping".
    The offset of framing is sampled from unifom random in "valid" range.
    Wrapper subclass of Frame.

    Parameters
    ----------
        w : int/float
            width of output image.
        h : int/float
            height of output image.
    """
    def __init__(self, w, h):
        super().__init__(w, h, (-0.5, 0.5), (-0.5, 0.5))

class FrameCenter(Frame):
    """Frame images with specified size, similar to a "center cropping".
    Wrapper subclass of Frame.

    Parameters
    ----------
        w : int/float
            width of output image.
        h : int/float
            height of output image.
    """
    def __init__(self, w, h):
        super().__init__(w, h, 0, 0)

class Resize(TransformSampler):
    """Resize to specified size.

    Parameters
    ----------
        w : int/float
            width of output image.
        h : int/float
            height of output image.
        keep_aspect_ratio : bool
            If True, do not change aspect ratio.
    """
    def __init__(self, w, h, keep_aspect_ratio=True):
        self.w_s = NumberSampler(w)
        self.h_s = NumberSampler(h)
        self.keep_aspect_ratio = keep_aspect_ratio

    def get_next(self, size):
        w = self.w_s()
        h = self.h_s()
        sx = w/size[0]
        sy = h/size[1]
        if self.keep_aspect_ratio:
            s = min(sx, sy)
            sx = s
            sy = s
        return sktf.AffineTransform(scale=(sx, sy)), lambda s: (w, h)

class GridPerturbation(TransformSampler):
    """Map image to uniform grid and add perturbation to it.

    Parameters
    ----------
        perturbation : acceptable types of NumberSampler()
            perturbation of each grid, interpreted as relative to grid size.
        num_grid : (int, int)
            number of grid points in x and y direction (includes each side).
    """
    def __init__(self, perturbation=(-0.1,0.1), num_grid=(5,5)):
        self.perturbation = NumberSampler(perturbation)
        self.num_grid = num_grid

    def get_next(self, size):
        nx, ny = self.num_grid
        gy, gx = np.meshgrid(size[1]*np.linspace(-0.5, 0.5, ny), size[0]*np.linspace(-0.5, 0.5, nx))
        g = np.stack([gx.flatten(), gy.flatten()], axis=-1).astype(np.float32)
        g1 = np.copy(g)
        for i in range(g.shape[0]):
            g1[i, 0] = g[i, 0]+size[0]/(nx-1)*self.perturbation()
            g1[i, 1] = g[i, 1]+size[1]/(ny-1)*self.perturbation()
        t = sktf.PiecewiseAffineTransform()
        t.estimate(g, g1)
        return t, lambda s: s
