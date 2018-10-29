import numpy as np
from skimage import transform as sktf
from .transform import Translate

class FuseWarp(object):
    """FuseWarp class manages sequence of TransformSampler classes.

    Parameters
    ----------
    samplers : tuple of TransformSampler
    np_reseed : bool
        If True, call np.random.seed() in every next_sample() call, for avoiding duplication of random number
        (the duplication occurs when you use numpy.random module **with multiprocessing**).
        Note that it breaks reproducibility of randomness.
    """
    def __init__(self, samplers, np_reseed=False):
        self.samplers = samplers
        self.np_reseed = np_reseed
        self.matrix = None

        self.src_size = None
        self.dst_size = None
        self._coord_tfs = []

    def next_sample(self, src_size):
        """Resample transforms.

        Parameters
        ----------
        src_size : (int,int)
            (width, heigth)
            size of input image to transform next.
        """
        if self.np_reseed:
            np.random.seed()
        self.src_size = src_size

        # imamge center => origin
        coord_tf, size_tf = Translate(-0.5, -0.5).get_next(src_size)
        self._coord_tfs = [coord_tf]
        size = size_tf(src_size)
        # fuse transforms
        for sampler in self.samplers:
            coord_tf, size_tf = sampler.get_next(size)
            try:
                self._coord_tfs[-1] += coord_tf  # "+" leads to dot product (defined in skimage.transform.ProjectiveTransform).
            except TypeError:
                self._coord_tfs.append(coord_tf)
            size = size_tf(size)
        # origin => image top-left
        resotre_tf, size_tf = Translate(0.5, 0.5).get_next(size)
        try:
            self._coord_tfs[-1] += resotre_tf  # "+" leads to dot product (defined in skimage.transform.ProjectiveTransform).
        except NotImplementedError:
            self._coord_tfs.append(resotre_tf)
        self.dst_size = size_tf(size)

        if len(self._coord_tfs) == 1 and hasattr(self._coord_tfs[0], "__add__"):
            self.matrix = self._coord_tfs[0].params

    def get_matrix(self):
        """Return fused homography matrix that apply all the transforms.
        The matrix can be affine matrix (3rd row = (0,0,1)) when there is no projective transform (affine transform is faster than homography(projection) transform).

        Returns
        ----------
        matrix : 3x3 ndarray or None
            If there is non-Homography transform in the transforms, return None.
        """
        return self.matrix

    def get_transform(self):
        """Return fused transform that apply all the transforms.

        Returns
        ----------
        transform : skimage.transform.ProjectiveTransform or SequentialTransform
            If all the samplers are Homography transform, return ProjectiveTransform.
            Else, return SequentialTransform.
            INFO: types of each returns are subclass of skimage.tranform.GeometricTransform.
        """
        if self.matrix is None:
            return SequentialTransform(self._coord_tfs)
        else:
            return sktf.ProjectiveTransform(self.matrix)

    # TODO implement
    # def get_mesh_transform(self,mesh_size):
    # """Return pproximate transform for faster non-homography warping
    # """
    #     # init mesh
    #     # warp vertices
    #     return sktf.PiecewiseAffine(...)

    def get_size(self, cast_to_integer=True):
        """Return size of transformed image.

        Parameters
        ----------
        cast_to_integer : bool, default True
            If True, return is casted to (int,int).

        Returns
        ----------
        dst_size : (int,int) or (float,float)
        """
        if cast_to_integer:
            return tuple(map(int, map(round, self.dst_size)))
        else:
            return self.dst_size


class SequentialTransform(sktf._geometric.GeometricTransform):
    """GeometricTrnsform subclass that apply specified transforms sequentially.

    Parameters
    ----------
    transforms : tuple of skimage.transform.GeometricTransform
    """
    def __init__(self, transforms):
        self.transforms = transforms
        super().__init__()

    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) array
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Destination coordinates.
        """
        for t in self.transforms:
            coords = t(coords)
            # TODO boundary check
        return coords

    def inverse(self, coords):
        """Apply inverse transformation.

        Parameters
        ----------
        coords : (N, 2) array
            Destination coordinates.

        Returns
        -------
        coords : (N, 2) array
            Source coordinates.
        """
        for t in self.transforms[::-1]:
            coords = t.inverse(coords)
            # TODO boundary check
        return coords
