import xarray as xr
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


def compute_output_size(pool_width, stride, in_dim,
                        kernel_size, n_filters):
    """
    Compute output of conv. layers

    output = (input - filter + 1) / stride
    """
    for i, k in enumerate(kernel_size):

        in_dim = (in_dim - k[0] + 1) / stride
        in_dim = in_dim // pool_width

    return int(in_dim * in_dim * n_filters[-1])


def apply_transforms(
    X: xr.DataArray,
    fraction: float = None,
    trfs: transforms.transforms.Compose = None,
    verbose: bool = False,
):
    """
    Generate new sample images by applying a set of transformations.

    Inputs:
    ------
    X: np.ndarray
        Input dataset of size (frames, height, width, depth)
    fraction: float | None
        Fraction of frames to apply the transformations.
        The frames are chosen randomly. If None all frames
        are used.
    trfs: torchvision.transforms.transforms.Compose | None
        Composition of torchvision transformations.

    Outputs:
    -------
        Transformed frames
    """

    # Choose frames to use
    if isinstance(fraction, float):

        n_frames = int(fraction * X.sizes["frames"])

        frames_to_use = np.random.choice(
            np.arange(0, X.sizes["frames"], dtype=int),
            size=n_frames, replace=False
        )
    else:
        frames_to_use = np.arange(0, X.sizes["frames"], dtype=int)

    # Get labels for selected frames
    y_trf = X.frames.data[frames_to_use]

    # Iterator
    _iter = tqdm(frames_to_use) if verbose else frames_to_use

    X_trf = [np.array(trfs(Image.fromarray(X.data[f]))) for f in _iter]

    return np.transpose(X_trf, (0, 3, 1, 2)), y_trf
