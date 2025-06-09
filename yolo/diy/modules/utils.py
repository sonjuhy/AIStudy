import torch
import torch.nn as nn
import numpy as np
import cv2
import tqdm
import os
import warnings

from typing import List, Optional


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


if TQDM_RICH := str(os.getenv("YOLO_TQDM_RICH", False)).lower() == "true":
    from tqdm import rich

VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format


class TQDM(rich.tqdm if TQDM_RICH else tqdm.tqdm):
    """
    A custom TQDM progress bar class that extends the original tqdm functionality.

    This class modifies the behavior of the original tqdm progress bar based on global settings and provides
    additional customization options for Ultralytics projects. The progress bar is automatically disabled when
    VERBOSE is False or when explicitly disabled.

    Attributes:
        disable (bool): Whether to disable the progress bar. Determined by the global VERBOSE setting and
            any passed 'disable' argument.
        bar_format (str): The format string for the progress bar. Uses the global TQDM_BAR_FORMAT if not
            explicitly set.

    Methods:
        __init__: Initialize the TQDM object with custom settings.
        __iter__: Return self as iterator to satisfy Iterable interface.

    Examples:
        >>> from ultralytics.utils import TQDM
        >>> for i in TQDM(range(100)):
        ...     # Your processing code here
        ...     pass
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a custom TQDM progress bar with Ultralytics-specific settings.

        Args:
            *args (Any): Variable length argument list to be passed to the original tqdm constructor.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the original tqdm constructor.

        Notes:
            - The progress bar is disabled if VERBOSE is False or if 'disable' is explicitly set to True in kwargs.
            - The default bar format is set to TQDM_BAR_FORMAT unless overridden in kwargs.

        Examples:
            >>> from ultralytics.utils import TQDM
            >>> for i in TQDM(range(100)):
            ...     # Your code here
            ...     pass
        """
        warnings.filterwarnings(
            "ignore", category=tqdm.TqdmExperimentalWarning
        )  # suppress tqdm.rich warning
        kwargs["disable"] = not VERBOSE or kwargs.get("disable", False)
        kwargs.setdefault(
            "bar_format", TQDM_BAR_FORMAT
        )  # override default value if passed
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """Return self as iterator to satisfy Iterable interface."""
        return super().__iter__()


def imread(filename: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Read an image from a file with multilanguage filename support.

    Args:
        filename (str): Path to the file to read.
        flags (int, optional): Flag that can take values of cv2.IMREAD_*. Controls how the image is read.

    Returns:
        (np.ndarray | None): The read image array, or None if reading fails.

    Examples:
        >>> img = imread("path/to/image.jpg")
        >>> img = imread("path/to/image.jpg", cv2.IMREAD_GRAYSCALE)
    """
    file_bytes = np.fromfile(filename, np.uint8)
    if filename.endswith((".tiff", ".tif")):
        success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
        if success:
            # Handle RGB images in tif/tiff format
            return (
                frames[0]
                if len(frames) == 1 and frames[0].ndim == 3
                else np.stack(frames, axis=2)
            )
        return None
    else:
        im = cv2.imdecode(file_bytes, flags)
        return im[..., None] if im.ndim == 2 else im  # Always ensure 3 dimensions
