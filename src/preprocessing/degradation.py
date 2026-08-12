import numpy as np
import cv2

def add_gaussian_noise(image: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image with pixel values in [0, 1].

    sigma : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    np.ndarray
        Noisy image with values clipped to [0, 1].
    """

    noise = np.random.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape
    )

    noisy_image = image + noise

    return np.clip(noisy_image, 0.0, 1.0)
def add_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian blur to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image with pixel values in [0, 1].

    kernel_size : int
        Size of the Gaussian kernel. Must be a positive odd number.

    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Blurred image with values clipped to [0, 1].
    """

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number.")

    blurred_image = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma
    )

    return np.clip(blurred_image, 0.0, 1.0)
def degrade_image(
    image: np.ndarray,
    noise_sigma: float = 0.05,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.0
) -> np.ndarray:
    """
    Apply a combined degradation pipeline.

    The image is first blurred and then Gaussian noise is added.

    Parameters
    ----------
    image : np.ndarray
        Clean image with pixel values in [0, 1].

    noise_sigma : float
        Standard deviation of Gaussian noise.

    blur_kernel_size : int
        Size of the Gaussian blur kernel.

    blur_sigma : float
        Standard deviation of the Gaussian blur.

    Returns
    -------
    np.ndarray
        Degraded image with values clipped to [0, 1].
    """

    degraded = add_gaussian_blur(
        image,
        kernel_size=blur_kernel_size,
        sigma=blur_sigma
    )

    degraded = add_gaussian_noise(
        degraded,
        sigma=noise_sigma
    )

    return np.clip(degraded, 0.0, 1.0)