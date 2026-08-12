import numpy as np
import cv2

from src.preprocessing.degradation import (
    add_gaussian_noise,
    add_gaussian_blur,
    degrade_image,
)


def main():
    # Create a synthetic image
    image = np.zeros((256, 256), dtype=np.float32)

    # Large square
    image[70:190, 70:190] = 1.0

    # Smaller bright feature
    image[105:155, 105:155] = 0.5

    # Individual degradation stages
    noisy = add_gaussian_noise(image, sigma=0.05)

    blurred = add_gaussian_blur(
        image,
        kernel_size=5,
        sigma=1.0,
    )

    degraded = degrade_image(
        image,
        noise_sigma=0.05,
        blur_kernel_size=5,
        blur_sigma=1.0,
    )

    # Save outputs
    cv2.imwrite(
        "outputs/degraded/original.png",
        (image * 255).astype(np.uint8),
    )

    cv2.imwrite(
        "outputs/degraded/noisy.png",
        (noisy * 255).astype(np.uint8),
    )

    cv2.imwrite(
        "outputs/degraded/blurred.png",
        (blurred * 255).astype(np.uint8),
    )

    cv2.imwrite(
        "outputs/degraded/degraded.png",
        (degraded * 255).astype(np.uint8),
    )

    print("Degradation demo completed.")
    print("Saved images to outputs/degraded/")


if __name__ == "__main__":
    main()