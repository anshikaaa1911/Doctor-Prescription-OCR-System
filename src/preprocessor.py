"""Image preprocessing pipeline for prescription OCR."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PreprocessMetadata(TypedDict):
    """Metadata produced by preprocessing."""

    original_shape: tuple[int, ...]
    processed_shape: tuple[int, ...]
    grayscale_applied: bool
    clahe_applied: bool
    denoise_applied: bool
    deskew_applied: bool
    skew_angle_degrees: float
    adaptive_threshold_applied: bool
    morphology_applied: bool
    confidence: float


class PreprocessResult(TypedDict):
    """Preprocessing result."""

    image: np.ndarray
    metadata: PreprocessMetadata


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration.

    Args:
        config_path: Optional path to a YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    path = config_path or Path.cwd() / "config.yaml"
    if not path.exists():
        logger.warning("Config file not found at %s; using defaults.", path)
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML is not installed; using default configuration.")
        return {}
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return loaded if isinstance(loaded, dict) else {}


def load_image(image_path: Path) -> np.ndarray:
    """Load an image from disk with OpenCV.

    Args:
        image_path: Path to the image file.

    Returns:
        Loaded BGR image.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If OpenCV cannot decode the image.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")
    return image


def resize_image(image: np.ndarray, width: int | None) -> np.ndarray:
    """Resize an image while preserving aspect ratio.

    Args:
        image: Input image.
        width: Target width, or None to keep the original size.

    Returns:
        Resized image.
    """
    if width is None or width <= 0 or image.shape[1] == width:
        return image.copy()
    ratio = width / float(image.shape[1])
    height = max(1, int(round(image.shape[0] * ratio)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale.

    Args:
        image: BGR or grayscale input image.

    Returns:
        Grayscale image.
    """
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE contrast enhancement.

    Args:
        image: Grayscale input image.
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE tile grid size.

    Returns:
        Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def remove_noise(image: np.ndarray, enabled: bool = True) -> np.ndarray:
    """Remove image noise using non-local means denoising.

    Args:
        image: Grayscale input image.
        enabled: Whether denoising should be applied.

    Returns:
        Denoised image when enabled; otherwise a copy of the input.
    """
    if not enabled:
        return image.copy()
    return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)


def _normalise_hough_angle(theta: float) -> float:
    """Convert a Hough line theta value into a document skew angle.

    Args:
        theta: Hough line angle in radians.

    Returns:
        Skew angle in degrees.
    """
    angle = math.degrees(theta) - 90.0
    if angle < -45.0:
        angle += 90.0
    if angle > 45.0:
        angle -= 90.0
    return angle


def estimate_skew_angle(image: np.ndarray) -> float:
    """Estimate document skew angle with a Hough transform.

    Args:
        image: Grayscale image.

    Returns:
        Estimated skew angle in degrees. Zero means no reliable skew found.
    """
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(80, image.shape[1] // 8))
    if lines is None:
        return 0.0

    angles = [_normalise_hough_angle(float(line[0][1])) for line in lines[:50]]
    filtered = [angle for angle in angles if abs(angle) <= 30.0]
    if not filtered:
        return 0.0
    return float(np.median(filtered))


def deskew_image(image: np.ndarray, enabled: bool = True) -> tuple[np.ndarray, float]:
    """Deskew an image using Hough-transform angle estimation.

    Args:
        image: Grayscale input image.
        enabled: Whether deskewing should be applied.

    Returns:
        Tuple containing the deskewed image and detected skew angle.
    """
    if not enabled:
        return image.copy(), 0.0

    angle = estimate_skew_angle(image)
    if abs(angle) < 0.1:
        return image.copy(), 0.0

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def adaptive_threshold(image: np.ndarray, block_size: int = 31, c_value: int = 11) -> np.ndarray:
    """Apply adaptive thresholding for uneven illumination.

    Args:
        image: Grayscale input image.
        block_size: Odd neighborhood size used for local thresholding.
        c_value: Constant subtracted from the local threshold.

    Returns:
        Binary image.
    """
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )


def apply_morphology(image: np.ndarray, kernel_size: tuple[int, int] = (2, 2)) -> np.ndarray:
    """Clean thresholded text with morphology.

    Args:
        image: Binary input image.
        kernel_size: Morphological kernel size.

    Returns:
        Morphologically cleaned image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)


def calculate_preprocess_confidence(image: np.ndarray, skew_angle: float) -> float:
    """Calculate a heuristic image-readiness confidence score.

    Args:
        image: Processed binary image.
        skew_angle: Detected skew angle before deskewing.

    Returns:
        Confidence score in the range 0.0 to 1.0.
    """
    foreground_ratio = float(np.mean(image < 128))
    foreground_score = 1.0 - min(abs(foreground_ratio - 0.12) / 0.25, 1.0)
    contrast_score = min(float(np.std(image)) / 90.0, 1.0)
    skew_score = 1.0 - min(abs(skew_angle) / 30.0, 1.0)
    score = (foreground_score * 0.4) + (contrast_score * 0.4) + (skew_score * 0.2)
    return round(max(0.0, min(score, 1.0)), 3)


def preprocess_array(image: np.ndarray, config: dict[str, Any] | None = None) -> PreprocessResult:
    """Run the full preprocessing pipeline for an in-memory image.

    Args:
        image: Input BGR or grayscale image.
        config: Optional configuration dictionary.

    Returns:
        Processed image and confidence metadata.
    """
    settings = (config or {}).get("preprocessing", {})
    resized = resize_image(image, settings.get("resize_width", 1600))
    gray = to_grayscale(resized)
    tile_size = tuple(settings.get("clahe_tile_grid_size", [8, 8]))
    enhanced = apply_clahe(gray, float(settings.get("clahe_clip_limit", 2.0)), (int(tile_size[0]), int(tile_size[1])))
    denoised = remove_noise(enhanced, bool(settings.get("denoise", True)))
    deskewed, skew_angle = deskew_image(denoised, bool(settings.get("deskew", True)))
    thresholded = adaptive_threshold(
        deskewed,
        int(settings.get("adaptive_block_size", 31)),
        int(settings.get("adaptive_c", 11)),
    )
    kernel_values = tuple(settings.get("morphology_kernel_size", [2, 2]))
    processed = apply_morphology(thresholded, (int(kernel_values[0]), int(kernel_values[1])))
    confidence = calculate_preprocess_confidence(processed, skew_angle)

    metadata: PreprocessMetadata = {
        "original_shape": tuple(image.shape),
        "processed_shape": tuple(processed.shape),
        "grayscale_applied": True,
        "clahe_applied": True,
        "denoise_applied": bool(settings.get("denoise", True)),
        "deskew_applied": bool(settings.get("deskew", True)),
        "skew_angle_degrees": round(skew_angle, 3),
        "adaptive_threshold_applied": True,
        "morphology_applied": True,
        "confidence": confidence,
    }
    logger.info("Preprocessing complete with confidence %.3f", confidence)
    return {"image": processed, "metadata": metadata}


def preprocess_image(image_path: Path, config_path: Path | None = None) -> PreprocessResult:
    """Load and preprocess an image from disk.

    Args:
        image_path: Path to input image.
        config_path: Optional config path.

    Returns:
        Processed image and metadata.
    """
    config = load_config(config_path)
    image = load_image(image_path)
    return preprocess_array(image, config)
