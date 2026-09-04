"""Calcul de la surface de la plaie à partir d'un masque binaire."""
import cv2
import numpy as np


def wound_area_px(mask: np.ndarray) -> int:
    """Nombre de pixels appartenant à la plaie (mask > 0)."""
    return int((mask > 0).sum())


def pixels_per_cm_from_reference(
    mask_or_image: np.ndarray,
    reference_length_cm: float,
    reference_length_px: float,
) -> float:
    """Convertit une longueur de référence connue (ex: réglette visible sur la
    photo) en ratio pixels/cm, utilisable ensuite pour convertir une surface.

    reference_length_px: longueur en pixels de l'objet de référence, mesurée
    manuellement (ex: avec cv2.selectROI) ou automatiquement si l'objet est
    détectable (ex: pastille de couleur connue).
    """
    return reference_length_px / reference_length_cm


def area_px_to_cm2(area_px: int, pixels_per_cm: float) -> float:
    """Convertit une surface en pixels^2 vers cm^2, connaissant le ratio px/cm."""
    return area_px / (pixels_per_cm ** 2)


def extract_largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def extract_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Tous les contours externes du masque (pas seulement le plus grand) :
    la plaie peut être fragmentée en plusieurs morceaux disjoints, tous
    comptés dans la surface, donc tous doivent apparaître dans le contour
    affiché pour que le tracé corresponde au masque."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)
