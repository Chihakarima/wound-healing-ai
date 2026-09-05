"""Calcul de la surface de la plaie à partir d'un masque binaire."""
import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff


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


def hausdorff_distance_px(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float | None:
    """Distance de Hausdorff symétrique (en pixels) entre les contours du masque
    prédit et de la vérité terrain : la pire distance entre un point du contour
    d'un masque et le point le plus proche de l'autre contour. Contrairement au
    Dice/IoU (qui mesurent le recouvrement global), elle est sensible à une seule
    zone mal segmentée, même petite, ce qui la rend complémentaire pour évaluer
    la qualité du contour tracé.

    Calculée sur les contours (CHAIN_APPROX_NONE, tous les pixels de bord) plutôt
    que sur tous les pixels du masque : même résultat, beaucoup moins de points.

    Retourne None si l'un des deux masques est vide (distance non définie).
    """
    pred_contours = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    gt_contours = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not pred_contours or not gt_contours:
        return None

    pred_pts = np.vstack([c.reshape(-1, 2) for c in pred_contours])
    gt_pts = np.vstack([c.reshape(-1, 2) for c in gt_contours])

    return max(
        directed_hausdorff(pred_pts, gt_pts)[0],
        directed_hausdorff(gt_pts, pred_pts)[0],
    )


def normalize_hausdorff(hausdorff_px: float, shape_hw: tuple[int, int]) -> float:
    """Exprime une distance de Hausdorff (en pixels) en fraction de la diagonale
    de l'image sur laquelle elle a été calculée.

    Une distance en pixels bruts n'est comparable que si toutes les images
    évaluées ont la même résolution : ici les images sources font des tailles
    variées, et la baseline classique (src/baseline.py) travaille en résolution
    native pendant que l'évaluation du U-Net (src/evaluate.py) travaille sur des
    masques redimensionnés à img_size x img_size. Diviser par la diagonale rend
    les deux comparables (fraction de la plus grande distance possible dans
    l'image), au prix de perdre l'information de distance physique absolue.
    """
    h, w = shape_hw
    diagonal = (h ** 2 + w ** 2) ** 0.5
    return hausdorff_px / diagonal
