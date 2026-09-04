"""Nettoyage classique (OpenCV, sans IA) d'une image avant prédiction :
débruitage, correction d'éclairage inégal, renforcement du contraste.

Ces filtres changent l'apparence de l'image par rapport à ce que le modèle a
vu à l'entraînement -- à activer au cas par cas et à comparer avec l'image
brute (voir app.py), pas à considérer comme systématiquement meilleur.
"""
import cv2
import numpy as np


def denoise(image_bgr: np.ndarray) -> np.ndarray:
    """Lisse le grain/bruit tout en préservant les bords (filtre bilatéral)."""
    return cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)


def flatten_illumination(image_bgr: np.ndarray, blur_fraction: float = 0.25, max_gain: float = 2.5) -> np.ndarray:
    """Corrige un éclairage inégal (ex: centre plus clair que les bords) en
    divisant l'image par une version très floutée d'elle-même (flat-field).

    Le flou doit être beaucoup plus large que la plaie elle-même, sinon il la
    traite comme une variation d'éclairage et l'efface en partie (bug observé :
    un flou fixe de 65px, bien plus petit que la bande de plaie sur une image
    ~1200-1600px, aplatissait le signal qu'on veut justement garder). Le
    noyau est donc calculé en proportion de la taille de l'image.

    Le fond noir autour du champ du microscope a une luminosité proche de 0 :
    sans plafond, le gain de correction y explose (x40 observé) et transforme
    le bruit résiduel en artefacts colorés très visibles. max_gain empêche ça.
    """
    h, w = image_bgr.shape[:2]
    ksize = int(min(h, w) * blur_fraction) | 1  # doit être impair
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    mean_bg = background.mean()
    background = np.clip(background, mean_bg / max_gain, None)
    gain = (mean_bg / background)[..., None]
    return np.clip(image_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def enhance_contrast(image_bgr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Renforce le contraste local (CLAHE sur le canal de luminance) sans
    sur-exposer l'image entière.

    Des tuiles trop petites (8x8, ~150px sur ces images) normalisent
    l'histogramme à l'intérieur même de la bande de plaie et effacent le
    contraste qui la distingue du tissu environnant. Des tuiles plus grandes
    (4x4) laissent la plaie et son voisinage dans la même tuile.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def clean_background(image_bgr: np.ndarray, black_threshold: int = 15) -> np.ndarray:
    """Met à zéro le bruit résiduel dans le fond noir autour du champ circulaire
    du microscope (n'affecte que les pixels déjà quasi noirs, ne recadre pas)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    result = image_bgr.copy()
    result[gray < black_threshold] = 0
    return result


def clean_image(
    image_bgr: np.ndarray,
    denoise_on: bool = True,
    flatten_on: bool = True,
    contrast_on: bool = True,
    clean_background_on: bool = True,
) -> np.ndarray:
    """Pipeline de nettoyage classique complet, dans un ordre qui a du sens :
    débruiter -> corriger l'éclairage -> contraste -> fond."""
    result = image_bgr
    if denoise_on:
        result = denoise(result)
    if flatten_on:
        result = flatten_illumination(result)
    if contrast_on:
        result = enhance_contrast(result)
    if clean_background_on:
        result = clean_background(result)
    return result
