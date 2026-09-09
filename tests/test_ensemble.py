import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ensemble import carte_accord, masque_consensus, overlay_confiance, pourcentage_zone_incertaine


def _mask(valeurs):
    return np.array(valeurs, dtype=np.uint8)


def test_masque_consensus_unanime():
    m = _mask([1, 1, 1])
    masks = [m.copy(), m.copy(), m.copy()]
    assert np.array_equal(masque_consensus(masks), m)


def test_masque_consensus_vote_majoritaire():
    # pixel 0: 3/3 votent plaie -> plaie ; pixel 1: 1/3 -> pas plaie ; pixel 2: 2/3 -> plaie
    masks = [
        _mask([1, 0, 1]),
        _mask([1, 1, 1]),
        _mask([1, 0, 0]),
    ]
    assert list(masque_consensus(masks)) == [1, 0, 1]


def test_carte_accord_compte_les_votes():
    masks = [_mask([1, 0]), _mask([1, 1]), _mask([0, 0])]
    assert list(carte_accord(masks)) == [2, 1]


def test_pourcentage_zone_incertaine_desaccord_partiel():
    # 4 pixels dans l'union (au moins un modèle vote plaie) : 2 unanimes, 2 en désaccord
    masks = [
        np.array([1, 1, 0, 1], dtype=np.uint8),
        np.array([1, 0, 0, 1], dtype=np.uint8),
        np.array([1, 0, 1, 1], dtype=np.uint8),
    ]
    # union = [1,1,1,1] (4 pixels), désaccord sur pixels 1 et 2 (votes 1 et 1, ni 0 ni 3)
    assert pourcentage_zone_incertaine(masks) == pytest.approx(50.0)


def test_pourcentage_zone_incertaine_aucune_detection():
    masks = [np.zeros((5, 5), dtype=np.uint8) for _ in range(3)]
    assert pourcentage_zone_incertaine(masks) == 0.0


def test_pourcentage_zone_incertaine_unanime_est_zero():
    m = np.ones((3, 3), dtype=np.uint8)
    masks = [m.copy(), m.copy(), m.copy()]
    assert pourcentage_zone_incertaine(masks) == 0.0


def test_overlay_confiance_ne_modifie_pas_le_fond():
    image = np.full((4, 4, 3), 100, dtype=np.uint8)
    carte = np.zeros((4, 4), dtype=np.uint8)  # aucun modèle ne détecte de plaie nulle part
    overlay = overlay_confiance(image, carte, n_modeles=3)
    assert np.array_equal(overlay, image)


def test_overlay_confiance_teinte_zone_accord_et_desaccord():
    image = np.full((2, 2, 3), 100, dtype=np.uint8)
    carte = np.array([[3, 1], [0, 0]], dtype=np.uint8)  # accord total, désaccord, fond, fond
    overlay = overlay_confiance(image, carte, n_modeles=3)
    assert not np.array_equal(overlay[0, 0], image[0, 0])  # zone d'accord teintée
    assert not np.array_equal(overlay[0, 1], image[0, 1])  # zone de désaccord teintée
    assert np.array_equal(overlay[1, 0], image[1, 0])       # fond inchangé
    assert np.array_equal(overlay[1, 1], image[1, 1])       # fond inchangé
