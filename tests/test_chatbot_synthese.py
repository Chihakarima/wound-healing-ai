"""Vérifie que les chiffres injectés dans le prompt du résumé scientifique
(synthèse globale + vitesses par intervalle) correspondent aux mesures fournies,
sans jamais dépendre du LLM. Couvre la régression corrigée dans chatbot.py :
le LLM inventait des pourcentages/vitesses différents des mesures réelles quand
on le laissait les calculer lui-même."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chatbot"))
from chatbot import (
    _calculer_intervalles,
    _calculer_synthese,
    _format_intervalles,
    _format_synthese,
    _phrase_heterogeneite,
    _phrase_interpretation,
    _phrase_reouverture,
)

ROWS = [
    {"time_h": 0, "area_px2": 8410, "closure_pct": 0.0},
    {"time_h": 24, "area_px2": 5063, "closure_pct": 40.0},
    {"time_h": 48, "area_px2": 2407, "closure_pct": 71.6},
]

# Cas réel signalé par le biologiste (2026-09-09) : la plaie se referme puis se
# rouvre partiellement -- delta_closure_pct négatif sur le 2e intervalle.
ROWS_REOUVERTURE = [
    {"time_h": 0, "area_px2": 94229, "closure_pct": 0.0},
    {"time_h": 24, "area_px2": 73791, "closure_pct": 21.7},
    {"time_h": 48, "area_px2": 92521, "closure_pct": 1.8},
]

# Mêmes mesures que ROWS_REOUVERTURE mais dans l'ordre chronologique correct
# (sans réouverture) : les deux intervalles sont positifs (0,07%/h puis 0,83%/h).
ROWS_SANS_REOUVERTURE = [
    {"time_h": 0, "area_px2": 94229, "closure_pct": 0.0},
    {"time_h": 24, "area_px2": 92521, "closure_pct": 1.8},
    {"time_h": 48, "area_px2": 73791, "closure_pct": 21.7},
]


def test_synthese_reprend_les_valeurs_finales_sans_les_recalculer():
    synthese = _calculer_synthese(ROWS)
    assert synthese["t0"]["area_px2"] == 8410
    assert synthese["t_final"]["area_px2"] == 2407
    assert synthese["fermeture_finale_pct"] == 71.6
    assert synthese["duree_h"] == 48
    assert synthese["vitesse_moyenne_pct_h"] == pytest.approx(1.49, abs=0.01)


def test_format_synthese_contient_les_bonnes_valeurs():
    texte = _format_synthese(_calculer_synthese(ROWS), n_mesures=len(ROWS))
    assert "8410" in texte
    assert "71,6%" in texte
    assert "1,49%/h" in texte
    assert "3" in texte  # nombre de mesures rappelé pour éviter de sur-interpréter la tendance


def test_intervalles_detectent_le_ralentissement():
    intervalles = _calculer_intervalles(ROWS)
    assert len(intervalles) == 2
    premier, second = intervalles
    assert premier["t_debut"] == 0 and premier["t_fin"] == 24
    assert second["t_debut"] == 24 and second["t_fin"] == 48
    # la fermeture ralentit nettement entre le premier et le second intervalle
    assert second["vitesse_pct_h"] < premier["vitesse_pct_h"]


def test_format_intervalles_signale_le_plus_rapide_et_le_plus_lent():
    texte = _format_intervalles(_calculer_intervalles(ROWS))
    assert "0h-24h" in texte
    assert "24h-48h" in texte


def test_un_seul_point_ne_produit_aucun_intervalle():
    assert _calculer_intervalles(ROWS[:1]) == []
    assert "pas d'intervalle" in _format_intervalles([])


def test_phrase_heterogeneite_utilise_le_taux_global_pas_un_taux_d_intervalle():
    """Régression : le LLM a une fois recopié le taux d'un intervalle (1,58%/h) à la
    place du taux moyen global (0,83%/h) en composant cette phrase lui-même. La
    phrase est maintenant entièrement pré-calculée dans le code pour que ce genre
    d'erreur de recopie ne puisse plus se produire, quel que soit le LLM utilisé."""
    synthese = _calculer_synthese(ROWS)
    phrase = _phrase_heterogeneite(synthese, _calculer_intervalles(ROWS))
    assert "1,49%/h" in phrase  # taux moyen global, pas un taux d'intervalle
    assert "1,67%/h" in phrase  # intervalle le plus rapide (0h-24h)
    assert "1,32%/h" in phrase  # intervalle le plus lent (24h-48h)


def test_phrase_heterogeneite_absente_avec_un_seul_intervalle():
    synthese = _calculer_synthese(ROWS[:2])
    phrase = _phrase_heterogeneite(synthese, _calculer_intervalles(ROWS[:2]))
    assert phrase.startswith("(")


def test_format_intervalles_ne_produit_jamais_un_signe_double_sur_delta_negatif():
    """Régression : un delta négatif (réouverture) produisait littéralement
    "fermeture +-19,9 points de %" à cause d'un "+" codé en dur devant le nombre,
    injecté tel quel dans le prompt du LLM (corrigé le 2026-09-09)."""
    texte = _format_intervalles(_calculer_intervalles(ROWS_REOUVERTURE))
    assert "+-" not in texte


def test_format_intervalles_signale_la_reouverture_partielle():
    texte = _format_intervalles(_calculer_intervalles(ROWS_REOUVERTURE))
    assert "réouverture partielle" in texte
    assert "19,9" in texte


def test_phrase_reouverture_signale_le_bon_intervalle_et_la_bonne_valeur():
    """Retour biologiste (2026-09-09) : un delta négatif décrit en "taux plus
    faible" laisse croire à un simple ralentissement, alors que la surface non
    colonisée a réellement augmenté -- signalé explicitement en une phrase
    pré-calculée, comme _phrase_heterogeneite, plutôt que laissé au LLM."""
    phrase = _phrase_reouverture(_calculer_intervalles(ROWS_REOUVERTURE))
    assert "24h" in phrase and "48h" in phrase
    assert "19,9" in phrase


def test_phrase_reouverture_absente_sans_delta_negatif():
    phrase = _phrase_reouverture(_calculer_intervalles(ROWS))
    assert phrase.startswith("(")


def test_phrase_interpretation_conserve_le_signe_negatif_sur_reouverture():
    """Régression (2026-09-09) : même avec un patron explicite laissant le LLM
    remplir bornes/vitesses lui-même, mistral 7B a fabriqué un signe "-" ou inversé
    l'ordre des bornes. La phrase est maintenant entièrement pré-calculée, comme
    _phrase_heterogeneite, pour éliminer ce risque plutôt que le corriger par une
    consigne de prompt supplémentaire."""
    phrase = _phrase_interpretation(_calculer_intervalles(ROWS_REOUVERTURE))
    assert "0h-24h" in phrase and "0,9%/h" in phrase
    assert "24h-48h" in phrase and "-0,83%/h" in phrase


def test_phrase_interpretation_sans_reouverture_ne_fabrique_pas_de_signe():
    phrase = _phrase_interpretation(_calculer_intervalles(ROWS_SANS_REOUVERTURE))
    assert "24h-48h" in phrase and "0,83%/h" in phrase
    assert "-0,83" not in phrase
    assert "48h-24h" not in phrase


def test_phrase_interpretation_absente_avec_un_seul_intervalle():
    phrase = _phrase_interpretation(_calculer_intervalles(ROWS[:2]))
    assert phrase.startswith("(")
