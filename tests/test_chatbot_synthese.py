"""Vérifie que les chiffres injectés dans le prompt du résumé scientifique
(synthèse globale + vitesses par intervalle) correspondent aux mesures fournies,
sans jamais dépendre du LLM. Couvre la régression corrigée dans chatbot.py :
le LLM inventait des pourcentages/vitesses différents des mesures réelles quand
on le laissait les calculer lui-même."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chatbot"))
from chatbot import _calculer_intervalles, _calculer_synthese, _format_intervalles, _format_synthese

ROWS = [
    {"time_h": 0, "area_px2": 8410, "closure_pct": 0.0},
    {"time_h": 24, "area_px2": 5063, "closure_pct": 40.0},
    {"time_h": 48, "area_px2": 2407, "closure_pct": 71.6},
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
