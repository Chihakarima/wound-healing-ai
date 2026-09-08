"""Test de régression contre les hallucinations numériques du résumé LLM
(generer_resume_stream). Ne vérifie jamais que le LLM "a bien raisonné" -
seulement que le texte produit reprend les chiffres déjà calculés par le
pipeline (voir _calculer_synthese / _calculer_intervalles) et ne contient pas
les valeurs fabriquées observées lors du développement :
- confusion entre le taux d'un intervalle et le taux moyen global,
- nombre de mesures inventé (aucune donnée du prompt ne le justifiait),
- surfaces et bornes d'intervalle entièrement fabriquées, provoquées une fois
  par une formulation de prompt trop permissive (depuis retirée, voir README
  "Fiabilité de la génération par LLM").

Nécessite un serveur Ollama local avec le modèle mistral : ignoré
automatiquement s'il n'est pas disponible (ex: CI GitHub Actions), pour ne
jamais faire dépendre la suite de tests d'un service externe. Comme la
génération reste probabiliste, un échec isolé de ce test doit être rejoué
avant d'être considéré comme une vraie régression."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chatbot"))
from chatbot import generer_resume_stream


def _ollama_disponible() -> bool:
    try:
        import ollama
        ollama.Client().list()
        return True
    except Exception:
        return False


ROWS = [
    {"time_h": 0, "area_px2": 94229, "closure_pct": 0.0},
    {"time_h": 24, "area_px2": 92521, "closure_pct": 1.8},
    {"time_h": 48, "area_px2": 56758, "closure_pct": 39.8},
]

# Chiffres corrects, calculés indépendamment de ce test (voir test_chatbot_synthese.py) :
# vitesse moyenne = 39,8/48 = 0,83 %/h ; intervalle 0h-24h = 1,8/24 = 0,07 %/h ;
# intervalle 24h-48h = 38/24 = 1,58 %/h.
VALEURS_ATTENDUES = ["94229", "56758", "39,8", "0,83", "1,58", "0,07"]

# Valeurs fabriquées observées en pratique lors des régressions corrigées cette session -
# ne doivent plus jamais apparaître, quelle que soit la formulation du prompt.
# Le groupe ci-dessous (36-48h, 0-36h, 40,0/40,1/40,47 %) vient de l'isolation test du
# 2026-09-08 sur chercher(hybride=True) : voir chatbot_prompt_fragility (mémoire projet)
# et generer_resume_stream ci-dessus (hybride=True gardé malgré ce risque, sur demande
# explicite -- ce test sert de garde-fou si l'hallucination redevient fréquente).
VALEURS_FABRIQUEES_CONNUES = [
    "87646", "72761", "32h-48h", "0h-8h", "40-48h", "24-40h",
    "36-48 heures", "36 heures", "0-36 heures", "40,47", "40,0 %", "40,1%",
]


@pytest.mark.skipif(not _ollama_disponible(), reason="Ollama non disponible localement")
def test_resume_llm_respecte_les_valeurs_calculees():
    morceaux, _sources = generer_resume_stream(ROWS)
    texte = "".join(morceaux)

    manquantes = [v for v in VALEURS_ATTENDUES if v not in texte]
    assert not manquantes, (
        f"valeur(s) attendue(s) absente(s) du résumé généré : {manquantes}\n\n{texte}"
    )

    fabriquees = [v for v in VALEURS_FABRIQUEES_CONNUES if v in texte]
    assert not fabriquees, (
        f"valeur(s) fabriquée(s) connue(s) détectée(s) dans le résumé généré : {fabriquees}\n\n{texte}"
    )
