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

Couvre trois scénarios de mesures volontairement différents (nombre de points,
ordre de grandeur des surfaces, présence ou non d'un seul intervalle) plutôt
qu'un unique fixture fixe : un "3/3 propre" sur un seul jeu de mesures ne dit
rien de la fiabilité sur un nombre de points différent ou des surfaces d'un
tout autre ordre de grandeur.

Nécessite un serveur Ollama local avec le modèle mistral : ignoré
automatiquement s'il n'est pas disponible (ex: CI GitHub Actions), pour ne
jamais faire dépendre la suite de tests d'un service externe. Comme la
génération reste probabiliste, un échec isolé de ce test doit être rejoué
avant d'être considéré comme une vraie régression."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chatbot"))
from chatbot import _calculer_intervalles, _calculer_synthese, _fr, generer_resume_stream


def _ollama_disponible() -> bool:
    try:
        import ollama
        ollama.Client().list()
        return True
    except Exception:
        return False


def _avec_separateur_milliers(chiffres: str) -> str:
    """Ex: '500000' -> '500 000'. Observé en pratique sur les scénarios à grandes
    surfaces (ROWS_2_POINTS, ROWS_5_POINTS ci-dessous) : le LLM insère parfois un
    séparateur de milliers que _format_synthese lui-même n'utilise jamais -- une
    variante de mise en forme du même nombre, pas un chiffre recalculé ou inventé."""
    groupes = []
    while len(chiffres) > 3:
        groupes.insert(0, chiffres[-3:])
        chiffres = chiffres[:-3]
    groupes.insert(0, chiffres)
    return " ".join(groupes)


def _valeur_presente(valeur: str, texte: str) -> bool:
    if valeur in texte:
        return True
    if valeur.isdigit() and len(valeur) > 3:
        variante = _avec_separateur_milliers(valeur)
        # espace normal, insécable, ou insécable étroite (les trois observées dans
        # des sorties LLM ou une mise en forme française usuelle des grands nombres)
        for espace in (" ", "\xa0", " "):
            if variante.replace(" ", espace) in texte:
                return True
    return False


# Valeurs fabriquées observées en pratique lors des régressions corrigées au fil du
# développement -- ne doivent plus jamais apparaître, quel que soit le scénario de
# mesures ou la formulation du prompt. Le groupe (36-48h, 0-36h, 40,0/40,1/40,47 %)
# vient de l'isolation test du 2026-09-08 sur chercher(hybride=True) : voir
# chatbot_prompt_fragility (mémoire projet) et generer_resume_stream (hybride=True
# gardé disponible pour le retrieval seul malgré ce risque, jamais pour la
# génération) -- ce test sert de garde-fou si l'hallucination redevient fréquente.
VALEURS_FABRIQUEES_CONNUES = [
    "87646", "72761", "32h-48h", "0h-8h", "40-48h", "24-40h",
    "36-48 heures", "36 heures", "0-36 heures", "40,47", "40,0 %", "40,1%",
    "7,2%/h", "12,76%/h", "4,2%/h", "38,39%",  # occurrences #4 et #7, voir README
]


def _valeurs_attendues(rows: list[dict]) -> list[str]:
    """Dérive les chiffres attendus depuis les mêmes fonctions pures que le pipeline
    utilise réellement (déjà couvertes indépendamment par test_chatbot_synthese.py),
    plutôt que de les recalculer à la main dans ce fichier -- élimine tout risque
    d'erreur d'arithmétique de ma part sur les nouveaux scénarios ci-dessous.

    N'inclut QUE le plus rapide et le plus lent parmi les taux par intervalle, pas
    tous : avec plus de 2 intervalles (voir ROWS_5_POINTS), le prompt ne demande au
    LLM de reprendre explicitement que ces deux-là (ligne "Intervalle le plus
    rapide/le plus lent" de _format_intervalles) -- exiger tous les intervalles
    ferait échouer le test sur un résumé pourtant correct."""
    synthese = _calculer_synthese(rows)
    intervalles = _calculer_intervalles(rows)
    valeurs = [
        str(synthese["t0"]["area_px2"]),
        str(synthese["t_final"]["area_px2"]),
        _fr(synthese["fermeture_finale_pct"]),
        _fr(synthese["vitesse_moyenne_pct_h"]),
    ]
    if len(intervalles) >= 2:
        plus_rapide = max(intervalles, key=lambda iv: iv["vitesse_pct_h"])
        plus_lent = min(intervalles, key=lambda iv: iv["vitesse_pct_h"])
        valeurs += {_fr(plus_rapide["vitesse_pct_h"]), _fr(plus_lent["vitesse_pct_h"])}
    elif intervalles:
        valeurs.append(_fr(intervalles[0]["vitesse_pct_h"]))
    return valeurs


def _generer_et_verifier(rows: list[dict]):
    morceaux, _sources = generer_resume_stream(rows)
    texte = "".join(morceaux)

    manquantes = [v for v in _valeurs_attendues(rows) if not _valeur_presente(v, texte)]
    assert not manquantes, (
        f"valeur(s) attendue(s) absente(s) du résumé généré : {manquantes}\n\n{texte}"
    )

    fabriquees = [v for v in VALEURS_FABRIQUEES_CONNUES if v in texte]
    assert not fabriquees, (
        f"valeur(s) fabriquée(s) connue(s) détectée(s) dans le résumé généré : {fabriquees}\n\n{texte}"
    )
    return texte


# Scénario canonique (celui de tout l'historique documenté dans le README) :
# 3 points, ordre de grandeur "normal" pour ce projet.
ROWS_3_POINTS = [
    {"time_h": 0, "area_px2": 94229, "closure_pct": 0.0},
    {"time_h": 24, "area_px2": 92521, "closure_pct": 1.8},
    {"time_h": 48, "area_px2": 56758, "closure_pct": 39.8},
]

# Cas limite structurel : un seul intervalle -- _format_intervalles et
# _phrase_heterogeneite empruntent un chemin différent (pas de ligne "le plus
# rapide/le plus lent", pas de phrase de liaison, voir test_chatbot_synthese.py).
# Jamais couvert par un vrai appel LLM avant cet ajout.
ROWS_2_POINTS = [
    {"time_h": 0, "area_px2": 50000, "closure_pct": 0.0},
    {"time_h": 72, "area_px2": 30000, "closure_pct": 40.0},
]

# 5 points, ordre de grandeur très différent (x5-6), et coïncidence volontaire :
# la vitesse moyenne globale (1,67%/h) est numériquement égale au taux du DERNIER
# intervalle (18h-24h) -- stress-test délibéré pour la confusion déjà observée par
# le passé entre "taux d'un intervalle" et "vitesse moyenne globale".
ROWS_5_POINTS = [
    {"time_h": 0, "area_px2": 500000, "closure_pct": 0.0},
    {"time_h": 6, "area_px2": 480000, "closure_pct": 4.0},
    {"time_h": 12, "area_px2": 420000, "closure_pct": 16.0},
    {"time_h": 18, "area_px2": 350000, "closure_pct": 30.0},
    {"time_h": 24, "area_px2": 300000, "closure_pct": 40.0},
]


@pytest.mark.skipif(not _ollama_disponible(), reason="Ollama non disponible localement")
def test_resume_llm_respecte_les_valeurs_calculees_3_points():
    _generer_et_verifier(ROWS_3_POINTS)


@pytest.mark.skipif(not _ollama_disponible(), reason="Ollama non disponible localement")
def test_resume_llm_respecte_les_valeurs_calculees_2_points_un_seul_intervalle():
    _generer_et_verifier(ROWS_2_POINTS)


@pytest.mark.skipif(not _ollama_disponible(), reason="Ollama non disponible localement")
def test_resume_llm_respecte_les_valeurs_calculees_5_points_grande_surface():
    _generer_et_verifier(ROWS_5_POINTS)
