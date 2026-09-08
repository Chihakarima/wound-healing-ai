import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chatbot"))
from chatbot import detecter_citations_suspectes

SOURCES = [
    "Studying Cell Migration (Random and Wound Healing) Parameters with Imaging and MATLAB Analysis",
    "Study of Wound Healing Dynamics by Single Pseudo-Particle Tracking in Phase Contrast Images Acquired in Time-Lapse",
    "Migratory Metrics of Wound Healing: A Quantification Approach for in vitro Scratch Assays",
]


def test_titre_invente_absent_des_sources_est_detecte():
    # Cas réel observé (2026-09-08) : titre plausible mais absent du corpus.
    texte = (
        'Un article intitulé "Analysis of wound healing using digital image analysis" '
        "explique les méthodes de traitement de l'image."
    )
    suspectes = detecter_citations_suspectes(texte, SOURCES)
    assert suspectes == ["Analysis of wound healing using digital image analysis"]


def test_titre_recopie_exactement_n_est_pas_signale():
    texte = f'D\'après "{SOURCES[0]}", la migration cellulaire est mesurable par imagerie.'
    assert detecter_citations_suspectes(texte, SOURCES) == []


def test_titre_recopie_avec_diacritique_different_n_est_pas_signale():
    # Variante unicode plausible (apostrophe/tiret différent) : ne doit pas être
    # signalée comme fabriquée juste à cause d'un caractère de ponctuation.
    texte = 'Voir "Migratory Metrics of Wound Healing: A Quantification Approach for in-vitro Scratch Assays".'
    assert detecter_citations_suspectes(texte, SOURCES) == []


def test_aucune_citation_ne_renvoie_liste_vide():
    texte = "Ce texte ne cite aucun article entre guillemets, juste des chiffres : 39,8 %."
    assert detecter_citations_suspectes(texte, SOURCES) == []


def test_citation_courte_ignoree_pas_un_titre():
    # Emphase courte entre guillemets ("hétérogénéité"), pas une tentative de
    # citation d'article -- ne doit pas déclencher un faux positif.
    texte = 'Cette valeur cache une certaine "hétérogénéité" entre intervalles.'
    assert detecter_citations_suspectes(texte, SOURCES) == []


def test_aucune_source_disponible_signale_toute_citation_longue():
    texte = 'Voir "Un article totalement hors de propos et jamais fourni ici".'
    assert detecter_citations_suspectes(texte, []) == [
        "Un article totalement hors de propos et jamais fourni ici"
    ]
