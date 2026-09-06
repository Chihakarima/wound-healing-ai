"""Récupère des articles scientifiques par catégorie (scratch assay, automatisation,
segmentation par deep learning, métriques d'évaluation) depuis l'API Semantic Scholar,
pour rééquilibrer la base RAG vers le cœur du projet (segmentation automatique +
évaluation) plutôt que vers les études de molécules/traitements.

Usage:
    python chatbot/fetch_articles_by_category.py
"""
import json
import time
from pathlib import Path

import requests

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OUT_PATH = Path(__file__).parent / "articles_nouveaux_categories.json"

CATEGORY_QUERIES = {
    "scratch_assay_principles": [
        "scratch assay wound healing limitations review collective cell migration",
    ],
    "automated_analysis": [
        "automated scratch assay image analysis wound closure quantification reproducibility",
    ],
    "deep_learning_segmentation": [
        "deep learning U-Net biomedical image segmentation microscopy",
        "convolutional neural network cell image segmentation",
    ],
    "metrics_evaluation": [
        "Dice coefficient Jaccard index medical image segmentation evaluation metrics",
    ],
}
LIMIT_PER_QUERY = 10


def fetch_query(query: str, limit: int, max_retries: int = 6) -> list[dict]:
    """Interroge Semantic Scholar avec retries (l'API publique sans clé renvoie
    régulièrement 429 Too Many Requests sous charge partagée)."""
    params = {"query": query, "limit": limit, "fields": "title,abstract,year"}
    delay = 20
    for attempt in range(max_retries):
        response = requests.get(API_URL, params=params, timeout=30)
        if response.status_code == 200:
            papers = response.json().get("data", [])
            return [
                {"title": p.get("title"), "abstract": p.get("abstract"), "year": p.get("year")}
                for p in papers
                if p.get("abstract")
            ]
        print(f"  [{query!r}] tentative {attempt + 1}/{max_retries} -> {response.status_code}, "
              f"nouvelle tentative dans {delay}s")
        time.sleep(delay)
        delay = min(delay * 1.5, 90)
    print(f"  [{query!r}] échec après {max_retries} tentatives, ignoré")
    return []


def main():
    resultats = {}
    for category, queries in CATEGORY_QUERIES.items():
        print(f"Catégorie: {category}")
        vus_titres = set()
        articles = []
        for q in queries:
            for a in fetch_query(q, LIMIT_PER_QUERY):
                titre_norm = (a["title"] or "").strip().lower()
                if titre_norm and titre_norm not in vus_titres:
                    vus_titres.add(titre_norm)
                    articles.append(a)
            time.sleep(5)
        resultats[category] = articles
        print(f"  -> {len(articles)} articles récupérés\n")

    OUT_PATH.write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Résultats bruts (à trier manuellement) sauvegardés dans {OUT_PATH}")


if __name__ == "__main__":
    main()
