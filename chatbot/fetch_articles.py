"""Récupère des articles scientifiques sur la cicatrisation / migration cellulaire
depuis l'API Semantic Scholar, et les sauvegarde dans un fichier JSON local.

C'est la première brique du futur chatbot : cette base d'articles sera ensuite
indexée (ChromaDB) pour que le chatbot puisse retrouver les articles pertinents
selon la question posée.

Usage:
    python chatbot/fetch_articles.py
"""
import json
from pathlib import Path

import requests

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
QUERY = "in vitro wound healing assay cell migration wound closure"
LIMIT = 100
OUT_PATH = Path(__file__).parent / "articles_cicatrisation.json"


def fetch_articles(query: str = QUERY, limit: int = LIMIT) -> list[dict]:
    """Interroge Semantic Scholar et retourne les articles qui ont un résumé
    (certains articles n'en ont pas dans l'API, on les ignore)."""
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year",
    }
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    papers = response.json().get("data", [])

    return [
        {"title": p.get("title"), "abstract": p.get("abstract"), "year": p.get("year")}
        for p in papers
        if p.get("abstract")
    ]


def main():
    articles = fetch_articles()
    OUT_PATH.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(articles)} articles (avec résumé) sauvegardés dans {OUT_PATH}")


if __name__ == "__main__":
    main()
