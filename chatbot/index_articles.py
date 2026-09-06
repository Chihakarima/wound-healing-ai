"""Indexation et recherche sémantique des articles récupérés par fetch_articles.py.

Construit une base vectorielle persistante (ChromaDB) à partir de
chatbot/articles_cicatrisation.json, pour que le chatbot puisse retrouver les
articles pertinents selon la question posée (titre + résumé).

Usage:
    python chatbot/index_articles.py
"""
import json
from pathlib import Path

import chromadb

ARTICLES_PATH = Path(__file__).parent / "articles_cicatrisation.json"
DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "articles_cicatrisation"


def build_index():
    """(Re)construit la base vectorielle à partir du JSON d'articles."""
    articles = json.loads(ARTICLES_PATH.read_text(encoding="utf-8"))

    client = chromadb.PersistentClient(path=str(DB_PATH))
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    ids = [str(i) for i in range(len(articles))]
    documents = [f"{a['title']}. {a['abstract']}" for a in articles]
    metadatas = [
        {
            "title": a["title"],
            "year": a.get("year") or 0,
            "abstract": a["abstract"],
            "category": a.get("category") or "non_categorise",
            "topics": ",".join(a.get("topics", [])),
        }
        for a in articles
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"{len(articles)} articles indexés dans {DB_PATH}")


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Index introuvable. Lancez d'abord: python chatbot/index_articles.py"
        ) from exc


def chercher(question: str, n: int = 3) -> list[dict]:
    """Retourne les n articles les plus pertinents pour la question posée.

    Chaque résultat: {"titre", "annee", "extrait"}.
    """
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=n)

    return [
        {"titre": meta["title"], "annee": meta["year"], "extrait": meta["abstract"]}
        for meta in results["metadatas"][0]
    ]


if __name__ == "__main__":
    build_index()

    question_test = "effet du sérum (FCS) sur la vitesse de migration des fibroblastes"
    print(f"\nRecherche test: {question_test!r}\n")
    for i, article in enumerate(chercher(question_test, n=3), start=1):
        print(f"{i}. {article['titre']} ({article['annee']})")
        print(f"   {article['extrait'][:200]}...\n")
