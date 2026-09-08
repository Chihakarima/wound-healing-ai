"""Indexation et recherche sémantique des articles récupérés par fetch_articles.py.

Construit une base vectorielle persistante (ChromaDB) à partir de
chatbot/articles_cicatrisation.json, pour que le chatbot puisse retrouver les
articles pertinents selon la question posée (titre + résumé).

Usage:
    python chatbot/index_articles.py
"""
import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

ARTICLES_PATH = Path(__file__).parent / "articles_cicatrisation.json"
DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "articles_cicatrisation"

_TOKEN_RE = re.compile(r"[a-zà-öø-ÿ0-9]+")

# Index BM25 construit paresseusement en mémoire (corpus petit, ~40 articles :
# retokeniser à chaque appel serait quasi instantané, mais autant le mettre en
# cache pour éviter de relire/retokeniser le JSON à chaque question posée dans
# une même session de chat).
_bm25_cache: tuple[BM25Okapi, list[dict]] | None = None


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


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _get_bm25() -> tuple[BM25Okapi, list[dict]]:
    global _bm25_cache
    if _bm25_cache is None:
        articles = json.loads(ARTICLES_PATH.read_text(encoding="utf-8"))
        corpus_tokenise = [_tokenize(f"{a['title']}. {a['abstract']}") for a in articles]
        _bm25_cache = (BM25Okapi(corpus_tokenise), articles)
    return _bm25_cache


def chercher(question: str, n: int = 3, hybride: bool = False) -> list[dict]:
    """Retourne les n articles les plus pertinents pour la question posée.

    hybride=False (par défaut, comportement de production inchangé) : recherche
    sémantique seule (embeddings ChromaDB).

    hybride=True : combine cette recherche sémantique à une recherche par
    mots-clés (BM25, gratuit, local) via reciprocal rank fusion (RRF) — corrige
    le cas où le vocabulaire de la question ne correspond pas bien à l'espace
    des embeddings par défaut (ex: question en français, voir README section
    "Base documentaire du RAG"), sans remplacer la recherche sémantique. Piste
    expérimentale : à valider par `chatbot/evaluate_retrieval.py` avant/après,
    pas encore adoptée par défaut dans chatbot.py.

    Chaque résultat: {"titre", "annee", "extrait"}.
    """
    if not hybride:
        return _chercher_embeddings(question, n)
    return _chercher_hybride(question, n)


def _chercher_embeddings(question: str, n: int) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=n)
    return [
        {"titre": meta["title"], "annee": meta["year"], "extrait": meta["abstract"]}
        for meta in results["metadatas"][0]
    ]


K_PAR_METHODE_HYBRIDE = 10  # profondeur de recherche par méthode avant fusion
K_RRF = 60  # constante standard de la reciprocal rank fusion (Cormack et al. 2009)


def _chercher_hybride(question: str, n: int) -> list[dict]:
    collection = get_collection()
    resultats_embeddings = collection.query(query_texts=[question], n_results=K_PAR_METHODE_HYBRIDE)
    titres_embeddings = [meta["title"] for meta in resultats_embeddings["metadatas"][0]]

    bm25, articles = _get_bm25()
    scores_bm25 = bm25.get_scores(_tokenize(question))
    ordre_bm25 = sorted(range(len(articles)), key=lambda i: scores_bm25[i], reverse=True)
    titres_bm25 = [articles[i]["title"] for i in ordre_bm25[:K_PAR_METHODE_HYBRIDE]]

    rrf_scores: dict[str, float] = {}
    for classement in (titres_embeddings, titres_bm25):
        for rang, titre in enumerate(classement, start=1):
            rrf_scores[titre] = rrf_scores.get(titre, 0.0) + 1.0 / (K_RRF + rang)

    par_titre = {a["title"]: a for a in articles}
    titres_fusionnes = sorted(rrf_scores, key=lambda t: rrf_scores[t], reverse=True)[:n]

    return [
        {"titre": t, "annee": par_titre[t].get("year") or 0, "extrait": par_titre[t]["abstract"]}
        for t in titres_fusionnes
    ]


if __name__ == "__main__":
    build_index()

    question_test = "effet du sérum (FCS) sur la vitesse de migration des fibroblastes"
    print(f"\nRecherche test: {question_test!r}\n")
    for i, article in enumerate(chercher(question_test, n=3), start=1):
        print(f"{i}. {article['titre']} ({article['annee']})")
        print(f"   {article['extrait'][:200]}...\n")
