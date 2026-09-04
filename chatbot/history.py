"""Historique des conversations de l'assistant IA, persistant sur disque
(façon ChatGPT : plusieurs conversations distinctes, chacune retrouvable
après redémarrage de l'app, avec un titre dérivé de la première question).
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path(__file__).parent / "chat_history.json"


def load_conversations() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))


def save_conversations(conversations: list[dict]) -> None:
    HISTORY_PATH.write_text(
        json.dumps(conversations, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def new_conversation(resultat_segmentation: dict | None = None) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "titre": "Nouvelle conversation",
        "cree_le": datetime.now().isoformat(timespec="seconds"),
        "resultat_segmentation": resultat_segmentation,
        "messages": [],
    }


def add_message(conversation: dict, role: str, content: str, sources: list[str] | None = None) -> None:
    message = {"role": role, "content": content}
    if sources:
        message["sources"] = sources
    conversation["messages"].append(message)

    if role == "user" and conversation["titre"] == "Nouvelle conversation":
        conversation["titre"] = content[:60] + ("…" if len(content) > 60 else "")
