"""
ingest.py — Indexation de l'export Telegram dans ChromaDB
Usage : python ingest.py
"""

import json
import os
import uuid

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

EXPORT_PATH   = "data/telegram_export.json"
CHROMA_DIR    = "chroma_db"
COLLECTION    = "telegram_messages"
EMBED_MODEL   = "text-embedding-3-small"
CHUNK_WORDS   = 400   # taille cible d'un chunk en mots
OVERLAP_WORDS = 50    # chevauchement entre chunks
BATCH_SIZE    = 100   # nombre de chunks envoyés à Chroma en une fois


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_text(raw_text) -> str:
    """
    Le champ 'text' de Telegram peut être :
      - une str simple  → on la retourne telle quelle
      - une list de segments (str ou dict{"type","text"}) → on concatène
    """
    if isinstance(raw_text, str):
        return raw_text

    if isinstance(raw_text, list):
        parts = []
        for segment in raw_text:
            if isinstance(segment, str):
                parts.append(segment)
            elif isinstance(segment, dict) and "text" in segment:
                parts.append(segment["text"])
        return "".join(parts)

    return ""


def load_messages(path: str) -> list[str]:
    """Lit l'export JSON et retourne une liste de lignes '[Prénom]: texte'."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    lines = []
    for msg in messages:
        if msg.get("type") != "message":
            continue
        sender = msg.get("from") or "Inconnu"
        text   = parse_text(msg.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"[{sender}]: {text}")

    return lines


def chunk_lines(lines: list[str], chunk_words: int, overlap_words: int) -> list[str]:
    """
    Découpe la liste de lignes en chunks de ~chunk_words mots
    avec un chevauchement de overlap_words mots.
    On travaille sur les mots de l'ensemble concaténé.
    """
    full_text = "\n".join(lines)
    words = full_text.split()

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_words - overlap_words  # avance avec overlap

    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Génère les embeddings pour une liste de textes via OpenAI."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Vérification du fichier source
    if not os.path.exists(EXPORT_PATH):
        print(f"[ERREUR] Fichier introuvable : {EXPORT_PATH}")
        print("Place ton export Telegram dans data/telegram_export.json")
        return

    print("Chargement des messages…")
    lines = load_messages(EXPORT_PATH)
    print(f"  {len(lines)} messages valides trouvés.")

    print("Découpe en chunks…")
    chunks = chunk_lines(lines, CHUNK_WORDS, OVERLAP_WORDS)
    print(f"  {len(chunks)} chunks créés (≈{CHUNK_WORDS} mots, overlap {OVERLAP_WORDS}).")

    # Init clients
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection    = chroma_client.get_or_create_collection(name=COLLECTION)

    print("Vectorisation et indexation…")
    total = len(chunks)
    indexed = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_chunks = chunks[batch_start : batch_start + BATCH_SIZE]

        # Embeddings
        embeddings = embed_texts(openai_client, batch_chunks)

        # IDs uniques
        ids = [str(uuid.uuid4()) for _ in batch_chunks]

        # Insertion dans Chroma
        collection.add(
            documents=batch_chunks,   # on stocke le texte brut du chunk
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"text": c} for c in batch_chunks],
        )

        indexed += len(batch_chunks)
        print(f"  [{indexed}/{total}] chunks indexés…")

    print(f"\nTerminé. {indexed} chunks stockés dans '{CHROMA_DIR}/' (collection: {COLLECTION}).")


if __name__ == "__main__":
    main()
