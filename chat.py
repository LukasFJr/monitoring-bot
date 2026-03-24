"""
chat.py — Interface de chat avec RoboSalope
Usage : python chat.py
Quit  : Ctrl+C
"""

import os

import anthropic
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

CHROMA_DIR   = "chroma_db"
COLLECTION   = "telegram_messages"
EMBED_MODEL  = "text-embedding-3-small"
CLAUDE_MODEL = "claude-sonnet-4-6"
TEMPERATURE  = 0.59
MAX_TOKENS   = 150
TOP_K        = 8   # nombre de chunks récupérés dans Chroma

SYSTEM_PROMPT_TEMPLATE = """\
Ton nom est "RoboSalope", membre du groupe Telegram « Les Salopes ». \
Style: punchlines courtes, autosarcasme, emojis 🍑💦 admis, majuscules quand tu veux CRIER. \
Toujours borderline drôle. \
Donne toujours une réponse très courte, format SMS, une phrase max si possible. \
N'hésite pas à faire des fautes d'orthographe. \

Voici des extraits réels de conversations du groupe pour t'inspirer de leur style, \
leurs références, leurs blagues internes et leur façon de parler : \

{chunks}

Règles : tire absolument ton style, ta façon de parler et tes références de ces extraits. \
Ne sors jamais du personnage. \
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_embedding(client: OpenAI, text: str) -> list[float]:
    """Génère l'embedding d'un texte via OpenAI."""
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return response.data[0].embedding


def retrieve_chunks(collection, embedding: list[float], top_k: int) -> list[str]:
    """Interroge Chroma et retourne les top_k chunks les plus proches."""
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
    )
    # Les textes sont stockés dans les métadonnées
    metadatas = results.get("metadatas", [[]])[0]
    return [m["text"] for m in metadatas if "text" in m]


def build_system_prompt(chunks: list[str]) -> str:
    """Injecte les chunks dans le system prompt."""
    chunks_text = "\n---\n".join(chunks) if chunks else "(pas de contexte disponible)"
    return SYSTEM_PROMPT_TEMPLATE.format(chunks=chunks_text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Init clients
    openai_client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection    = chroma_client.get_or_create_collection(name=COLLECTION)

    # Bandeau de démarrage
    print("RoboSalope est en ligne 🍑")
    print("─" * 40)

    try:
        while True:
            user_input = input("Toi > ").strip()
            if not user_input:
                continue

            # 1. Embed la question
            embedding = get_embedding(openai_client, user_input)

            # 2. Récupère les chunks pertinents depuis Chroma
            chunks = retrieve_chunks(collection, embedding, TOP_K)

            # 3. Construit le system prompt avec les chunks injectés
            system_prompt = build_system_prompt(chunks)

            # 4. Appel Claude
            message = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_input}],
            )

            # 5. Affiche la réponse
            response_text = message.content[0].text
            print(f"RoboSalope > {response_text}")

    except KeyboardInterrupt:
        print("\nByeee 🍑")


if __name__ == "__main__":
    main()
