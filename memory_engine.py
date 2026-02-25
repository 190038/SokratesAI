import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import hashlib
from datetime import datetime

client = chromadb.PersistentClient(path="./chroma_db")

memories = client.get_or_create_collection("sokrates_memories")
entities = client.get_or_create_collection("sokrates_entities")

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def _stable_id(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return h


def store_item(text: str, tag="general", kind="memory", importance=0.5, meta=None):
    """
    kind: 'memory' or 'entity'
    importance: 0..1
    """
    col = memories if kind == "memory" else entities
    meta = meta or {}
    meta.update({
        "tag": tag,
        "kind": kind,
        "importance": float(importance),
        "ts": datetime.utcnow().isoformat() + "Z"
    })

    emb = embedder.encode(text).tolist()

    # Dedupe: if same text exists, skip
    item_id = _stable_id(f"{kind}:{tag}:{text}")
    try:
        col.add(
            ids=[item_id],
            documents=[text],
            embeddings=[emb],
            metadatas=[meta],
        )
    except Exception:
        # likely duplicate id -> ignore
        pass


def retrieve(query: str, n=5, tag=None, kind=None, min_importance=0.0):
    """
    Filter by tag/kind and minimum importance.
    Chroma requires exactly one top-level operator in 'where' when combining conditions.
    """
    emb = embedder.encode(query).tolist()

    conditions = []

    if tag is not None:
        conditions.append({"tag": tag})
    if kind is not None:
        conditions.append({"kind": kind})
    if min_importance > 0:
        conditions.append({"importance": {"$gte": float(min_importance)}})

    kwargs = dict(query_embeddings=[emb], n_results=n)

    if len(conditions) == 1:
        kwargs["where"] = conditions[0]
    elif len(conditions) > 1:
        kwargs["where"] = {"$and": conditions}

    # Choose correct collection
    col = entities if kind == "entity" else memories

    res = col.query(**kwargs)
    docs = res.get("documents", [[]])[0] or []
    return docs