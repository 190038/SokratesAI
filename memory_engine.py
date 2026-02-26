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


def retrieve(query: str, n=5, tag=None, kind=None, min_importance=0.0, reinforce: float = 0.02):
    """
    Filter by tag/kind and minimum importance.

    If `reinforce` &gt; 0, the importance of every returned item is increased by
    that amount (capped at 1.0). This lets the system gradually boost memories
    that keep being accessed.

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

    # optional importance reinforcement
    if reinforce and reinforce > 0:
        ids = res.get("ids", [[]])
        metas = res.get("metadatas", [[]])

        if ids and metas:
            ids = ids[0]
            metas = metas[0]

            for item_id, meta in zip(ids, metas):
                old_imp = float(meta.get("importance", 0.5))

                # diminishing returns reinforcement
                new_imp = old_imp + reinforce * (1.0 - old_imp)
                new_imp = min(1.0, new_imp)

                try:
                    col.update(
                        ids=[item_id],
                        metadatas=[{**meta, "importance": new_imp}]
                    )
                except Exception:
                    pass

    docs = res.get("documents", [[]])[0] or []
    return docs


def get_all_entries(kind: str = None, include_embeddings: bool = False):
    """
    Return all entries from Chroma collections.

    kind: None|'memory'|'entity' - limit to a single collection
    include_embeddings: whether to include raw embedding vectors
    Returns a dict or a single collection dict depending on `kind`.
    """
    # Chroma's `get(..., include=...)` accepts only these keys
    fields = ["documents", "metadatas"]
    if include_embeddings:
        fields.append("embeddings")

    if kind == "entity":
        return entities.get(include=fields)
    elif kind == "memory":
        return memories.get(include=fields)
    else:
        return {
            "memories": memories.get(include=fields),
            "entities": entities.get(include=fields),
        }


def export_chroma_readable(include_embeddings: bool = False):
    """
    Export Chroma DB in human-readable format with importance clearly visible.
    Returns: {"memories": [...], "entities": [...]} where each entry is 
    {"text": "...", "importance": 0.7, "metadata": {...}, ...}
    """
    def format_collection(raw_data):
        """Convert raw Chroma format to readable entry list."""
        docs = raw_data.get("documents", [])
        metas = raw_data.get("metadatas", [])
        embeds = raw_data.get("embeddings", []) if include_embeddings else []
        
        entries = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            entry = {
                "text": doc,
                "importance": float(meta.get("importance", 0.0)),
                "tag": meta.get("tag", ""),
                "kind": meta.get("kind", ""),
                "timestamp": meta.get("ts", ""),
            }
            if include_embeddings and i < len(embeds):
                entry["embedding"] = embeds[i]
            # Add any extra metadata
            for k, v in meta.items():
                if k not in ["importance", "tag", "kind", "ts"]:
                    entry[f"meta_{k}"] = v
            entries.append(entry)
        
        # Sort by importance descending
        return sorted(entries, key=lambda x: x["importance"], reverse=True)
    
    raw = get_all_entries()
    return {
        "memories": format_collection(raw["memories"]),
        "entities": format_collection(raw["entities"]),
    }

