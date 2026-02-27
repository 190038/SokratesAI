import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime, timezone

client = chromadb.PersistentClient(path="./chroma_db")

memories = client.get_or_create_collection("sokrates_memories")
entities = client.get_or_create_collection("sokrates_entities")

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Decay configuration
# -----------------------------

# How much importance is lost per idle day (no recalls)
DECAY_RATES = {
    "profile":          0.003,  # slowest — core user preferences
    "entity":           0.005,  # slow — names, projects, people
    "open_loop":        0.015,  # medium — unresolved tasks fade if ignored
    "other":            0.020,  # default
    "episode":          0.030,  # faster — specific conversation summaries
    "research":         0.040,  # fastest — web info goes stale quickly
    "research_source":  0.050,  # raw URLs become stale very fast
}

DECAY_RATE_DEFAULT = 0.020   # fallback for unknown tags
PRUNE_THRESHOLD    = 0.09    # delete entries that fall below this
MIN_IMPORTANCE     = 0.10    # floor — entries won't decay below this until pruned


# -----------------------------
# Internal helpers
# -----------------------------

def _stable_id(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return h


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _days_since(ts_str: str) -> float:
    """Return fractional days elapsed since an ISO-8601 timestamp string."""
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - ts
        return max(0.0, delta.total_seconds() / 86400.0)
    except Exception:
        return 0.0


# -----------------------------
# Core API
# -----------------------------

def store_item(text: str, tag="general", kind="memory", importance=0.5, meta=None):
    """
    Store a memory or entity in ChromaDB.

    kind:       'memory' or 'entity'
    importance: 0..1
    """
    col = memories if kind == "memory" else entities
    meta = meta or {}
    meta.update({
        "tag":        tag,
        "kind":       kind,
        "importance": float(importance),
        "ts":         _now_iso(),
    })

    emb = embedder.encode(text).tolist()
    item_id = _stable_id(f"{kind}:{tag}:{text}")

    try:
        col.add(
            ids=[item_id],
            documents=[text],
            embeddings=[emb],
            metadatas=[meta],
        )
    except Exception:
        # Duplicate id — silently ignore
        pass


def retrieve(query: str, n=5, tag=None, kind=None, min_importance=0.0, reinforce: float = 0.02):
    """
    Semantic retrieval with optional tag/kind/importance filters.

    reinforce:  If > 0, bumps importance of every returned item by this amount
                (capped at 1.0) AND refreshes its timestamp, resetting decay.
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

    col = entities if kind == "entity" else memories
    res = col.query(**kwargs)

    # Reinforce recalled memories — boost importance AND reset the clock
    if reinforce and reinforce > 0:
        ids   = res.get("ids",       [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        for item_id, meta in zip(ids, metas):
            old_imp = float(meta.get("importance", 0.5))
            new_imp = min(1.0, old_imp + reinforce)
            try:
                col.update(
                    ids=[item_id],
                    metadatas=[{**meta, "importance": new_imp, "ts": _now_iso()}]
                )
            except Exception:
                pass

    return res.get("documents", [[]])[0] or []


# -----------------------------
# Memory decay
# -----------------------------

def decay_memories(dry_run: bool = False) -> dict:
    """
    Apply time-based importance decay to all memories and entities.

    Each item loses importance proportional to the number of days since it was
    last recalled.  Items that drop below PRUNE_THRESHOLD are deleted.

    Items that have been recalled recently (ts refreshed by retrieve()) are
    effectively protected — their idle-day count is near zero.

    dry_run:  If True, compute what would happen but make no changes.
              Returns a report dict for inspection.

    Returns:
        {
            "decayed":  int,   # items whose importance was lowered
            "pruned":   int,   # items deleted
            "skipped":  int,   # items with no timestamp or already at floor
            "report":   list   # per-item details (only populated in dry_run)
        }
    """
    stats = {"decayed": 0, "pruned": 0, "skipped": 0, "report": []}

    for col_name, col in [("memory", memories), ("entity", entities)]:
        raw = col.get(include=["metadatas", "documents"])

        ids   = raw.get("ids",       []) or []
        metas = raw.get("metadatas", []) or []
        docs  = raw.get("documents", []) or []

        ids_to_delete  = []
        ids_to_update  = []
        metas_to_update = []

        for item_id, meta, doc in zip(ids, metas, docs):
            ts = meta.get("ts")
            if not ts:
                stats["skipped"] += 1
                continue

            days   = _days_since(ts)
            tag    = meta.get("tag", "other")
            rate   = DECAY_RATES.get(tag, DECAY_RATE_DEFAULT)
            old_imp = float(meta.get("importance", 0.5))

            # Compute raw decay (unclipped) — prune check must happen BEFORE floor
            raw_imp = old_imp - (rate * days)

            if raw_imp <= PRUNE_THRESHOLD:
                action  = "prune"
                new_imp = raw_imp                        # show true value in report
            elif abs(raw_imp - old_imp) > 0.001:
                action  = "decay"
                new_imp = max(MIN_IMPORTANCE, raw_imp)   # floor only when keeping
            else:
                action  = "keep"
                new_imp = old_imp

            if dry_run:
                stats["report"].append({
                    "text":      doc[:80],
                    "tag":       tag,
                    "kind":      col_name,
                    "days_idle": round(days, 1),
                    "old_imp":   round(old_imp, 3),
                    "new_imp":   round(new_imp, 3),
                    "action":    action,
                })
                if action == "prune":   stats["pruned"]  += 1
                elif action == "decay": stats["decayed"] += 1
                else:                   stats["skipped"] += 1
                continue

            if action == "prune":
                ids_to_delete.append(item_id)
                stats["pruned"] += 1
            elif action == "decay":
                ids_to_update.append(item_id)
                metas_to_update.append({**meta, "importance": round(new_imp, 4)})
                stats["decayed"] += 1
            else:
                stats["skipped"] += 1

        if not dry_run:
            if ids_to_delete:
                col.delete(ids=ids_to_delete)
            for item_id, meta in zip(ids_to_update, metas_to_update):
                try:
                    col.update(ids=[item_id], metadatas=[meta])
                except Exception:
                    pass

    if not dry_run:
        print(
            f"[Memory decay] decayed={stats['decayed']}  "
            f"pruned={stats['pruned']}  skipped={stats['skipped']}"
        )

    return stats


# -----------------------------
# Inspection / export helpers
# -----------------------------

def get_all_entries(kind: str = None, include_embeddings: bool = False):
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
    Export Chroma DB in human-readable format, sorted by importance descending.
    Each entry includes a projected importance after 7 and 30 days of no recall,
    so you can see at a glance which memories are at risk of fading.
    """
    def _projected(importance: float, tag: str, days: float) -> float:
        rate = DECAY_RATES.get(tag, DECAY_RATE_DEFAULT)
        return round(max(MIN_IMPORTANCE, importance - rate * days), 3)

    def format_collection(raw_data):
        docs   = raw_data.get("documents", [])
        metas  = raw_data.get("metadatas", [])
        embeds = raw_data.get("embeddings", []) if include_embeddings else []

        entries = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            imp  = float(meta.get("importance", 0.0))
            tag  = meta.get("tag", "")

            entry = {
                "text":            doc,
                "importance":      imp,
                "imp_in_7d":       _projected(imp, tag, 7),
                "imp_in_30d":      _projected(imp, tag, 30),
                "tag":             tag,
                "kind":            meta.get("kind", ""),
                "last_recalled":   meta.get("ts", ""),
                "days_idle":       round(_days_since(meta.get("ts", "")), 1),
            }
            if include_embeddings and i < len(embeds):
                entry["embedding"] = embeds[i]
            for k, v in meta.items():
                if k not in ["importance", "tag", "kind", "ts"]:
                    entry[f"meta_{k}"] = v
            entries.append(entry)

        return sorted(entries, key=lambda x: x["importance"], reverse=True)

    raw = get_all_entries()
    return {
        "memories": format_collection(raw["memories"]),
        "entities": format_collection(raw["entities"]),
    }