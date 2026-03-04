from memory_engine import store_item, decay_memories, memories, entities
from datetime import datetime, timezone, timedelta

print("=== This script will inject test memories with various ages and importance levels, then run a dry-run of the decay process to show which would be decayed or pruned. ===\n")

def fake_ts(days_ago: int) -> str:
    """Generate a timestamp N days in the past."""
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


def inject_test_memories():
    test_items = [
        # --- Should SURVIVE (recent or high importance + slow decay) ---
        ("User prefers dark mode",           "profile",         "memory", 0.8,  2),   # 0.8  - (0.003*2)  = 0.794 ✓
        ("Project: build Sokrates web app",  "entity",          "memory", 0.9,  5),   # 0.9  - (0.005*5)  = 0.875 ✓
        ("User likes chess",                 "profile",         "memory", 0.7,  10),  # 0.7  - (0.003*10) = 0.670 ✓

        # --- Should DECAY but survive ---
        ("User mentioned feeling tired",     "episode",         "memory", 0.65, 10),  # 0.65 - (0.03*10)  = 0.350 ✓
        ("Open task: review memory engine",  "open_loop",       "memory", 0.6,  15),  # 0.6  - (0.015*15) = 0.375 ✓

        # --- Should be PRUNED (raw value drops below 0.09) ---
        ("Research: quantum computing 2024", "research",        "memory", 0.4,  30),  # 0.4  - (0.04*30)  = -0.8  → pruned
        ("Source: some-old-url.com",         "research_source", "memory", 0.3,  20),  # 0.3  - (0.05*20)  = -0.7  → pruned
        ("User said hello",                  "episode",         "memory", 0.25, 15),  # 0.25 - (0.03*15)  = -0.2  → pruned
    ]
    
    for text, tag, kind, importance, days_ago in test_items:
        # Store normally first
        store_item(text, tag=tag, kind=kind, importance=importance)

        # Then backdating the timestamp directly in ChromaDB
        item_id = __import__('hashlib').sha256(
            f"{kind}:{tag}:{text}".encode()
        ).hexdigest()[:24]

        col = memories  # all test items go to memories collection
        try:
            col.update(
                ids=[item_id],
                metadatas=[{
                    "tag": tag,
                    "kind": kind,
                    "importance": float(importance),
                    "ts": fake_ts(days_ago)
                }]
            )
            print(f"  Injected [{tag:15}] imp={importance}  {days_ago}d ago  \"{text}\"")
        except Exception as e:
            print(f"  Failed to backdate: {e}")


if __name__ == "__main__":
    print("=== Injecting test memories ===\n")
    inject_test_memories()

    print("\n=== Dry run BEFORE actual decay ===\n")
    report = decay_memories(dry_run=True)
    for item in report["report"]:
        print(
            f"  [{item['action'].upper():5}]  "
            f"idle={item['days_idle']}d  "
            f"imp {item['old_imp']} → {item['new_imp']}  "
            f"tag={item['tag']:15}  "
            f"\"{item['text']}\""
        )

    print(f"\n  Would decay: {report['decayed']}")
    print(f"  Would prune: {report['pruned']}")

    confirm = input("\nRun REAL decay now? (yes/no): ").strip().lower()
    if confirm == "yes":
        real = decay_memories(dry_run=False)
        print(f"\n=== Real decay done ===")
        print(f"  Decayed: {real['decayed']}")
        print(f"  Pruned:  {real['pruned']}")
    else:
        print("\nSkipped real decay — test data remains in DB.")
        print("Run 'export memory' in Sokrates to inspect it, then re-run this script to clean up.")