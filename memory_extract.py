import json
import re
import ollama

MODEL = "deepseek-r1:8b"

MEMORY_SCHEMA = {
  "entities": [{"name": "", "type": "person|place|org|project|thing", "importance": 0.0}],
  "facts": [{"text": "", "importance": 0.0, "tag": "profile|project|preference|plan|bio|other"}],
  "open_loops": [{"text": "", "importance": 0.0}],
  "episode_summary": {"text": "", "importance": 0.0, "topic": ""}
}

def extract_memories(user_text: str, assistant_text: str, profile_json: str):
    prompt = [
        {
            "role": "system",
            "content": (
                "You extract human-like long-term memory from a conversation turn.\n"
                "Return ONLY valid JSON with keys: entities, facts, open_loops, episode_summary.\n"
                "Rules:\n"
                "- Keep ONLY stable, useful info a human would remember (names, roles, preferences, decisions, plans, projects).\n"
                "- importance is 0..1. Use >0.7 only for very important.\n"
                "- entities must include names if any appear.\n"
                "- facts should be short, atomic.\n"
                "- open_loops are unresolved questions/tasks.\n"
                "- episode_summary: 1-3 sentences.\n"
                "No markdown, no commentary."
            )
        },
        {"role": "user", "content": f"USER PROFILE (json):\n{profile_json}\n\nUSER SAID:\n{user_text}\n\nASSISTANT SAID:\n{assistant_text}"}
    ]

    out = ollama.chat(model=MODEL, messages=prompt)["message"]["content"]

    # robust JSON extraction
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except:
        return None