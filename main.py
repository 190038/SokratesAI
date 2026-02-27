import ollama
import json
import os
import re 

from research_engine import run_research
from memory_engine import store_item, retrieve, export_chroma_readable, decay_memories
from memory_extract import extract_memories 

MEMORY_FILE = "memory.json"
PROFILE_FILE = "user_profile.json"
MODEL = "deepseek-r1:8b"



# -----------------------------
# Load persistent chat memory
# -----------------------------
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        messages = json.load(f)
else:
    messages = []


# -----------------------------
# Ensure profile file exists
# -----------------------------
if not os.path.exists(PROFILE_FILE):
    profile = {
        "interests": [],
        "projects": [],
        "reasoning_style": "",
        "recurring_topics": [],
        "long_term_questions": []
    }
    with open(PROFILE_FILE, "w") as f:
        json.dump(profile, f, indent=2)


# -----------------------------
# Function: analyze user input
# -----------------------------
def analyze_user_input(user_text):
    """
    Extract structured user info and update profile safely.
    """

    try:
        analysis = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract structured information from the user input.\n"
                        "Return ONLY valid JSON with fields:\n"
                        "interests, projects, recurring_topics, long_term_questions.\n"
                        "Each must be a list of short strings.\n"
                        "If nothing applies, return empty lists."
                    )
                },
                {"role": "user", "content": user_text}
            ]
        )

        extracted = analysis["message"]["content"]

        # Extract first JSON block safely
        match = re.search(r"\{.*\}", extracted, re.DOTALL)
        if not match:
            print("[Profile parser skipped invalid JSON]")
            return

        try:
            data = json.loads(match.group(0))
        except:
            print("[Profile parser skipped invalid JSON]")
            return

        # Load existing profile
        with open(PROFILE_FILE, "r") as f:
            profile = json.load(f)

        # Merge without duplicates
        for key in ["interests", "projects", "recurring_topics", "long_term_questions"]:
            if key in data and isinstance(data[key], list):
                profile[key] = list(set(profile[key] + data[key]))

        # Save updated profile
        with open(PROFILE_FILE, "w") as f:
            json.dump(profile, f, indent=2)

        print("\n[Profile updated]")
        print(profile)
        print()

    except Exception as e:
        print("Profile analysis failed:", e)


print("Sokrates ready. Type 'exit' to quit.\n")


# Export entire Chroma DB to JSON and filter by importance for debugging/inspection
def get_all_entries(kind=None, include_embeddings=False):
    try:
        all_entries = export_chroma_readable(include_embeddings=False)
        with open("all_chroma.json", "w", encoding="utf-8") as f:
            json.dump(all_entries, f, indent=2, ensure_ascii=False)
        print("[Exported chroma DB] all_chroma.json")
    except Exception as e:
        print("[Failed to export chroma DB]", e)


# -----------------------------
# Main conversation loop
# -----------------------------
while True:
    user = input("You: ")
    if user.lower().strip() == "research":
        # Use last ~12 messages as the "topic context"
        context_chunk = "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in messages[-12:]]
        )
        
        print("\n[Research mode] Running web research...\n")

        report, queries, search_results, fetched = run_research(context_chunk)

        print("[Search queries]")
        for q in queries:
            print(f"  • {q}")
        print()

        # Save a readable report
        with open("research_report.md", "w", encoding="utf-8") as f:
            f.write("# Sokrates Research Report\n\n")
            f.write("## Queries\n")
            for q in queries:
                f.write(f"- {q}\n")
            f.write("\n## Report\n")
            f.write(report)
            f.write("\n")

        print("[Research mode] Done. Saved: research_report.md\n")
        continue

    if user.lower().strip() == "research summary":
        if os.path.exists("last_research.json"):
            with open("last_research.json", "r", encoding="utf-8") as f:
                last = json.load(f)

            # Ask DeepSeek to produce a clean summary
            summary_prompt = [
                {"role": "system", "content": "Summarize the research report clearly and compactly. Include the most important points and any key caveats."},
                {"role": "user", "content": last["report"]}
            ]
            summary = ollama.chat(model=MODEL, messages=summary_prompt)["message"]["content"]
            print("\nSokrates (research summary):", summary, "\n")
        else:
            print("\nSokrates: No research found yet. Run `research` first.\n")
        continue
    
    if user.lower().strip() == "decay dry run":
        report = decay_memories(dry_run=True)

        print(f"\n[Decay dry run]")
        print(f"  Would decay:  {report['decayed']}")
        print(f"  Would prune:  {report['pruned']}")
        print(f"  Skipped:      {report['skipped']}")

        for item in report["report"]:
            print(
                f"  [{item['action'].upper():5}] "
                f"idle={item['days_idle']}d  "
                f"imp {item['old_imp']} → {item['new_imp']}  "
                f"tag={item['tag']:15}  "
                f"\"{item['text']}\""
            )
        continue

    if user.lower().strip() == "export memory":
        get_all_entries()
        continue
    
    if user.lower() == "exit":
        break

    # -------------------------
    # Retrieve relevant memory
    # -------------------------
   # Build prompt fresh each turn
    prompt_messages = []

    # Add profile
    with open(PROFILE_FILE, "r") as f:
        profile = json.load(f)

        prompt_messages.append({
            "role": "system",
            "content": "User profile:\n" + json.dumps(profile, indent=2)
        })

        # Add semantic memory
        related_memories = retrieve(user, n=5)
        if related_memories:
            prompt_messages.append({
                "role": "system",
                "content": "Relevant past knowledge:\n" + "\n".join(related_memories)
            })

        # Add recent chat history (last 10 messages only)
        prompt_messages.extend(messages[-10:])

        # Add new user input
        prompt_messages.append({"role": "user", "content": user})
        
        related_research = retrieve(user, n=3, tag="research")
        if related_research:
            prompt_messages.append({
                "role": "system",
                "content": "Relevant research you previously gathered:\n" + "\n".join(related_research)
            })
            # Entity recall (names etc.)
        entity_hits = retrieve(user, n=6, kind="entity")
        if entity_hits:
            prompt_messages.append({
                "role": "system",
                "content": "Known entities (names, projects, etc.):\n" + "\n".join(entity_hits)
            })

        # Important memories recall
        important_hits = retrieve(user, n=6, kind="memory", min_importance=0.6)
        if important_hits:
            prompt_messages.append({
                "role": "system",
                "content": "Relevant high-importance memories:\n" + "\n".join(important_hits)
            })
        # -------------------------
        # Call DeepSeek
        # -------------------------
        response = ollama.chat(
            model=MODEL,
            messages=prompt_messages
        )

        reply = response["message"]["content"]

        print("\nSokrates:", reply, "\n")

        # Add assistant reply to chat memory
        messages.append({"role": "assistant", "content": reply})
    
            # Load profile for extraction context
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            profile_json = f.read()

        extracted = extract_memories(user, reply, profile_json)

        if extracted:
            # Entities (names etc.)
            for e in extracted.get("entities", []):
                # Support both dict items and simple string items returned by the extractor
                if isinstance(e, dict):
                    name = (e.get("name") or "").strip()
                    etype = (e.get("type") or "thing").strip()
                    try:
                        imp = float(e.get("importance") or 0.4)
                    except Exception:
                        imp = 0.4
                elif isinstance(e, str):
                    name = e.strip()
                    etype = "thing"
                    imp = 0.5
                else:
                    # Fallback: coerce to string
                    name = str(e).strip()
                    etype = "thing"
                    imp = 0.4

                if name and imp >= 0.45:
                    store_item(
                        text=f"{name} ({etype})",
                        tag="entity",
                        kind="entity",
                        importance=imp,
                        meta={"entity_type": etype}
                    )

            # Facts
            for fact in extracted.get("facts", []):
                # Support both dict items and simple string items returned by the extractor
                if isinstance(fact, dict):
                    txt = (fact.get("text") or "").strip()
                    try:
                        imp = float(fact.get("importance") or 0.4)
                    except Exception:
                        imp = 0.4
                    tag = (fact.get("tag") or "other").strip()
                elif isinstance(fact, str):
                    txt = fact.strip()
                    imp = 0.5
                    tag = "other"
                else:
                    txt = str(fact).strip()
                    imp = 0.4
                    tag = "other"

                # only store important facts
                if txt and imp >= 0.60:
                    store_item(txt, tag=tag, kind="memory", importance=imp)

            # Open loops
            for loop in extracted.get("open_loops", []):
                # Support both dict items and simple string items returned by the extractor
                if isinstance(loop, dict):
                    txt = (loop.get("text") or "").strip()
                    try:
                        imp = float(loop.get("importance") or 0.5)
                    except Exception:
                        imp = 0.5
                elif isinstance(loop, str):
                    txt = loop.strip()
                    imp = 0.5
                else:
                    txt = str(loop).strip()
                    imp = 0.5

                if txt and imp >= 0.55:
                    store_item(txt, tag="open_loop", kind="memory", importance=imp)

            # Episode summary
            ep = extracted.get("episode_summary", {}) or {}
            # Support episode_summary as dict or simple string
            if isinstance(ep, dict):
                ep_txt = (ep.get("text") or "").strip()
                try:
                    ep_imp = float(ep.get("importance") or 0.55)
                except Exception:
                    ep_imp = 0.55
                topic = (ep.get("topic") or "general").strip()
            elif isinstance(ep, str):
                ep_txt = ep.strip()
                ep_imp = 0.55
                topic = "general"
            else:
                ep_txt = str(ep).strip()
                ep_imp = 0.55
                topic = "general"

            if ep_txt and ep_imp >= 0.55:
                store_item(f"[{topic}] {ep_txt}", tag="episode", kind="memory", importance=ep_imp)

            # -------------------------
            # Save persistent chat file
            # -------------------------
            with open(MEMORY_FILE, "w") as f:
                json.dump(messages, f)

            # -------------------------
            # Analyze user input (Step 4)
            # -------------------------
            analyze_user_input(user)