# sokrates_app.py
"""
Engine for CLI + FastAPI.

Root cause of the "assistant prints my profile JSON" bug:
- Your previous code injected the *raw* user_profile.json into the chat prompt as a system message:
    "User profile:\n{...json...}"
  DeepSeek then often replies to that JSON (mirrors/explains it) instead of answering the user.

Fix (real fix, not a workaround):
- The chat model no longer receives the raw profile JSON at all.
- Profile is used only internally for memory extraction + profile updater.

Keeps your current feature set:
- loads last 20 messages from memory.json
- commands: research, research summary, decay dry run, export memory, exit
- uses Chroma memory + extractor + profile updater
"""

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional

import ollama

from research_engine import run_research
from memory_engine import store_item, retrieve, export_chroma_readable, decay_memories
from memory_extract import extract_memories

Message = Dict[str, str]


class SokratesApp:
    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        memory_file: str = "memory.json",
        profile_file: str = "user_profile.json",
    ):
        self.model = model
        self.memory_file = memory_file
        self.profile_file = profile_file

        # Web safety: multiple concurrent requests
        self._lock = threading.Lock()

        self.messages: List[Message] = self._load_messages_last20()
        self._ensure_profile_file()

    # -----------------------------
    # Persistence
    # -----------------------------
    def _load_messages_last20(self) -> List[Message]:
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                msgs = json.load(f)
            return msgs[-20:]
        return []

    def _save_messages(self) -> None:
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def _ensure_profile_file(self) -> None:
        if not os.path.exists(self.profile_file):
            profile = {
                "interests": [],
                "projects": [],
                "reasoning_style": "",
                "recurring_topics": [],
                "long_term_questions": [],
            }
            with open(self.profile_file, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)

    def _load_profile(self) -> Dict[str, Any]:
        with open(self.profile_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_profile(self, profile: Dict[str, Any]) -> None:
        with open(self.profile_file, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

    # -----------------------------
    # Profile updater (unchanged logic)
    # -----------------------------
    def analyze_user_input(self, user_text: str) -> Optional[Dict[str, Any]]:
        """Extract structured info and merge into profile."""
        try:
            analysis = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract structured information from the user input.\n"
                            "Return ONLY valid JSON with fields:\n"
                            "interests, projects, recurring_topics, long_term_questions.\n"
                            "Each must be a list of short strings.\n"
                            "If nothing applies, return empty lists."
                        ),
                    },
                    {"role": "user", "content": user_text},
                ],
            )

            extracted = analysis["message"]["content"]
            match = re.search(r"\{.*\}", extracted, re.DOTALL)
            if not match:
                return None

            try:
                data = json.loads(match.group(0))
            except Exception:
                return None

            profile = self._load_profile()

            for key in ["interests", "projects", "recurring_topics", "long_term_questions"]:
                if key in data and isinstance(data[key], list):
                    profile[key] = list(set(profile.get(key, []) + data[key]))

            self._save_profile(profile)
            return profile
        except Exception:
            return None

    # -----------------------------
    # Export helper
    # -----------------------------
    def export_memory(self) -> Dict[str, Any]:
        all_entries = export_chroma_readable(include_embeddings=False)
        with open("all_chroma.json", "w", encoding="utf-8") as f:
            json.dump(all_entries, f, indent=2, ensure_ascii=False)
        return all_entries

    # -----------------------------
    # Main entrypoint: one user turn
    # -----------------------------
    def handle_turn(self, user_text: str) -> Dict[str, Any]:
        user_text = (user_text or "").strip()
        if not user_text:
            return {"type": "error", "reply": "Bitte schreib etwas."}

        with self._lock:
            low = user_text.lower().strip()

            # -------------------------
            # Commands
            # -------------------------
            if low == "research":
                last_user_msg = next(
                    (m["content"] for m in reversed(self.messages) if m.get("role") == "user"),
                    None,
                )
                if not last_user_msg:
                    return {"type": "error", "reply": "Nothing to research yet — say something first."}

                report, queries, search_results, fetched = run_research(last_user_msg)

                with open("research_report.md", "w", encoding="utf-8") as f:
                    f.write("# Sokrates Research Report\n\n")
                    f.write("## Queries\n")
                    for q in queries:
                        f.write(f"- {q}\n")
                    f.write("\n## Report\n")
                    f.write(report)
                    f.write("\n")

                return {
                    "type": "research_done",
                    "reply": "Research done. Saved: research_report.md",
                    "queries": queries,
                }

            if low == "research summary":
                if os.path.exists("last_research.json"):
                    with open("last_research.json", "r", encoding="utf-8") as f:
                        last = json.load(f)

                    summary_prompt = [
                        {
                            "role": "system",
                            "content": (
                                "Summarize the research report clearly and compactly. "
                                "Include the most important points and any key caveats."
                            ),
                        },
                        {"role": "user", "content": last.get("report", "")},
                    ]
                    summary = ollama.chat(model=self.model, messages=summary_prompt)["message"]["content"]
                    return {"type": "research_summary", "reply": summary}
                return {"type": "error", "reply": "No research found yet. Run `research` first."}

            if low == "decay dry run":
                report = decay_memories(dry_run=True)
                return {"type": "decay_dry_run", "reply": "OK", "report": report}

            if low == "export memory":
                self.export_memory()
                return {"type": "export_done", "reply": "Exported chroma DB → all_chroma.json"}

            if low == "exit":
                return {"type": "exit", "reply": "Bye."}

            # -------------------------
            # Build chat prompt (FIXED: no raw profile JSON injected)
            # -------------------------
            prompt_messages: List[Message] = []

            # (Optional) very small behavioral instruction, not user data
            prompt_messages.append({
                "role": "system",
                "content": "Antworte direkt und natürlich auf Deutsch."
            })

            # Semantic memory (chroma)
            related_memories = retrieve(user_text, n=5)
            if related_memories:
                prompt_messages.append({
                    "role": "system",
                    "content": "Relevant past knowledge:\n" + "\n".join(related_memories)
                })

            # Recent chat history (last 10)
            prompt_messages.extend(self.messages[-10:])

            # Current user input
            prompt_messages.append({"role": "user", "content": user_text})

            # Optional other retrieval layers (as in your code)
            related_research = retrieve(user_text, n=3, tag="research")
            if related_research:
                prompt_messages.append({
                    "role": "system",
                    "content": "Relevant research you previously gathered:\n" + "\n".join(related_research)
                })

            entity_hits = retrieve(user_text, n=6, kind="entity")
            if entity_hits:
                prompt_messages.append({
                    "role": "system",
                    "content": "Known entities (names, projects, etc.):\n" + "\n".join(entity_hits)
                })

            important_hits = retrieve(user_text, n=6, kind="memory", min_importance=0.6)
            if important_hits:
                prompt_messages.append({
                    "role": "system",
                    "content": "Relevant high-importance memories:\n" + "\n".join(important_hits)
                })

            # -------------------------
            # Call DeepSeek (Ollama)
            # -------------------------
            response = ollama.chat(model=self.model, messages=prompt_messages)
            reply = response["message"]["content"]

            # -------------------------
            # Persist BOTH turns (user + assistant), ALWAYS save
            # -------------------------
            self.messages.append({"role": "user", "content": user_text})
            self.messages.append({"role": "assistant", "content": reply})
            self._save_messages()

            # -------------------------
            # Memory extraction + storage (profile used ONLY here)
            # -------------------------
            profile = self._load_profile()
            profile_json = json.dumps(profile, ensure_ascii=False, indent=2)
            extracted = extract_memories(user_text, reply, profile_json)

            if extracted:
                # Entities
                for e in extracted.get("entities", []):
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
                        name = str(e).strip()
                        etype = "thing"
                        imp = 0.4

                    if name and imp >= 0.45:
                        store_item(
                            text=f"{name} ({etype})",
                            tag="entity",
                            kind="entity",
                            importance=imp,
                            meta={"entity_type": etype},
                        )

                # Facts
                for fact in extracted.get("facts", []):
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

                    if txt and imp >= 0.60:
                        store_item(txt, tag=tag, kind="memory", importance=imp)

                # Open loops
                for loop in extracted.get("open_loops", []):
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

            # Profile update step (unchanged)
            updated_profile = self.analyze_user_input(user_text)

            return {"type": "chat", "reply": reply, "profile_updated": bool(updated_profile)}
