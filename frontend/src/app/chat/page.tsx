"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";

type Msg = { role: "user" | "assistant"; content: string };

function cn(...c: (string | false | undefined)[]) {
  return c.filter(Boolean).join(" ");
}

export default function ChatPage() {
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

  const [messages, setMessages] = useState<Msg[]>([
    { role: "assistant", content: "Hi — wie kann ich helfen?" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const listRef = useRef<HTMLDivElement | null>(null);
  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, loading]);

  async function send() {
    if (!canSend) return;

    const userText = input.trim();
    setInput("");

    const next = [...messages, { role: "user", content: userText } as Msg];
    setMessages(next);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_text: userText }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data?.detail ? JSON.stringify(data.detail) : (data?.error || "Request failed");
        throw new Error(detail);
      }

      setMessages([...next, { role: "assistant", content: data.reply ?? "(keine Antwort)" }]);
    } catch (e: any) {
      setMessages([
        ...next,
        { role: "assistant", content: `⚠️ Backend-Fehler: ${String(e?.message ?? e)}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function clearChat() {
    setMessages([{ role: "assistant", content: "Neuer Chat — was möchtest du wissen?" }]);
  }

  return (
    <div className="min-h-screen bg-white text-zinc-900">
      {/* Top Bar */}
      <header className="sticky top-0 z-20 border-b border-zinc-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-full border border-zinc-200 bg-zinc-50 text-sm font-semibold">
              S
            </div>
            <div className="leading-tight">
              <div className="text-sm font-semibold">Sokrates</div>
              <div className="text-xs text-zinc-500">lokal · FastAPI · Ollama</div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={clearChat}
              className="rounded-full px-3 py-1.5 text-sm text-zinc-700 hover:bg-zinc-100"
            >
              New Chat
            </button>
            <Link
              href="/settings"
              className="rounded-full px-3 py-1.5 text-sm text-zinc-700 hover:bg-zinc-100"
            >
              Settings
            </Link>
          </div>
        </div>
      </header>

      {/* Chat */}
      <main className="mx-auto flex max-w-3xl flex-col px-4">
        <div className="pt-6" />

        <div
          ref={listRef}
          className="flex-1 space-y-4 overflow-auto pb-44"
          style={{ height: "calc(100vh - 64px)" }}
        >
          {messages.map((m, i) => (
            <div key={i} className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}>
              <div className="max-w-[85%]">
                <div
                  className={cn(
                    "whitespace-pre-wrap rounded-2xl px-4 py-3 text-[15px] leading-relaxed shadow-sm",
                    m.role === "user"
                      ? "bg-zinc-900 text-white"
                      : "border border-zinc-200 bg-zinc-50 text-zinc-900"
                  )}
                >
                  {m.content}
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl border border-zinc-200 bg-zinc-50 px-4 py-3 text-[15px] text-zinc-700 shadow-sm">
                <span className="inline-flex items-center gap-2">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-zinc-400" />
                  denkt nach…
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Composer */}
        <div className="fixed bottom-0 left-0 right-0 border-t border-zinc-200 bg-white/85 backdrop-blur">
          <div className="mx-auto max-w-3xl px-4 py-4">
            <div className="rounded-2xl border border-zinc-200 bg-white shadow-sm">
              <textarea
                className="w-full resize-none rounded-2xl bg-white px-4 py-3 text-[15px] outline-none placeholder:text-zinc-400"
                rows={3}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Schreib eine Nachricht…"
                disabled={loading}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
              />
              <div className="flex items-center justify-between px-3 pb-3">
                <div className="text-xs text-zinc-500">Enter = senden · Shift+Enter = neue Zeile</div>
                <button
                  onClick={send}
                  disabled={!canSend}
                  className={cn(
                    "rounded-xl px-4 py-2 text-sm font-medium transition",
                    canSend ? "bg-zinc-900 text-white hover:bg-zinc-800" : "bg-zinc-200 text-zinc-500"
                  )}
                >
                  Senden
                </button>
              </div>
            </div>

            <div className="mt-2 text-center text-[11px] text-zinc-500">
              Backend: {API_BASE}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}