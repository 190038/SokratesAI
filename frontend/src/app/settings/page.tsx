import Link from "next/link";

export default function SettingsPage() {
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

  return (
    <div className="min-h-screen bg-white text-zinc-900">
      <header className="sticky top-0 z-20 border-b border-zinc-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-4 py-3">
          <div className="text-sm font-semibold">Settings</div>
          <Link className="rounded-full px-3 py-1.5 text-sm text-zinc-700 hover:bg-zinc-100" href="/chat">
            Zurück
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-4 py-6">
        <div className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
          <div className="text-sm text-zinc-600">
            Hier kannst du später Optionen hinzufügen (Model, Temperature, System Prompt, Memory-Optionen, usw.).
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-zinc-200 p-4">
              <div className="text-sm font-medium">API Base</div>
              <div className="mt-1 text-sm text-zinc-600">{API_BASE}</div>
            </div>

            <div className="rounded-xl border border-zinc-200 p-4">
              <div className="text-sm font-medium">Auth</div>
              <div className="mt-1 text-sm text-zinc-600">Derzeit deaktiviert</div>
            </div>

            <div className="rounded-xl border border-zinc-200 p-4 sm:col-span-2">
              <div className="text-sm font-medium">Geplant</div>
              <ul className="mt-2 list-disc pl-5 text-sm text-zinc-600">
                <li>Streaming Antworten</li>
                <li>Markdown Rendering (Codeblöcke)</li>
                <li>Chat Sessions (Sidebar)</li>
                <li>Model Auswahl + Temperature</li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}