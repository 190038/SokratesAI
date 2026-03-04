# main_refactored.py
"""
CLI runner that uses the same engine as the FastAPI server.

If you want, rename this to main.py after you verify it works.

Run:
  python main_refactored.py
"""

from sokrates_app import SokratesApp


def run_cli():
    app = SokratesApp(model="deepseek-r1:8b")
    print("Sokrates ready. Type 'exit' to quit.\n")

    while True:
        user = input("You: ").strip()
        out = app.handle_turn(user)

        if out.get("type") == "exit":
            break

        if out.get("type") == "decay_dry_run":
            report = out.get("report", {})
            print(f"\n[Decay dry run]")
            print(f"  Would decay:  {report.get('decayed')}")
            print(f"  Would prune:  {report.get('pruned')}")
            print(f"  Skipped:      {report.get('skipped')}")
            for item in report.get("report", []):
                print(
                    f"  [{item['action'].upper():5}] "
                    f"idle={item['days_idle']}d  "
                    f"imp {item['old_imp']} → {item['new_imp']}  "
                    f"tag={item['tag']:15}  "
                    f"\"{item['text']}\""
                )
            print()
            continue

        print("\nSokrates:", out.get("reply", ""), "\n")


if __name__ == "__main__":
    run_cli()
