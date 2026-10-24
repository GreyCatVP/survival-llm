"""Terminal chat with a local model that can search the offline library.

Usage: chat.py [--model NAME] [question ...]   (default: the first chat model Ollama has)
With a question it answers once and exits; without one it opens a chat.
In a chat: /sources lists what the last answer used, /open N opens one, /new, /think,
/limit normal|long|none sets how long an answer may run, /bye.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import LENGTH_LIMITS, capabilities, chat_events, installed_models, system_prompt, trim  # noqa: E402
from library.search import DB_PATH, Library  # noqa: E402

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def open_article(library, passage_id):
    """Write the article to a Markdown file, marking the cited passage, and open it."""
    doc = library.document(passage_id)
    if not doc:
        return
    lines = [f"# {doc['title']}", f"Source: {doc['source']}" + (f" | {doc['url']}" if doc["url"] else ""), ""]
    for part in doc["parts"]:
        if part["id"] == passage_id:
            lines += ["", ">>> THE PASSAGE THE ANSWER CAME FROM <<<", ""]
        lines.append(part["text"])
        lines.append("")
    path = os.path.join(tempfile.gettempdir(), f"survival-llm-article-{doc['doc_id']}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if sys.platform == "darwin":
        subprocess.run(["open", "-t", path])
    elif os.name == "nt":
        os.startfile(path)
    else:
        subprocess.run(["xdg-open", path])
    print(f"{DIM}  · opened: {doc['title']} ({doc['source']}){RESET}")


def run_turn(model, library, messages, deep, caps, length="normal"):
    """Print one answer as it streams; returns the sources it used."""
    sources = []
    fresh = True  # nothing printed since the last lookup; skip the blank lines models emit before tool calls
    for ev in chat_events(model, library, messages, deep=deep, caps=caps, length=length):
        kind = ev["type"]
        if kind not in ("token", "tick"):
            fresh = True
        if kind == "thinking":
            print(f"{DIM}  · thinking{RESET}", flush=True)
        elif kind == "search":
            topic = f" [{ev['topic']}]" if ev["topic"] != "any" else ""
            print(f"{DIM}  · searching: {ev['query']}{topic}{RESET}", flush=True)
        elif kind == "read":
            print(f"{DIM}  · reading: {ev['title']} ({ev['source']}){RESET}", flush=True)
        elif kind == "show":
            open_article(library, ev["id"])
        elif kind == "token":
            if fresh and not ev["text"].strip():
                continue
            fresh = False
            print(ev["text"], end="", flush=True)
        elif kind == "done":
            sources = ev["sources"]
            speed = f" · {ev['tokens'] / ev['seconds']:.0f} tok/s" if ev["seconds"] else ""
            note = " · /sources to see what it read" if sources else ""
            if ev.get("cut_off"):
                print(f"\n{DIM}[stopped: the answer ran too long; try asking again]{RESET}")
            elif ev.get("unfinished"):
                print(f"\n{DIM}[it kept looking things up and did not finish; try asking again]{RESET}")
            print(f"\n{DIM}[{ev['tokens']} tokens{speed}{note}]{RESET}")
    return sources


def main():
    args = sys.argv[1:]
    model = None
    if args[:1] == ["--model"]:
        model, args = args[1], args[2:]
    if not model:
        names = [m for m in installed_models() or [] if not any(x in m for x in ("embed", "bge-", "nomic", "minilm"))]
        if not names:
            sys.exit("No AI model found. Start Ollama, download a model (ollama pull granite4.2:8b), or use --model.")
        model = names[0]
    library = None
    if os.path.exists(DB_PATH):
        try:
            library = Library()
            if library.stats()[0] == 0:
                library = None
        except Exception:
            library = None
    caps = capabilities(model)
    messages = [{"role": "system", "content": system_prompt(library, "tools" in caps)}]
    deep, length = False, "normal"

    if args:  # one question, one answer
        messages.append({"role": "user", "content": " ".join(args)})
        run_turn(model, library, messages, deep, caps)
        return

    try:
        import readline  # noqa: F401  (arrow keys and history in input())
    except ImportError:
        pass
    docs, n_sources = library.stats() if library else (0, 0)
    lib_note = f"library: {docs:,} articles from {n_sources} sources" if library else "no library found"
    print(f"{BOLD}{model.removesuffix(':latest')}{RESET} {DIM}· offline · {lib_note}{RESET}")
    print(f"{DIM}Ask anything. /sources shows what the last answer read, /open N opens one, "
          f"/new starts over, /think toggles deep thinking, /limit sets the answer length limit, /bye quits.{RESET}")
    last_sources = []
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question in ("/bye", "/quit", "/exit"):
            return
        if question == "/new":
            messages, last_sources = messages[:1], []
            print(f"{DIM}Started over.{RESET}")
            continue
        if question == "/think":
            deep = not deep
            print(f"{DIM}Deep thinking {'on (slower, more careful)' if deep else 'off'}.{RESET}")
            continue
        if question.startswith("/limit"):
            pick = question[6:].strip()
            if pick in LENGTH_LIMITS:
                length = pick
                print(f"{DIM}Length limit: {pick}{' (use Ctrl+C to stop a stuck answer)' if pick == 'none' else ''}.{RESET}")
            else:
                print(f"{DIM}Use /limit normal, /limit long or /limit none (now: {length}).{RESET}")
            continue
        if question == "/sources":
            if not last_sources:
                print(f"{DIM}The last answer did not read anything from the library.{RESET}")
            for n, s in enumerate(last_sources, 1):
                print(f"  {n}. {s['title']} ({s['source']})")
            continue
        if question.startswith("/open"):
            pick = question[5:].strip() or "1"
            if pick.isdigit() and 1 <= int(pick) <= len(last_sources) and library:
                open_article(library, last_sources[int(pick) - 1]["id"])
            else:
                print(f"{DIM}Use /open N, where N is a number from /sources.{RESET}")
            continue
        messages.append({"role": "user", "content": question})
        trim(messages)
        print()
        try:
            last_sources = run_turn(model, library, messages, deep, caps, length) or last_sources
        except KeyboardInterrupt:
            print(f"\n{DIM}[stopped]{RESET}")
        except OSError as exc:
            print(f"\n{DIM}[could not reach the local model: {exc}. Is Ollama running? Try: ollama serve]{RESET}")


if __name__ == "__main__":
    main()
