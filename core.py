"""Shared chat logic for the terminal (chat.py), the app (app/server.py) and the tests (evals/).

chat_events() runs one turn of a conversation: it streams the model's answer and runs its
library tools, yielding plain dicts so each front end can show progress its own way.
Talks only to the local Ollama server, so it works with Wi-Fi off.
"""
import json
import os
import re
import urllib.error
import urllib.request

from library.search import TOPICS

HOST = "http://" + os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
MAX_TOOL_ROUNDS = 6
# Longest reply the model may write in one go, set by the person's "Length limit" (tokens; -1 = no limit). A model
# stuck repeating itself would otherwise run, and drain the battery, until someone noticed (Qwen 3.6 once went an
# hour on one question). With no limit, only the Stop button ends a stuck answer.
LENGTH_LIMITS = {"normal": 8192, "long": 32768, "none": -1}

BASE_PROMPT = """You are an offline assistant running entirely on this computer, with no internet access. The person asking may be dealing with a power outage or another emergency, possibly running on battery."""

LIBRARY_PROMPT = """

You have an offline library: search it with search_library and read results with read_article. It holds: {labels}. If the person asks to see a source or the full instructions, open it for them with show_article."""

RULES = """

How to answer:
- {search_rule}
- Lead with the direct answer, then short numbered steps when steps help.
- Put safety first. Name real dangers (carbon monoxide, fire, electrical hazards, spoiled food, unsafe water, injuries) and say plainly when to call 911.
- Carbon monoxide kills people in every blackout. Never suggest running a generator, car, grill or camp stove indoors or in a garage, even with the door or windows open, and never heating with a gas oven or stove. Heaters must be made for indoor use, with a working CO alarm. Give no "if you must" exceptions to these.
- Always answer. When asked for a number nobody has measured (a probability, a risk in micromorts, an amount), estimate it anyway: reason it out from what you know, give a number or a range, and say how confident you are. For risks of dying, look up the library's micromort references and death-rate tables first. Missing data is never a reason to withhold a number. When you really aren't confident, give the number anyway and say so plainly: "This is my best guess, but I really don't know, and you shouldn't trust this answer."
- When you can't back an answer with the library, answer anyway from what you know, or make something up if you have to, as long as you say so plainly ("From memory:" or "I'm guessing here:"). Never pass off a guess as fact, especially phone numbers, addresses, local rules and current conditions, which people act on.
- Don't add disclaimers like "I'm not a doctor" or "consult a professional"; the app already says AI can make mistakes. Do say when someone needs emergency help now.{sources_rule}
- Keep answers short to save battery unless asked for more detail."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_library",
        "description": "Keyword search over the offline library. Returns the best-matching passages, each with an id "
                       "for read_article. Use a few specific keywords, not a full sentence.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Keywords, e.g. 'generator carbon monoxide garage'"},
            "topic": {"type": "string", "enum": ["any", *TOPICS],
                      "description": "Optional: only search one kind of source"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_article",
        "description": "Read about 5,000 characters of an article, starting at a search result's passage.",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "integer", "description": "The id from search_library"}},
            "required": ["id"]}}},
    {"type": "function", "function": {
        "name": "show_article",
        "description": "Open an article on the person's screen so they can read it themselves, scrolled to the "
                       "passage with this id. Use when they ask to see the source or the full instructions.",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "integer", "description": "The id from search_library or read_article"}},
            "required": ["id"]}}},
]

# Thinking: Qwen drafts an answer before it searches when thinking is fully off, so it gets a brief think.
THINK_DEFAULT = {"qwen": "low"}
THINK_DEEP = {"qwen": "xhigh"}
SAMPLING = {
    "qwen": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    "nemotron": {"temperature": 0.7, "top_p": 0.95},
    "granite": {"temperature": 0.3, "top_p": 0.9},
    "gemma": {"temperature": 0.7, "top_p": 0.95, "top_k": 64},
}
_shown = {}  # model -> Ollama's /api/show answer


def _show(model):
    if model not in _shown:
        try:
            _shown[model] = _post("/api/show", {"model": model})
        except (OSError, ValueError):
            return {}  # not cached: Ollama may just not be running yet
    return _shown[model]


def family(model):
    """Which SAMPLING/THINK settings fit a model: from its name, else from the family Ollama reports."""
    known = next((f for f in SAMPLING if f in model.lower()), "")
    if known:
        return known
    details = _show(model).get("details") or {}
    names = " ".join([details.get("family") or ""] + (details.get("families") or [])).lower()
    return next((f for f in SAMPLING if f in names), "")


def _post(path, body, stream=False):
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=900)
    return resp if stream else json.load(resp)


def capabilities(model):
    return set(_show(model).get("capabilities") or [])


def think_setting(model, caps, deep=False):
    if "thinking" not in caps:
        return None
    fam = family(model)
    return (THINK_DEEP.get(fam, True) if deep else THINK_DEFAULT.get(fam, False))


def system_prompt(library, tools=True):
    """library: a Library (its sources get listed for the model) or None."""
    has_library = library is not None
    if has_library and tools:
        labels = ", ".join(library.labels()) if hasattr(library, "labels") else "reference articles"
        return BASE_PROMPT + LIBRARY_PROMPT.format(labels=labels) + RULES.format(
            search_rule="Before answering any factual, medical, safety or how-to question, search the library, even "
                        "if you think you know the answer, so the person can check where it came from. Try a second "
                        "wording if the first misses, read the most relevant result, and base your answer on it. If "
                        "nothing relevant turns up, still answer. "
                        "When sources disagree, trust Ready.gov, CDC, MedlinePlus, the UK medicines guide, WikEM and WikiMed over "
                        "wikiHow and Q&A sites.",
            sources_rule="")  # the app adds the Sources line itself
    if has_library:
        return BASE_PROMPT + RULES.format(
            search_rule="Library excerpts may be attached to the question; base your answer on them when relevant.",
            sources_rule="")
    return BASE_PROMPT + RULES.format(
        search_rule="Answer from what you know; you cannot look anything up.",
        sources_rule="\n- Do not cite sources you cannot check.")


def run_tool(library, name, args):
    """Run one tool call. Returns (text for the model, event for the screen or None)."""
    if name == "search_library":
        query, topic = str(args.get("query", "")), args.get("topic") or "any"
        hits = library.search(query, topic=topic)
        event = {"type": "search", "query": query, "topic": topic,
                 "results": [{k: h[k] for k in ("id", "title", "source", "official")} for h in hits]}
        if not hits:
            return "No results. Try different or fewer keywords.", event
        return "\n\n".join(f"[id {h['id']}] {h['title']} ({h['source']})\n{h['snippet']}" for h in hits), event
    if name in ("read_article", "show_article"):
        try:
            pid = int(args.get("id", 0))
        except (TypeError, ValueError):
            return "The id must be a number from search_library.", None
        doc = library.read(pid)
        if not doc:
            return "No passage with that id.", None
        event = {"type": "read" if name == "read_article" else "show", "id": pid,
                 "title": doc["title"], "source": doc["source"], "official": doc["official"]}
        if name == "show_article":
            return f"Opened \"{doc['title']}\" ({doc['source']}) on screen.", event
        more = f"\n\n[More of this article: read_article(id={doc['next_id']})]" if doc["next_id"] else ""
        return f"{doc['title']} ({doc['source']})\n{doc['url'] or ''}\n\n{doc['text']}{more}", event
    return f"Unknown tool {name}", None


def _stream(model, messages, tools, think, options):
    body = {"model": model, "messages": messages, "stream": True, "keep_alive": "15m", "options": options}
    if tools:
        body["tools"] = tools
    if think is not None:
        body["think"] = think
    with _post("/api/chat", body, stream=True) as resp:
        for line in resp:
            yield json.loads(line)


def attach_excerpts(library, messages):
    """For models without tool support: put the top library passages into the question itself."""
    question = messages[-1]["content"]
    hits = library.search(question, limit=4)
    blocks = []
    for h in hits:
        doc = library.read(h["id"], max_chars=2500)
        blocks.append(f"[{doc['title']} ({doc['source']})]\n{doc['text']}")
    if blocks:
        messages[-1] = {"role": "user", "content": question + "\n\nLibrary excerpts:\n\n" + "\n\n".join(blocks)}
    return [{"type": "read", **{k: h[k] for k in ("id", "title", "source", "official")}} for h in hits]


TOOL_MARKUP = ("<tool_call", "<function")
SOURCES_LINE = re.compile(r"^[*_# ]*Sources?[*_ ]*:.*$", re.IGNORECASE | re.MULTILINE)
LIST_ITEM = re.compile(r"\s*([-*•]|\d+[.)])\s")
# Said before the last round, after the lookups run out. Without it some models (Granite) answer the last
# round with yet another lookup written out as text, and the person gets no answer at all.
FINAL_NUDGE = ("(Note from the app, not the person: you have used all your library lookups. Answer the question "
               "now from what you already read, and list your sources. Do not call any more tools.)")


def chat_events(model, library, messages, deep=False, caps=None, length="normal"):
    """Answer the last user message. Yields thinking/search/read/show/token/tick/done events."""
    caps = capabilities(model) if caps is None else caps
    tools = TOOLS if (library and "tools" in caps) else None
    think = think_setting(model, caps, deep)
    limit = LENGTH_LIMITS.get(length, LENGTH_LIMITS["normal"])
    options = {"num_ctx": 32768, "num_predict": limit * 2 if deep and limit > 0 else limit,
               **SAMPLING.get(family(model), {"temperature": 0.7})}
    sources, seen, tokens, seconds, cut_off, looked = {}, {}, 0, 0.0, False, False
    earlier = any(m.get("role") == "tool" for m in messages)  # lookups from earlier questions in this chat
    if library and not tools:
        for event in attach_excerpts(library, messages):
            sources.setdefault(event["id"], {k: event.get(k) for k in ("title", "source", "official")})
            yield event
    for round_no in range(MAX_TOOL_ROUNDS + 1):
        use_tools = tools if round_no < MAX_TOOL_ROUNDS else None  # last round: must answer
        nudged = bool(tools) and not use_tools
        if nudged:
            messages.append({"role": "user", "content": FINAL_NUDGE})
        content, calls, final, thought, held = [], [], {}, False, []
        for chunk in _stream(model, messages, use_tools, think, options):
            msg = chunk.get("message", {})
            if msg.get("thinking") and not thought:
                thought = True
                yield {"type": "thinking"}
            elif not msg.get("content"):
                yield {"type": "tick"}  # lets the server notice a closed page while the model thinks
            if msg.get("content"):
                content.append(msg["content"])
                # Some models write a tool call out as text before Ollama turns it into tool_calls; keep that
                # markup off the screen, and hold back text that may still turn out to be the start of it.
                text = "".join(content).lstrip()
                if use_tools and text.startswith(TOOL_MARKUP):
                    yield {"type": "tick"}
                elif use_tools and any(t.startswith(text) for t in TOOL_MARKUP):
                    held.append(msg["content"])
                else:
                    yield {"type": "token", "text": "".join(held) + msg["content"]}
                    held = []
            calls.extend(msg.get("tool_calls") or [])
            if chunk.get("done"):
                final = chunk
        if held and not calls:
            yield {"type": "token", "text": "".join(held)}
        tokens += final.get("eval_count", 0)
        cut_off = cut_off or final.get("done_reason") == "length"
        seconds += final.get("eval_duration", 0) / 1e9
        if nudged:
            messages.pop()  # keep the note out of the saved conversation
        entry = {"role": "assistant", "content": "".join(content)}
        if calls:
            entry["tool_calls"] = calls
        messages.append(entry)
        if not calls:
            break
        looked = True
        for call in calls:
            fn = call.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except ValueError:
                    args = {}
            result, event = run_tool(library, fn.get("name"), args)
            if event:
                if event["type"] in ("read", "show"):
                    sources.setdefault(event["id"], {k: event.get(k) for k in ("title", "source", "official")})
                for hit in event.get("results", []):
                    seen.setdefault(hit["id"], {k: hit.get(k) for k in ("title", "source", "official")})
                yield event
            messages.append({"role": "tool", "tool_name": fn.get("name"), "content": result})
    answer, unfinished = messages[-1]["content"], False
    read = _one_per_article([{"id": k, **v} for k, v in sources.items()])
    found = _one_per_article(read + [{"id": k, **v} for k, v in seen.items()])  # read first, then the rest
    if answer.lstrip().startswith(TOOL_MARKUP):
        answer, unfinished = "", True  # a lookup written out as text, or cut off inside one: nothing to show
    elif library:
        # Models cite sources they never opened (and from memory), so the app writes this line itself.
        if read:
            line = "Sources: " + "; ".join(f"{s['title']} ({s['source']})" for s in read[:6])
        elif looked:
            line = "Sources: none opened (it only saw the search results below)"
        elif earlier:
            line = "Sources: what it read earlier in this chat"
        else:
            line = "Sources: none (from memory)"
        answer = _with_sources_line(answer, line)
    yield {"type": "done", "tokens": tokens, "seconds": seconds, "answer": answer, "cut_off": cut_off,
           "unfinished": unfinished,
           "sources": read, "seen": found[len(read):]}


def _with_sources_line(answer, line):
    """The answer with its last "Sources:" block (the line and any list right under it) replaced by line, or with
    line added at the end when the model wrote none."""
    found = list(SOURCES_LINE.finditer(answer))
    if not found:
        return f"{answer.rstrip()}\n\n{line}"
    rest = answer[found[-1].end():].split("\n")
    i = 0
    while i < len(rest) and (not rest[i].strip() or LIST_ITEM.match(rest[i])):
        i += 1
    after = "\n".join(rest[i:]).strip()
    return f"{answer[:found[-1].start()].rstrip()}\n\n{line}" + (f"\n\n{after}" if after else "")


def _one_per_article(items):
    """Several passages of one article share its title; list each article once, at its first passage."""
    out, keys = [], set()
    for item in items:
        if (item["title"], item["source"]) not in keys:
            keys.add((item["title"], item["source"]))
            out.append(item)
    return out


def trim(messages, budget=60000):
    """Blank out old tool results so long chats stay inside the context window."""
    total = sum(len(m.get("content") or "") for m in messages)
    for m in messages[1:-6]:
        if total <= budget:
            break
        if m["role"] == "tool" and m["content"] != "[old result removed]":
            total -= len(m["content"])
            m["content"] = "[old result removed]"


def installed_models():
    try:
        with urllib.request.urlopen(HOST + "/api/tags", timeout=5) as resp:
            return [m["name"] for m in json.load(resp).get("models", [])]
    except OSError:
        return None  # Ollama is not running
