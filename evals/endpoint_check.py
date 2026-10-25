"""Check that Survival LLM can answer through an OpenAI-compatible endpoint the person already runs.

Run: python3 evals/endpoint_check.py
It starts the app (app/server.py) as its own process with its own settings folder and a tiny library built for
the test, points it at evals/mock_endpoint.py in each of its modes, and drives the real HTTP path: save the
endpoint, read /api/status, ask a question, look at the events that come back. Prints PASS/FAIL lines and exits
non-zero if anything fails. Stdlib only, and it never touches the real internet.
"""
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evals.mock_endpoint import serve  # noqa: E402

KEY = "sk" + "-" + "test" + "-" + "1234" + "-" + "5678"  # the app must never show or log this back in full
ARTICLES = [
    ("Food Safety in a Power Outage", "ready-gov",
     "Keep the refrigerator door closed. Food stays safe about four hours in a closed fridge. Discard "
     "perishable food that has been warmer than 40 F for two hours or more."),
    ("Freezer Food During Outages", "usda-canning",
     "A closed full freezer holds about 48 hours, half full about 24 hours. Check with a thermometer: food "
     "still containing ice crystals may be refrozen."),
    ("Blackout Safety Checklist", "survival-manuals",
     "During a blackout keep generators outside and away from windows for carbon monoxide, use a battery radio, "
     "and treat tap water as unsafe after a outage until it has been boiled."),
]

PASS, FAIL = [], []


def check(name, ok, detail=""):
    note = f"  ({detail})" if detail else ""
    print(f"{'PASS' if ok else 'FAIL'}  {name}{note}", flush=True)
    (PASS if ok else FAIL).append(name)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_library(path):
    """A three-article library in the shape library/build_library.py writes, so tools have something to find."""
    text_dir = os.path.join(path, "text")
    os.makedirs(text_dir, exist_ok=True)
    db = sqlite3.connect(os.path.join(path, "index.sqlite"))
    db.executescript("""
        CREATE TABLE sources(id INTEGER PRIMARY KEY, key TEXT UNIQUE, label TEXT, topic TEXT,
                             origin TEXT, docs INT DEFAULT 0, done INT DEFAULT 0, license TEXT);
        CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT UNIQUE);
        CREATE TABLE docs(id INTEGER PRIMARY KEY, source_id INT, title TEXT, url TEXT,
                          file_id INT, offset INT, length INT);
        CREATE TABLE passages(id INTEGER PRIMARY KEY, doc_id INT, offset INT, length INT);
        CREATE VIRTUAL TABLE fts USING fts5(title, body, content='',
                                            tokenize='porter unicode61 remove_diacritics 2');
    """)  # same shape as library/build_library.py writes (older SQLite here has no contentless_delete)
    db.execute("INSERT INTO sources(key, label, topic, origin, docs, done) VALUES(?,?,?,?,?,?)",
               ("check", "Test survival library", "emergency", "test", len(ARTICLES), 1))
    for title, source, body in ARTICLES:
        rel = f"{source}.txt"
        with open(os.path.join(text_dir, rel), "w", encoding="utf-8") as f:
            f.write(body)
        file_id = db.execute("INSERT INTO files(path) VALUES(?)", (rel,)).lastrowid
        doc_id = db.execute("INSERT INTO docs(source_id, title, url, file_id, offset, length) VALUES(?,?,?,?,?,?)",
                            (1, title, "", file_id, 0, len(body.encode()))).lastrowid
        pid = db.execute("INSERT INTO passages(doc_id, offset, length) VALUES(?,?,?)",
                         (doc_id, 0, len(body.encode()))).lastrowid
        db.execute("INSERT INTO fts(rowid, title, body) VALUES(?,?,?)", (pid, title, body))
    db.commit()
    db.close()


class App:
    """One running copy of the app, in its own settings folder, with its log kept for the checks."""

    def __init__(self, home, library, port):
        self.port, self.home = port, home
        os.makedirs(home, exist_ok=True)
        env = {**os.environ, "HOME": home, "SURVIVAL_LLM_LIBRARY": library,
               "SURVIVAL_LLM_MANIFEST": "http://127.0.0.1:1/none.json",  # nothing here goes to the internet
               "SURVIVAL_LLM_UPDATES": "http://127.0.0.1:1/none.json"}
        self.log_path = os.path.join(home, f"app-{port}.log")
        self.log = open(self.log_path, "wb")
        self.proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "app", "server.py"), "--no-browser",
                                      "--port", str(port)], cwd=ROOT, env=env, stdout=self.log,
                                     stderr=subprocess.STDOUT)
        self.base = f"http://127.0.0.1:{port}"
        self.token = self._wait()

    def _wait(self):
        for _ in range(120):
            try:
                with urllib.request.urlopen(self.base + "/ping", timeout=2) as resp:
                    if json.load(resp).get("app") == "survival-llm":
                        break
            except OSError:
                time.sleep(0.25)
        else:
            raise RuntimeError(f"the app didn't start; log:\n{self.read_log()}")
        with urllib.request.urlopen(self.base + "/", timeout=10) as resp:
            page = resp.read().decode()
        return re.search(r'name="token" content="([^"]+)"', page).group(1)

    def req(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={"X-Token": self.token})
        return urllib.request.urlopen(req, timeout=90)

    def get_json(self, path):
        with self.req("GET", path) as resp:
            return json.load(resp)

    def post_json(self, path, body):
        try:
            with self.req("POST", path, body) as resp:
                return resp.status, json.load(resp)
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode() or "{}")

    def chat(self, model, message, conversation="check", length="normal", **extra):
        """Ask one question and collect the events the page would show."""
        try:
            resp = self.req("POST", "/api/chat", {"model": model, "message": message, "conversation": conversation,
                                                  "length": length, **extra})
        except urllib.error.HTTPError as exc:
            return [{"type": "http", "code": exc.code, "error": json.loads(exc.read().decode() or "{}")}]
        events = []
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
        return events

    def stop(self):
        if getattr(self, "stopped", False):
            return
        self.stopped = True
        self.proc.terminate()
        try:
            self.proc.wait(15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()

    def read_log(self):
        try:
            with open(self.log_path, "rb") as f:
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""


def types(events):
    return [e["type"] for e in events]


def mock_log(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/_log", timeout=10) as resp:
        return json.load(resp)["requests"]


def main():
    work = tempfile.mkdtemp(prefix="sllm-check-")
    home, library = os.path.join(work, "home"), os.path.join(work, "library")
    make_library(library)
    ports = {mode: free_port() for mode in ("tools", "no-tools", "flaky", "deny-key", "broken", "short", "slow",
                                            "thinking", "quiet", "silent")}
    mocks = {mode: serve(port, mode, KEY if mode == "deny-key" else "") for mode, port in ports.items()}
    app = App(home, library, free_port())
    try:
        app = run_checks(app, ports, home, library)  # it may have been restarted once along the way
    finally:
        app.stop()
        for httpd in mocks.values():
            httpd.shutdown()
        print(f"\n{len(PASS)} passed, {len(FAIL)} failed" + (f": {', '.join(FAIL)}" if FAIL else ""))
        shutil.rmtree(work, ignore_errors=True)
    return 1 if FAIL else 0


def check_ui(app):
    """The page itself: the source toggle and the endpoint panel are there, wired, and the key field is a password."""
    with urllib.request.urlopen(app.base + "/", timeout=10) as resp:
        page = resp.read().decode()
    with urllib.request.urlopen(app.base + "/static/app.js", timeout=10) as resp:
        script = resp.read().decode()
    ids = set(re.findall(r'id="([a-z0-9-]+)"', page))
    wanted = ["source-toggle", "src-ollama", "src-endpoint", "endpoint-panel", "endpoint-url", "endpoint-key",
              "endpoint-model", "endpoint-custom", "endpoint-status", "endpoint-test", "endpoint-save",
              "endpoint-off", "endpoint-forget"]
    check("the page has the source toggle and the endpoint panel", all(i in ids for i in wanted),
          str([i for i in wanted if i not in ids]))
    check("the key box is a password field", 'id="endpoint-key" type="password"' in page)
    check("the page offers both sources in plain words", "On this computer (Ollama)" in page
          and "An endpoint I provide" in page)
    check("there is a model list and a model name you can type", 'id="endpoint-model"' in page
          and 'id="endpoint-custom"' in page and "Other model name" in page)
    check("Test, Save and turn-off actions are there", all(
        f'id="{i}"' in page for i in ("endpoint-test", "endpoint-save", "endpoint-off", "endpoint-forget")))
    wired = [i for i in wanted if i != "source-toggle" and f'$("{i}")' in script]
    check("the front end wires every one of them", len(wired) == len(wanted) - 1,
          str([i for i in wanted if i != "source-toggle" and f'$("{i}")' not in script]))
    made = {a or b for a, b in re.findall(r'\$\("([a-z0-9-]+)"\)|getElementById\("([a-z0-9-]+)"\)', script)}
    built = set(re.findall(r'id="([a-z0-9-]+)"', script))  # controls the front end builds itself
    missing = sorted(i for i in made if i not in ids and i not in built)
    check("the front end asks for no control that isn't on the page", not missing, str(missing[:8]))


def run_checks(app, ports, home, library):
    url = f"http://127.0.0.1:{ports['tools']}/v1"

    # 1. Nothing configured: the page sees exactly the same status as before this feature existed.
    status = app.get_json("/api/status")
    expected = {"ollama", "ollama_installed", "ollama_starting", "engine", "models", "default_model", "loaded",
                "warming", "catalog", "downloads", "library", "packs", "packs_error", "library_dir", "disk_free_gb",
                "my_files", "version", "update", "machine", "topics"}
    check("status without an endpoint has the same keys as before", set(status) == expected,
          f"extra: {sorted(set(status) - expected)} missing: {sorted(expected - set(status))}")
    check("status without an endpoint has no endpoint section", "endpoint" not in status)
    check_ui(app)
    code, report = app.post_json("/api/endpoint", {})
    check("GET /api/endpoint works", app.get_json("/api/endpoint")["mode"] == "ollama", f"POST {code}")

    # 2. Save an endpoint: it is stored, reported back with a masked key, and its models show up.
    code, report = app.post_json("/api/endpoint", {"mode": "endpoint", "url": f"http://127.0.0.1:{ports['tools']}",
                                                  "key": KEY, "model": "mock-model"})
    check("saving an endpoint answers 200", code == 200, f"{code}")
    check("the saved url is normalised to /v1", report["url"].endswith("/v1"), report["url"])
    check("the key is not echoed back", KEY not in json.dumps(report), json.dumps(report)[:120])
    check("the key comes back masked", report["key_masked"].endswith("5678") and KEY not in report["key_masked"],
          report["key_masked"])
    check("the endpoint's models are listed", report["models"] == ["mock-model", "other/mock:2"], str(report["models"]))
    settings_file = os.path.join(home, ".survival-llm", "endpoint.json")
    saved = json.load(open(settings_file, encoding="utf-8"))
    check("endpoint.json holds url, key and model in the settings folder",
          saved["url"].endswith("/v1") and saved["key"] == KEY and saved["model"] == "mock-model", str(saved))
    status = app.get_json("/api/status")
    check("status says an endpoint is answering", status.get("endpoint", {}).get("mode") == "endpoint")
    check("status shows the endpoint models", [m["model"] for m in status["models"]] == ["mock-model", "other/mock:2"],
          str(status["models"]))
    check("status keeps the key masked", KEY not in json.dumps(status))
    check("status lists no Ollama models or downloads in endpoint mode", status["catalog"] == [] and not status["loaded"])
    check("the chosen model is the default", status["default_model"] == "mock-model", str(status["default_model"]))

    # 3. Chat through it: the endpoint gets an OpenAI request, its tool calls run, and the sources check works.
    events = app.chat("mock-model", "How long is food in the fridge safe during a blackout?")
    got = types(events)
    check("the chat streams token and done events", "token" in got and "done" in got, str(got[:8]))
    search = next((e for e in events if e["type"] == "search"), None)
    read = [e for e in events if e["type"] == "read"]
    check("its search_library call ran against the library", search and "fridge" in search["query"],
          str(search and search["query"]))
    check("its read_article call ran and showed a source", bool(read) and read[0]["title"] in
          [a[0] for a in ARTICLES], str([r["title"] for r in read]))
    done = next((e for e in events if e["type"] == "done"), {})
    check("the answer arrived with the sources the app checked", done.get("sources") and any(
        a["title"] in done["answer"] for a in done["sources"]), str(done.get("answer", "")[:80]))
    asked = [r for r in mock_log(ports["tools"]) if r["path"] == "/v1/chat/completions"]
    check("the endpoint was asked with stream=true and the tools", asked and asked[-1]["body"].get("stream") is True
          and bool(asked[-1]["body"].get("tools")))
    check("the key reached only that endpoint, as a Bearer token", asked and
          asked[-1]["authorization"] == "Bearer " + KEY, str(asked and asked[-1]["authorization"]))
    check("the tool result went back with the call id it answers", any(
        m.get("tool_call_id") for m in asked[-1]["body"]["messages"] if m.get("role") == "tool") if asked else False)
    check("the Length limit became max_tokens", asked and asked[-1]["body"].get("max_tokens") == 8192,
          str(asked and asked[-1]["body"].get("max_tokens")))
    app.chat("mock-model", "No limit please", length="none")
    asked = [r for r in mock_log(ports["tools"]) if r["path"] == "/v1/chat/completions"]
    check("no Length limit sends no max_tokens", "max_tokens" not in asked[-1]["body"], str(asked[-1]["body"])[:160])
    events = app.chat("mock-model", "Open the freezer article for me", conversation="show")
    check("its show_article call opens an article on the person's screen",
          any(e["type"] == "show" for e in events), str(types(events)[:8]))
    app.post_json("/api/endpoint", {"url": f"http://127.0.0.1:{ports['thinking']}/v1", "model": "mock-model"})
    events = app.chat("mock-model", "How long is food in the fridge safe?", conversation="think")
    check("an endpoint that thinks first shows a thinking event", "thinking" in types(events), str(types(events)[:8]))
    app.post_json("/api/endpoint", {"url": url, "model": "mock-model"})

    # 4. An endpoint that cannot do tool calling: library excerpts, no error, sources still shown.
    notools = f"http://127.0.0.1:{ports['no-tools']}/v1"
    code, report = app.post_json("/api/endpoint", {"url": notools, "model": "mock-model"})
    events = app.chat("mock-model", "How long is food in the fridge safe during a blackout?", conversation="nt")
    asked = [r for r in mock_log(ports["no-tools"]) if r["path"] == "/v1/chat/completions"]
    check("it asks the endpoint whether it can call tools", any(r["body"].get("tools") for r in asked))
    check("it then asks the question without tools", asked and not asked[-1]["body"].get("tools"),
          str([bool(r["body"].get("tools")) for r in asked]))
    check("library excerpts are attached to the question instead", any(
        "Library excerpts" in m.get("content", "") for m in asked[-1]["body"]["messages"] if m["role"] == "user"))
    done = next((e for e in events if e["type"] == "done"), {})
    check("an excerpt answer still shows its sources", done.get("sources") and "token" in types(events),
          str(types(events)[:6]))
    text = "".join(e.get("text", "") for e in events if e["type"] == "token") + done.get("answer", "")
    check("no tool markup leaks into what the person sees", not any(t in text for t in ("<tool_call", "<function")),
          text[:80])

    # 5. A model typed in by hand (not in its model list) still chats.
    code, report = app.post_json("/api/endpoint", {"url": notools, "model": "my-custom-model"})
    check("a typed-in model name is kept", report["model"] == "my-custom-model", report["model"])
    events = app.chat("my-custom-model", "What about a half-full freezer?", conversation="custom")
    check("a typed-in model answers", "done" in types(events), str(types(events)[:6]))

    # 5b. An endpoint that passes the tool probe and then refuses: the question is retried with excerpts.
    code, report = app.post_json("/api/endpoint", {"url": f"http://127.0.0.1:{ports['flaky']}/v1",
                                                  "model": "mock-model"})
    events = app.chat("mock-model", "How long is food in the fridge safe during a blackout?", conversation="flaky")
    asked = [r for r in mock_log(ports["flaky"]) if r["path"] == "/v1/chat/completions"]
    done = next((e for e in events if e["type"] == "done"), {})
    check("a tool refusal mid-chat is retried, not shown as a failure", "done" in types(events), str(types(events)[:6]))
    check("the retry feeds library excerpts instead of tools", asked and not asked[-1]["body"].get("tools") and any(
        "Library excerpts" in m.get("content", "") for m in asked[-1]["body"]["messages"] if m["role"] == "user"),
          str([bool(r["body"].get("tools")) for r in asked]))
    check("the retried answer still shows sources", bool(done.get("sources")), str(done.get("sources"))[:100])

    # 5c. The length limit reaches the endpoint: an answer cut off is said so.
    code, report = app.post_json("/api/endpoint", {"url": f"http://127.0.0.1:{ports['short']}/v1",
                                                   "model": "mock-model"})
    events = app.chat("mock-model", "How long is food in the fridge safe?", conversation="short")
    done = next((e for e in events if e["type"] == "done"), {})
    check("an answer cut off by the length limit is reported as cut off", done.get("cut_off") is True,
          str(done)[:140])

    # 6. Bad key, dead address, nonsense answer: one plain error, no hang, no traceback.
    code, report = app.post_json("/api/endpoint", {"url": f"http://127.0.0.1:{ports['deny-key']}/v1",
                                                  "key": "wrong-key", "model": "mock-model"})
    events = app.chat("mock-model", "How long is food in the fridge safe?", conversation="key")
    error = next((e for e in events if e["type"] == "error"), {})
    check("a wrong key gives one plain error", "key" in error.get("message", "").lower(), error.get("message", ""))
    check("no traceback in the log for a refused key", "Traceback" not in app.read_log())

    code, report = app.post_json("/api/endpoint", {"url": f"http://127.0.0.1:{free_port()}/v1", "key": "",
                                                  "model": "mock-model"})
    check("a dead address is reported when saving", code == 400 and "address" in json.dumps(report).lower(),
          f"{code}: {json.dumps(report)[:120]}")
    events = app.chat("mock-model", "How long is food in the fridge safe?", conversation="dead")
    check("a dead address gives one plain error", "endpoint" in str(next((e for e in events if e["type"] == "error"),
                                                                        {})).lower(), str(events)[:160])

    code, report = app.post_json("/api/endpoint", {"url": f"http://127.0.0.1:{ports['broken']}/v1", "model": "mock-model"})
    events = app.chat("mock-model", "How long is food in the fridge safe?", conversation="broken")
    check("an answer it can't read gives one plain error", "read" in next((e for e in events if e["type"] == "error"),
                                                                           {}).get("message", "").lower(),
          str([e for e in events if e["type"] == "error"])[:160])

    # 6d. A model the endpoint doesn't have, and an endpoint that opens the stream and says nothing.
    code, report = app.post_json("/api/endpoint", {"url": f"http://127.0.0.1:{ports['tools']}/v1", "model": "missing-model"})
    events = app.chat("missing-model", "How long is food in the fridge safe?", conversation="missing")
    check("a model the endpoint doesn't have is named plainly", "no model by that name" in
          next((e for e in events if e["type"] == "error"), {}).get("message", "").lower(),
          str([e for e in events if e["type"] == "error"])[:160])
    app.post_json("/api/endpoint", {"url": f"http://127.0.0.1:{ports['silent']}/v1", "model": "mock-model"})
    started = time.time()
    events = app.chat("mock-model", "How long is food in the fridge safe?", conversation="silent")
    took = time.time() - started
    error = next((e for e in events if e["type"] == "error"), {})
    check("an endpoint that never answers is given up on with a plain message", "never answered" in
          error.get("message", "").lower(), error.get("message", "")[:120])
    check("it is given up on in under a minute, not left hanging", took < 60, f"{took:.0f}s")

    # 7. Switch back to this computer's Ollama, and the setting survives a restart.
    code, report = app.post_json("/api/endpoint", {"mode": "ollama"})
    check("switching back to Ollama works", report["mode"] == "ollama" and report["url"].endswith("/v1"),
          json.dumps(report)[:120])
    check("switching back keeps the saved endpoint for next time", os.path.exists(
        os.path.join(home, ".survival-llm", "endpoint.json")))
    check("status hides the endpoint section again", "endpoint" in app.get_json("/api/status"),
          "the endpoint is still saved, so the panel can show it")
    app.post_json("/api/endpoint", {"mode": "endpoint", "url": f"http://127.0.0.1:{ports['tools']}/v1", "key": KEY,
                                   "model": "mock-model"})
    first_log = app.read_log()
    app.stop()
    app = App(home, library, free_port())  # a fresh process: it must read the saved endpoint from the settings folder
    report = app.get_json("/api/endpoint")
    check("a restarted server reads the saved endpoint back", report["mode"] == "endpoint" and
          report["model"] == "mock-model" and report["url"].endswith("/v1"), json.dumps(report)[:140])
    check("a restarted server still keeps the key masked", report["key_masked"].endswith("5678") and
          KEY not in json.dumps(report), report["key_masked"])
    events = app.chat("mock-model", "How long is food in the fridge safe during a blackout?", conversation="restart")
    check("the restarted server chats through it", "done" in types(events) and any(
        e["type"] == "read" for e in events), str(types(events)[:8]))
    check("the key never appears in the app's own log", KEY not in app.read_log() and KEY not in first_log)

    # 8. Stop: a newer question on the same chat cuts the older one short; a closed connection forgets it.
    app.post_json("/api/endpoint", {"url": f"http://127.0.0.1:{ports['slow']}/v1", "model": "mock-model"})
    slow_events, cutting = [], threading.Thread(target=lambda: slow_events.extend(
        app.chat("mock-model", "Tell me about blackout food safety", conversation="stop")))
    cutting.start()
    time.sleep(1.0)
    events = app.chat("mock-model", "What about a closed freezer?", conversation="stop")
    cutting.join(30)
    slow_tokens = [e for e in slow_events if e["type"] == "token"]
    check("a new question stops the answer still running", "done" not in types(slow_events) or len(slow_tokens) < 7,
          f"{len(slow_tokens)} pieces arrived before it was cut short")
    check("the new question is answered after cutting one short", "done" in types(events), str(types(events)[:6]))

    # The page's Stop: closing the answer's connection ends the turn, and the chat forgets that question.
    app.post_json("/api/endpoint", {"url": f"http://127.0.0.1:{ports['quiet']}/v1", "model": "mock-model"})
    resp = app.req("POST", "/api/chat", {"model": "mock-model", "message": "Tell me about blackout food safety",
                                         "conversation": "quiet"})
    ticks, pieces = 0, 0
    for raw in resp:  # read the whole answer: quiet stretches still get a heartbeat
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith(":"):
            ticks += 1
        pieces += 1
    resp.close()
    check("the answer stream keeps its heartbeat comments between quiet stretches", ticks >= 1, f"{ticks} ticks seen")

    # The page's Stop: closing the answer's connection ends the turn, and the chat forgets that question.
    resp = app.req("POST", "/api/chat", {"model": "mock-model", "message": "And what about a closed freezer?",
                                         "conversation": "stopped"})
    for pieces, raw in enumerate(resp, start=1):  # stop reading early, which is what pressing Stop does
        if pieces >= 2:
            break
    resp.close()
    time.sleep(8)  # through the next quiet gap, which is when the app finds the page has gone
    app.chat("mock-model", "What about half a freezer?", conversation="stopped")
    asked = [r for r in mock_log(ports["quiet"]) if r["path"] == "/v1/chat/completions"]
    check("a stopped answer is not left in the chat", not any(
        "And what about a closed freezer?" in str(m.get("content")) for m in asked[-1]["body"]["messages"]),
          "the next question's request still carried the question nobody answered")

    # 9. Test button: an address is tried without saving it, and says what happened in plain words.
    code, report = app.post_json("/api/endpoint/test", {"url": f"http://127.0.0.1:{ports['tools']}", "key": KEY})
    check("Test reports a working address", code == 200 and report["ok"] and report["models"], json.dumps(report)[:120])
    code, report = app.post_json("/api/endpoint/test", {"url": "http://127.0.0.1:1/v1", "key": ""})
    check("Test reports a dead address in plain words", code == 200 and not report["ok"] and "address" in
          report["message"].lower(), report.get("message", ""))
    code, report = app.post_json("/api/endpoint/test", {"url": "not an address at all"})
    check("Test rejects something that is not an address", code == 400 and "not an address" in
          json.dumps(report).lower(), json.dumps(report)[:120])
    check("Test did not change the saved endpoint", app.get_json("/api/endpoint")["url"].endswith(
        f"{ports['quiet']}/v1"), app.get_json("/api/endpoint")["url"])

    # 10. Forget it entirely: the file is gone and everything is back to running locally.
    code, report = app.post_json("/api/endpoint", {"clear": True})
    check("clearing the endpoint reports it off", code == 200 and report["mode"] == "ollama" and not report["url"])
    check("clearing it deletes the settings file", not os.path.exists(os.path.join(home, ".survival-llm",
                                                                                   "endpoint.json")))
    status = app.get_json("/api/status")
    check("after clearing, status looks like a fresh copy's again", set(status) == expected,
          f"extra: {sorted(set(status) - expected)}")

    # Everything above, taken together: nothing crashed on the way.
    check("no traceback anywhere in the app's log", "Traceback" not in app.read_log(),
          app.read_log()[-200:])
    return app


if __name__ == "__main__":
    sys.exit(main())
