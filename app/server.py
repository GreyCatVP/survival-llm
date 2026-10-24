"""Survival LLM: a local web app for chatting with a local model that searches the offline library.

Run: python app/server.py   (opens your browser; listens on 127.0.0.1 only; works with Wi-Fi off)
Runs models with Ollama (https://ollama.com): the app's own copy (inside the Mac app, or downloaded on first use
on Windows), or one the person installed. Everything else is the Python standard library.
"""
import atexit
import ctypes
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import HOST, capabilities, chat_events, installed_models, system_prompt, trim  # noqa: E402
from library.search import DB_PATH, LIB, TOPICS, Library  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TOKEN = secrets.token_urlsafe(16)
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png"}

# Models the app offers. mac/win are Ollama names; min_ram is total memory in GB; gb is the download size.
# Picks and blurbs come from evals/retrieval_eval.py (22 blackout and first-aid questions answered with the library).
CATALOG = [
    {"key": "nemotron-super", "name": "Nemotron 3.5 Super", "maker": "NVIDIA", "min_ram": 128,
     "mac": "nemotron-3.5-super:120b-a12b-nvfp4", "win": "nemotron-3.5-super:120b-a12b", "gb": {"mac": 80, "win": 90},
     "blurb": {"mac": "The smartest option: knows the most and is the least likely to make things up. Quick on Apple Silicon.",
               "win": "The smartest option: knows the most and is the least likely to make things up. Needs a powerful PC."}},
    {"key": "qwen-27b", "name": "Qwen3.8 27B", "maker": "Alibaba", "min_ram": 32,
     "mac": "qwen3.8:27b-mlx", "win": "qwen3.8:27b", "gb": {"mac": 18, "win": 18},
     "blurb": {"mac": "Careful and thorough: looks things up for every question. A great fit for 32-64 GB Macs.",
               "win": "Careful and thorough, but slow unless your PC has a graphics card with 24 GB of memory."}},
    {"key": "nemotron-lightning", "name": "Nemotron 3.5 Lightning", "maker": "NVIDIA", "min_ram": 32,
     "mac": "nemotron-3.5-lightning:30b-a3b-nvfp4", "win": "nemotron-3.5-lightning:30b-a3b", "gb": {"mac": 19, "win": 25},
     "blurb": {"mac": "The fastest: answers in a few seconds and saves battery. Sometimes misses details.",
               "win": "Quick even without a graphics card, since it uses only a small part of itself per word. "
                      "Sometimes misses details."}},
    {"key": "granite-8b", "name": "Granite 4.2 8B", "maker": "IBM", "min_ram": 12,
     "mac": "granite4.2:8b", "win": "granite4.2:8b", "gb": {"mac": 6, "win": 6},
     "blurb": {"mac": "For 16 GB computers. Always looks things up and shows its sources, but makes more mistakes "
                      "than the bigger models, so check the source.",
               "win": "For 16 GB computers. Always looks things up and shows its sources, but makes more mistakes "
                      "than the bigger models, so check the source."}},
    {"key": "gemma-e4b", "name": "Gemma 4 E4B", "maker": "Google", "min_ram": 12,
     "mac": "gemma4:latest", "win": "gemma4:latest", "gb": {"mac": 10, "win": 10},
     "blurb": {"mac": "Small and very fast, for 16 GB computers. Often answers from memory instead of the library, "
                      "so check what it says.",
               "win": "Small and very fast, for 16 GB computers. Often answers from memory instead of the library, "
                      "so check what it says."}},
]
# Order = what to recommend first, from the blind-judged tests in evals/ (README "How the models did").
# Granite 4.2 3B was dropped: 5 of its 22 test answers could hurt someone, 2 of them fatally.
# Nemotron Super needs 128 GB: macOS lets the GPU use about three quarters of memory, and the model is 80 GB.
# Windows doesn't get it: its 90 GB build would run mostly on the processor, far too slowly.
PREFERENCE = {"mac": ["nemotron-super", "qwen-27b", "nemotron-lightning", "granite-8b", "gemma-e4b"],
              "win": ["nemotron-lightning", "qwen-27b", "granite-8b", "gemma-e4b"]}
ALIASES = {"granite4.2": "granite-8b"}  # granite4.2:latest is the same 8B model
NOT_CHAT = ("embed", "bge-", "nomic", "minilm")
DISK_MARGIN_GB = 5  # keep this much free after a download

# The library packs on offer are listed in a small JSON file published with each library release
# (packaging/make_packs.py writes it): {"packs": [{"id", "label", "gb", "parts": [{"url", "bytes", "sha256"}]}]}.
# Packs are zips of index.sqlite + text/, split into parts because GitHub caps release files at 2 GB.
# It lives on the project's web page (not "latest release"), so app releases and library releases stay independent.
MANIFEST_URL = os.environ.get("SURVIVAL_LLM_MANIFEST") or \
    "https://anonnerdfrenzy.github.io/survival-llm/library-packs.json"

if sys.platform == "darwin":
    APP_DIR = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "Survival LLM")
elif os.name == "nt":
    APP_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "Survival LLM")
else:
    APP_DIR = os.path.join(os.path.expanduser("~"), ".survival-llm")
STATE = os.path.join(APP_DIR, "server.json")
CUSTOM_PROMPT = os.path.join(APP_DIR, "instructions.txt")  # the person's own edit of the AI's instructions

# The app's own Ollama. The Mac app carries it (Resources/ollama, next to the app code). On Windows it is 1.6 GB
# with its graphics-card libraries, so it is downloaded on first use instead; build.sh writes engine.json with its
# address and checksum. An Ollama that is already running is used as is.
if sys.platform == "darwin":
    ENGINE = os.path.join(os.path.dirname(ROOT), "ollama", "ollama")
elif os.name == "nt":
    ENGINE = os.path.join(os.environ.get("LOCALAPPDATA") or APP_DIR, "Survival LLM", "ollama", "ollama.exe")
else:
    ENGINE = None
ENGINE_INFO = os.path.join(ROOT, "engine.json")
engine_procs = []   # Ollama processes this app started, stopped when it quits

conversations = {}
caps_cache = {}
local = threading.local()
lib_version = [0]   # bumped when a new library is installed, so threads reopen it
downloads = {}      # key -> {key, label, status, completed, total, error}; runs in background threads
dl_lock = threading.Lock()
warming = {}        # model -> "loading" | "error"
manifest = {"packs": None, "failed_at": 0.0}
ollama_started_at = [0.0]
quit_hooks = []     # run when someone presses Quit (closes the app window)
quit_event = threading.Event()


# ---------- helpers ----------

def running_instance():
    """URL of an Survival LLM server this user already has running, if any."""
    try:
        with open(STATE) as f:
            port = json.load(f)["port"]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=2) as resp:
            if json.load(resp).get("app") == "survival-llm":
                return f"http://127.0.0.1:{port}/"
    except (OSError, ValueError, KeyError):
        return None


def library():
    """One SQLite connection per server thread, reopened after a new library is installed."""
    if getattr(local, "version", None) != lib_version[0] or getattr(local, "lib", None) is None:
        local.version = lib_version[0]
        local.lib = Library() if os.path.exists(DB_PATH) else None
    return local.lib


def ram_gb():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return round(status.ullTotalPhys / 2**30)
    return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)


def disk_free_gb():
    return round(shutil.disk_usage(os.path.expanduser("~")).free / 1e9)


def system_key():
    return "mac" if sys.platform == "darwin" else "win"


def catalog_entry(model):
    """The catalog entry for an Ollama model name (or one of the author's aliases), if any."""
    model = model.removesuffix(":latest")
    if model in ALIASES:
        return next(c for c in CATALOG if c["key"] == ALIASES[model])
    return next((c for c in CATALOG if model in (c["mac"].removesuffix(":latest"), c["win"].removesuffix(":latest"))), None)


def machine_catalog(installed):
    """Catalog entries for this computer, marked with what fits its memory and disk, and what is installed."""
    ram, disk, sys_key = ram_gb(), disk_free_gb(), system_key()
    rows = []
    for key in PREFERENCE[sys_key]:
        c = next(c for c in CATALOG if c["key"] == key)
        local_names = [m for m in installed if catalog_entry(m) and catalog_entry(m)["key"] == key]
        gb = c["gb"][sys_key]
        rows.append({"key": key, "name": c["name"], "maker": c["maker"], "blurb": c["blurb"][sys_key],
                     "model": local_names[0] if local_names else c[sys_key], "gb": gb, "min_ram": c["min_ram"],
                     "fits": c["min_ram"] <= ram, "fits_disk": bool(local_names) or gb + DISK_MARGIN_GB <= disk,
                     "installed": bool(local_names), "recommended": False})
    best = next((r for r in rows if r["fits"] and r["fits_disk"]), None)
    if best:
        best["recommended"] = True
    return rows


def friendly_name(model):
    entry = catalog_entry(model) if model else None
    return entry["name"] if entry else model


def model_caps(model):
    if not caps_cache.get(model):  # never cache an empty answer from a stopped Ollama
        caps_cache[model] = capabilities(model)
    return caps_cache[model]


def library_packs():
    """Packs from the published manifest; empty when offline (checked again after a minute)."""
    if manifest["packs"] is None and time.time() - manifest["failed_at"] > 60:
        try:
            with urllib.request.urlopen(MANIFEST_URL, timeout=6) as resp:
                manifest["packs"] = json.load(resp).get("packs", [])
        except (OSError, ValueError):
            manifest["failed_at"] = time.time()
    return manifest["packs"] or []


# ---------- Ollama: find it, start it, explain its errors ----------

def ollama_locations():
    """(app to open, command-line binary) for the app's own Ollama or an installed one, either may be None."""
    if sys.platform == "darwin":
        apps = ["/Applications/Ollama.app", os.path.expanduser("~/Applications/Ollama.app")]
        bins = [ENGINE, "/opt/homebrew/bin/ollama", "/usr/local/bin/ollama", shutil.which("ollama")]
    elif os.name == "nt":
        base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama")
        apps = [os.path.join(base, "ollama app.exe")]
        bins = [ENGINE, os.path.join(base, "ollama.exe"), shutil.which("ollama")]
    else:
        apps, bins = [], [shutil.which("ollama")]
    app = next((a for a in apps if a and os.path.exists(a)), None)
    binary = next((b for b in bins if b and os.path.exists(b)), None)
    return app, binary


def ollama_starting():
    return time.time() - ollama_started_at[0] < 60


def start_ollama():
    """Start the app's own Ollama, or an installed one, in the background (at most every 30 seconds)."""
    if time.time() - ollama_started_at[0] < 30:
        return
    app, binary = ollama_locations()
    if not (app or binary):
        return
    ollama_started_at[0] = time.time()
    quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    hidden = {"creationflags": 0x08000000} if os.name == "nt" else {}  # CREATE_NO_WINDOW
    try:
        if binary == ENGINE:  # ours runs only while the app does
            os.makedirs(APP_DIR, exist_ok=True)
            with open(os.path.join(APP_DIR, "ollama.log"), "wb") as log:
                engine_procs.append(subprocess.Popen([ENGINE, "serve"], stdout=log, stderr=log, **hidden))
        elif sys.platform == "darwin" and app:
            subprocess.Popen(["open", "-g", "-a", app], **quiet)
        elif os.name == "nt" and app:
            subprocess.Popen([app], **quiet, **hidden)
        else:
            subprocess.Popen([binary, "serve"], **quiet, **(hidden or {"start_new_session": True}))
    except OSError:
        pass


def stop_engine():
    """Stop the Ollama this app started, so it doesn't keep running (and holding memory) after the app quits."""
    for proc in engine_procs:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()


def engine_info():
    """{"url", "bytes", "sha256", "version"} for downloading the app's own Ollama, when this build offers one."""
    try:
        with open(ENGINE_INFO) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def loaded_models():
    """Models Ollama currently has in memory (ready to answer without loading)."""
    try:
        with urllib.request.urlopen(HOST + "/api/ps", timeout=3) as resp:
            return [m["name"].removesuffix(":latest") for m in json.load(resp).get("models", [])]
    except (OSError, ValueError):
        return []


def friendly_error(exc):
    """Plain-language explanation of an Ollama or network failure."""
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = json.loads(exc.read().decode("utf-8", "replace")).get("error", "")
        except (ValueError, OSError):
            detail = ""
        low = detail.lower()
        if exc.code == 404 or "not found" in low:
            return "This model isn't installed. Open Model at the top to download it or pick another."
        if "newer version" in low or "requires" in low and "ollama" in low:
            return "This model needs a newer AI engine. Update Ollama (ollama.com/download), then try again."
        if "memory" in low:
            return "There isn't enough free memory for this model. Close other apps, or pick a smaller model under Model."
        return f"The AI engine reported a problem: {detail or exc.reason}."
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, ConnectionRefusedError) or "refused" in str(reason).lower():
        start_ollama()
        return "The AI engine (Ollama) isn't running. Starting it now; try again in a few seconds."
    if "timed out" in str(reason).lower():
        return "The AI engine took too long to answer. Try again, or pick a faster model under Model."
    return f"Something went wrong: {exc}"


# ---------- background jobs: model downloads, library installs, warm-up ----------

def _update(key, **fields):
    with dl_lock:
        downloads[key].update(fields)


def start_job(key, label, worker, *args):
    """Run a download in the background so it survives page reloads. False if it is already running."""
    with dl_lock:
        current = downloads.get(key)
        if current and current["status"] in ("starting", "downloading", "unpacking"):
            return False
        downloads[key] = {"key": key, "label": label, "status": "starting", "completed": 0, "total": 0, "error": None}

    def run():
        try:
            worker(key, *args)
            _update(key, status="done")
        except Exception as exc:  # shown to the user until they retry
            _update(key, status="error", error=friendly_error(exc) if isinstance(exc, OSError) else str(exc))
    threading.Thread(target=run, daemon=True).start()
    return True


def pull_worker(key, model):
    req = urllib.request.Request(HOST + "/api/pull", data=json.dumps({"model": model}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            event = json.loads(line)
            if event.get("error"):
                raise RuntimeError(f"Download failed: {event['error']}")
            if event.get("total"):
                _update(key, status="downloading", completed=event.get("completed", 0), total=event["total"])
    caps_cache.pop(model, None)


def _download(key, url, path, size, done_before=0, total=None):
    """Fetch url into path, resuming a partial file; reports progress to job key."""
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have < size:
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(req, timeout=120) as resp:
            if have and resp.status != 206:  # server ignored the range: start over
                have = 0
            with open(path, "ab" if have else "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    have += len(chunk)
                    _update(key, status="downloading", completed=done_before + have, total=total or size)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def engine_worker(key):
    """Download and unpack the app's own Ollama (Windows), then start it."""
    info = engine_info()
    if not info or not ENGINE:
        raise RuntimeError("This version of the app can't download the AI engine. Install Ollama from ollama.com.")
    need = info["bytes"] * 2.5 / 1e9  # the zip plus its unpacked copy, briefly at the same time
    if need + DISK_MARGIN_GB > disk_free_gb():
        raise RuntimeError(f"Not enough free disk space: the AI engine needs about {need:.0f} GB while installing.")
    folder = os.path.dirname(ENGINE)
    zip_path, partial = folder + ".zip", folder + ".partial"
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    _download(key, info["url"], zip_path, info["bytes"])
    if _sha256(zip_path) != info["sha256"]:
        os.remove(zip_path)
        raise RuntimeError("The AI engine download was damaged in transit. Try again.")
    _update(key, status="unpacking", completed=info["bytes"], total=info["bytes"])
    shutil.rmtree(partial, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(partial)
    shutil.rmtree(folder, ignore_errors=True)
    os.replace(partial, folder)
    os.remove(zip_path)
    ollama_started_at[0] = 0
    start_ollama()


def library_worker(key, pack_id):
    """Download a pack's parts (resumable, checksummed), join them, unzip into the library folder."""
    pack = next((p for p in library_packs() if p["id"] == pack_id), None)
    if not pack:
        raise RuntimeError("That library pack isn't available. Check the internet connection.")
    total = sum(part["bytes"] for part in pack["parts"])
    need = total / 1e9 * 3.5  # the zip plus its unpacked copy (about 2.3 times the zip), briefly at the same time
    if need + DISK_MARGIN_GB > disk_free_gb():
        raise RuntimeError(f"Not enough free disk space: this pack needs about {need:.0f} GB while installing.")
    work, partial = LIB + ".download", LIB + ".partial"
    os.makedirs(work, exist_ok=True)
    done_before = 0
    for n, part in enumerate(pack["parts"]):
        path = os.path.join(work, f"{pack_id}.{n:03d}")
        _download(key, part["url"], path, part["bytes"], done_before, total)
        if _sha256(path) != part["sha256"]:
            os.remove(path)
            raise RuntimeError(f"Part {n + 1} of {len(pack['parts'])} was damaged in transit. Try again to re-download it.")
        done_before += part["bytes"]
    _update(key, status="unpacking", completed=total, total=total)
    zip_path = os.path.join(work, f"{pack_id}.zip")
    with open(zip_path, "wb") as out:  # join the parts, freeing each as we go
        for n in range(len(pack["parts"])):
            path = os.path.join(work, f"{pack_id}.{n:03d}")
            with open(path, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 24)
            os.remove(path)
    shutil.rmtree(partial, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(partial)
    if os.path.islink(LIB):
        os.remove(LIB)
    elif os.path.isdir(LIB):
        shutil.rmtree(LIB)
    os.replace(partial, LIB)
    shutil.rmtree(work, ignore_errors=True)
    lib_version[0] += 1


def warm_model(model):
    """Load a model into memory ahead of the first question, so that question isn't slow."""
    if warming.get(model) == "loading" or model.removesuffix(":latest") in loaded_models():
        return

    def run():
        warming[model] = "loading"
        try:
            req = urllib.request.Request(HOST + "/api/generate", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"model": model, "keep_alive": "15m"}).encode())
            urllib.request.urlopen(req, timeout=600).read()
            warming.pop(model, None)
        except OSError:
            warming[model] = "error"
    threading.Thread(target=run, daemon=True).start()


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _allowed(self, need_token=True):
        # Host check blocks DNS-rebinding pages; the token blocks other local web pages.
        host = self.headers.get("Host", "")
        if host not in (f"127.0.0.1:{PORT}", f"localhost:{PORT}"):
            return False
        if not need_token:
            return True
        query = parse_qs(urlparse(self.path).query)
        return secrets.compare_digest(self.headers.get("X-Token", query.get("t", [""])[0]), TOKEN)

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _stream_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _event(self, event):
        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        self.wfile.flush()

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/ping":
            return self._send(200 if self._allowed(need_token=False) else 403, {"app": "survival-llm"})
        if url.path == "/" or url.path.startswith("/static/"):
            if not self._allowed(need_token=False):
                return self._send(403, {"error": "forbidden"})
            name = "index.html" if url.path == "/" else os.path.basename(url.path)
            path = os.path.join(STATIC, name)
            if not os.path.isfile(path):
                return self._send(404, {"error": "not found"})
            with open(path, "rb") as f:
                data = f.read()
            if name == "index.html":
                # Other websites cannot read this page (no CORS headers), so the token stays private.
                data = data.replace(b"{{TOKEN}}", TOKEN.encode())
            return self._send(200, data, MIME.get(os.path.splitext(name)[1], "application/octet-stream"))
        if not self._allowed():
            return self._send(403, {"error": "forbidden"})
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/api/status":
            return self._send(200, self._status())
        if url.path == "/api/article":
            doc = library().document(int(query.get("id", 0))) if library() else None
            return self._send(200 if doc else 404, doc or {"error": "not found"})
        if url.path == "/api/prompt":
            return self._send(200, {"default": system_prompt(library(), True), "custom": custom_prompt()})
        if url.path == "/api/search":
            if not library():
                return self._send(200, {"results": []})
            return self._send(200, {"results": library().search(query.get("q", ""), topic=query.get("topic"), limit=25)})
        return self._send(404, {"error": "not found"})

    def _status(self):
        models = installed_models()
        if models is None:
            start_ollama()
        app, binary = ollama_locations()
        info = engine_info()
        lib = library()
        docs, sources = lib.stats() if lib else (0, 0)
        chat_models = [m.removesuffix(":latest") for m in models or [] if not any(x in m for x in NOT_CHAT)]
        catalog = machine_catalog(chat_models)
        best = next((c["model"] for c in catalog if c["installed"] and c["fits"]), None)
        with dl_lock:
            jobs = [dict(j) for j in downloads.values()]
        return {
            "ollama": models is not None, "ollama_installed": bool(app or binary), "ollama_starting": ollama_starting(),
            "engine": {"downloadable": bool(info and ENGINE and not (app or binary)),
                       "gb": round(info["bytes"] / 1e9, 1) if info else None},
            "models": [{"model": m, "name": friendly_name(m), "tested": bool(catalog_entry(m))} for m in chat_models],
            "default_model": best or (chat_models or [None])[0],
            "loaded": loaded_models() if models is not None else [],
            "warming": [m for m, s in warming.items() if s == "loading"],
            "catalog": catalog, "downloads": jobs,
            "library": {"articles": docs, "sources": sources} if docs else None,
            "packs": [] if docs else [{k: p[k] for k in ("id", "label", "gb")} for p in library_packs()],
            "library_dir": LIB, "disk_free_gb": disk_free_gb(),
            "machine": {"os": "macOS" if sys.platform == "darwin" else platform.system(), "ram_gb": ram_gb()},
            "topics": TOPICS}

    def do_POST(self):
        if not self._allowed():
            return self._send(403, {"error": "forbidden"})
        url = urlparse(self.path)
        body = self._body()
        model = str(body.get("model") or "")
        if model and not re.fullmatch(r"[\w.:/-]{1,120}", model):
            return self._send(400, {"error": "bad model name"})
        if url.path == "/api/prompt":  # save the person's instructions ("" or reset: back to the default)
            text = "" if body.get("reset") else str(body.get("text") or "").strip()
            os.makedirs(APP_DIR, exist_ok=True)
            if text:
                with open(CUSTOM_PROMPT, "w", encoding="utf-8") as f:
                    f.write(text + "\n")
            elif os.path.exists(CUSTOM_PROMPT):
                os.remove(CUSTOM_PROMPT)
            for conv in conversations.values():  # takes effect in open chats too, from the next question
                if conv.get("messages"):
                    conv["messages"][0]["content"] = prompt_for(model_caps(conv["model"]))
            return self._send(200, {"ok": True, "custom": custom_prompt()})
        if url.path == "/api/new":
            conversations.pop(body.get("conversation"), None)
            return self._send(200, {"conversation": uuid.uuid4().hex})
        if url.path == "/api/chat":
            return self._chat(body)
        if url.path == "/api/pull":
            started = start_job(f"model:{model}", friendly_name(model), pull_worker, model)
            return self._send(200 if started else 409, {"ok": started})
        if url.path == "/api/library":
            pack = str(body.get("pack", ""))
            label = next((p["label"] for p in library_packs() if p["id"] == pack), pack)
            started = start_job(f"library:{pack}", label, library_worker, pack)
            return self._send(200 if started else 409, {"ok": started})
        if url.path == "/api/warm":
            warm_model(model)
            return self._send(200, {"ok": True})
        if url.path == "/api/engine":
            started = start_job("engine", "AI engine (Ollama)", engine_worker)
            return self._send(200 if started else 409, {"ok": started})
        if url.path == "/api/start-ollama":
            ollama_started_at[0] = 0
            start_ollama()
            return self._send(200, {"ok": True})
        if url.path == "/api/quit":
            self._send(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            quit_event.set()
            for hook in quit_hooks:
                threading.Thread(target=hook, daemon=True).start()
            return None
        return self._send(404, {"error": "not found"})

    def _chat(self, body):
        model, question = body.get("model"), (body.get("message") or "").strip()
        if not model or not question:
            return self._send(400, {"error": "model and message are required"})
        conv = conversations.setdefault(body.get("conversation") or uuid.uuid4().hex, {"model": None, "run": 0})
        caps = model_caps(model)
        if conv.get("model") != model:  # new chat, or the model changed mid-chat
            conv.update(model=model, messages=[{"role": "system", "content": prompt_for(caps)}])
        conv["run"] += 1  # a newer question on this chat stops any older one still running
        run = conv["run"]
        messages = conv["messages"]
        keep = len(messages)
        messages.append({"role": "user", "content": question})
        trim(messages)
        self._stream_start()
        events = chat_events(model, library(), messages, deep=bool(body.get("deep")), caps=caps,
                             length=str(body.get("length") or "normal"))
        finished, last_tick = False, 0.0
        try:
            for event in events:
                if conv["run"] != run:
                    break
                if event["type"] == "tick":  # heartbeat: writing it is how we notice a closed or stopped page
                    if time.monotonic() - last_tick > 0.5:
                        last_tick = time.monotonic()
                        self.wfile.write(b": tick\n\n")
                        self.wfile.flush()
                    continue
                self._event(event)
            finished = conv["run"] == run
        except (BrokenPipeError, ConnectionResetError):
            pass  # stopped, or the page was closed
        except Exception as exc:
            try:
                self._event({"type": "error", "message": friendly_error(exc)})
            except OSError:
                pass
        finally:
            events.close()  # closes the request to Ollama, which stops generating
            if not finished and conv["run"] == run:
                del messages[keep:]  # forget the unanswered question so the next one starts clean


def custom_prompt():
    try:
        with open(CUSTOM_PROMPT, encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def prompt_for(caps):
    """The instructions a conversation starts with: the person's edited version, or the default for this model."""
    return custom_prompt() or system_prompt(library(), "tools" in caps)


def open_window(url):
    """Show the app in its own window (the system's built-in web view) until it is closed. False when this
    computer can't (pywebview or the web view missing), so the caller falls back to the browser."""
    try:
        import webview
    except ImportError:
        return False
    try:
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True  # source links go to the browser, not this window
        window = webview.create_window("Survival LLM", url, width=1280, height=860, min_size=(420, 520),
                                       text_select=True)
        quit_hooks.append(window.destroy)
        webview.start(gui="edgechromium" if os.name == "nt" else None, private_mode=False,
                      storage_path=os.path.join(APP_DIR, "window"))
        return True
    except Exception as exc:  # e.g. no WebView2 on an old Windows 10
        print(f"Couldn't open a window ({exc}); using the browser instead.", flush=True)
        return False


def make_server(preferred):
    """Listen on the preferred port, or any free one if something else holds it."""
    try:
        return ThreadingHTTPServer(("127.0.0.1", preferred), Handler)
    except OSError:
        return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


if __name__ == "__main__":
    if sys.stdout is None:  # Windows pythonw (no console): keep the log instead of losing it
        os.makedirs(APP_DIR, exist_ok=True)
        sys.stdout = sys.stderr = open(os.path.join(APP_DIR, "app.log"), "a", buffering=1)
    headless, use_browser = "--no-browser" in sys.argv, "--browser" in sys.argv
    existing = None if "--port" in sys.argv else running_instance()  # an explicit port means "start another"
    if existing:  # opened twice: just bring up the one already running
        print(f"Survival LLM is already running at {existing}")
        if not headless and (use_browser or not open_window(existing)):
            webbrowser.open(existing)
        sys.exit(0)
    atexit.register(stop_engine)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so a plain kill still stops the engine
    if installed_models() is None:
        start_ollama()
    port_arg = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "8765"
    server = make_server(int(port_arg))
    PORT = server.server_address[1]
    url = f"http://127.0.0.1:{PORT}/"
    os.makedirs(APP_DIR, exist_ok=True)
    with open(STATE, "w") as f:
        json.dump({"port": PORT, "pid": os.getpid()}, f)
    print(f"Survival LLM is running at {url}", flush=True)
    try:
        if headless or use_browser:
            if use_browser:
                threading.Timer(0.5, webbrowser.open, args=(url,)).start()
            server.serve_forever()
        else:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            if not open_window(url):  # no window support on this computer: use the browser after all
                webbrowser.open(url)
                quit_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            os.remove(STATE)
        except OSError:
            pass
