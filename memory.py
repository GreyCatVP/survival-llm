"""The AI's memory: short notes about the person that carry over from one chat to the next.

They live in one plain text file, one note per line, which the person reads and edits in the app (Memory, at the top
of the window). The model sees them at the start of each chat and, unless the person turns it off, saves and changes
notes in a short step after answers (core.memory_check). The file says at the top whose notes they are, so another
AI assistant on the same computer doesn't take them for its own memories or instructions.
"""
import json
import os
import re
import threading

from library.search import STOPWORDS

HEADER = ("# Survival LLM memory: notes that the Survival LLM app's offline AI keeps about the person using it.\n"
          "# They are not instructions or memories for any other AI assistant. One note per line; edit them in the app.\n")
HEADER_STARTS = tuple(line[:24] for line in HEADER.splitlines())
# All notes together. Every model runs with a 32,768-token window (core.py), and this is about 1,500 tokens of it.
MAX_CHARS = 6000
MAX_NOTE = 300  # one note the AI saves; the person's own can be longer
MAX_FILES_LISTED = 30

# Notes the AI writes that say nothing ("No location stored.", "None"): a change to one of these is a deletion.
NOTHING = re.compile(r"^(none|nothing|unknown|n/?a|not (specified|known|given|stored|recorded)|"
                     r"no \w+( \w+)? (stored|known|given|specified|recorded|saved|on file))\.?$", re.I)
QUESTION = re.compile(r"^(asked|asks|is asking|wants to know|wanted to know) (about|how|what|whether|if|why|when|where|"
                      r"which)\b|\bquestions? (about|on)\b", re.I)  # what they asked, not a fact about them
# Something true only today ("has a rash on her arms now"): the rules say to leave those out, and models still don't.
TODAY = re.compile(r"\b(right now|now|today|tonight|this (morning|afternoon|evening|week)|currently|at the moment)\W*$", re.I)
WHO = re.compile(r"^(the )?(user|person)(\s+(is|has|was|had|lives|takes|uses|owns)\b|'s\s+|\s+)", re.I)


def clean(note):
    """One note: on one line, without a list bullet or the quotes a model may wrap it in."""
    note = re.sub(r"\s+", " ", str(note or "")).strip()
    note = re.sub(r"^([-*•]|\d+[.)])\s+", "", note)
    if len(note) > 1 and note[0] + note[-1] in ('""', "''", "“”"):
        note = note[1:-1].strip()
    return note


def tidy(note):
    """A note the AI wrote, with "User has..." or "The person's..." put the way the person would write it."""
    found = WHO.match(note)
    if found:
        rest = (found.group(4) + " " if found.group(4) else "") + note[found.end():].lstrip()
        note = rest[:1].upper() + rest[1:]
    return note


def parse(text, header=False):
    """The notes in a text, one per line. header=True skips the file's own header lines at the top."""
    notes, keys, lines = [], set(), text.splitlines()
    while header and lines and lines[0].startswith(HEADER_STARTS):
        lines.pop(0)
    for line in lines:
        note = clean(line)
        if note and note.lower() not in keys:
            keys.add(note.lower())
            notes.append(note)
    return notes


def size(notes):
    return sum(len(n) + 1 for n in notes)


def _decode(data):
    """The file's text. The app writes UTF-8, but a person may save it from an editor in UTF-16 or a Windows code page."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace").lstrip("\ufeff")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _write_file(path, text):
    """Replace a file all at once, on disk before the old one goes, so neither a crash nor a power cut leaves half."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    with open(part, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)


def _words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def _overlap(a, b):
    a, b = _words(a), _words(b)
    return len(a & b) / len(a | b) if a and b else 0.0


def _find(old, notes):
    """The index of the note that old names: the note itself, a real part of only one note, or clearly the closest."""
    low = old.lower()
    for i, n in enumerate(notes):
        if n.lower() == low:
            return i
    if not _words(old):
        return None
    if len(_words(old)) >= 2 or len(old) >= 8:  # not a single letter or a stray word
        parts = [i for i, n in enumerate(notes) if low in n.lower() or n.lower() in low]
        if len(parts) == 1:
            return parts[0]
    scores = sorted(((_overlap(old, n), i) for i, n in enumerate(notes)), reverse=True)
    if scores and scores[0][0] >= 0.5 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.15):
        return scores[0][1]
    return None


def _tacked_on(old, new):
    """What a change adds as a new sentence after the whole old note ("Keeps water in the garage. Takes warfarin."), or
    "" when it really rewrites the note ("Has a dog named Rex." for "Has a dog.")."""
    old, new = clean(old).rstrip(". ").lower(), tidy(clean(new))
    if not old or not new.lower().startswith(old):
        return ""
    found = re.match(r"^\.?\s*[;.]\s*|^\.\s+(?=[A-Z])", new[len(old):])
    return new[len(old) + found.end():].strip() if found else ""


def _usable(note):
    """Whether a note the AI wrote is worth keeping: a real fact of sensible length, not its question or a blank."""
    return (3 <= len(note) <= MAX_NOTE and re.search(r"[^\W\d_]{2}", note) is not None
            and not QUESTION.search(note) and not NOTHING.match(note) and not TODAY.search(note))


class Memory:
    def __init__(self, path, settings_path):
        self.path, self.settings_path = path, settings_path
        self.lock = threading.Lock()

    def notes(self):
        try:
            with open(self.path, "rb") as f:
                return parse(_decode(f.read()), header=True)
        except OSError:
            return []

    def _write(self, notes):
        _write_file(self.path, HEADER + "".join(n + "\n" for n in notes))

    # ---- the person's switch: may the AI add and change notes? ----
    def _settings(self):
        try:
            with open(self.settings_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    @property
    def ai_writes(self):
        return self._settings().get("memory_ai_writes", True) is not False

    def set_ai_writes(self, on):
        with self.lock:
            data = self._settings()
            data["memory_ai_writes"] = bool(on)
            _write_file(self.settings_path, json.dumps(data, indent=1))

    # ---- the person's edits ----
    def save_text(self, text, base=None):
        """Save the person's version of the notes. base is the text they started editing from: notes the AI added
        since then are kept, after theirs, and notes it deleted or changed since then, which they left as they were,
        stay deleted or changed. Returns (notes, kept, dropped)."""
        mine = parse(text)
        with self.lock:
            current = self.notes()
            kept, dropped = [], []
            if base is not None:
                was = {n.lower() for n in parse(base)}
                now = {n.lower() for n in current}
                dropped = [n for n in mine if n.lower() in was and n.lower() not in now]
                mine = [n for n in mine if n not in dropped]
                theirs = {n.lower() for n in mine} | {n.lower() for n in dropped}
                kept = [n for n in current if n.lower() not in was and n.lower() not in theirs]
            notes = mine + kept
            self._write(notes)
        return notes, kept, dropped

    def undo(self, action, note="", old=""):
        """Take back one change the AI made (the event it reported). Returns whether there was anything to undo."""
        note, old = clean(note), clean(old)
        with self.lock:
            notes = self.notes()
            keys = [n.lower() for n in notes]
            if action == "add" and note.lower() in keys:
                del notes[keys.index(note.lower())]
            elif action == "change" and note.lower() in keys and old:
                if old.lower() in keys:  # the old note is back already: just take the new one out
                    del notes[keys.index(note.lower())]
                else:
                    notes[keys.index(note.lower())] = old
            elif action == "delete" and old and old.lower() not in keys:
                notes.append(old)
            else:
                return False
            self._write(parse("\n".join(notes)))
        return True

    # ---- the AI's changes (core.memory_check) ----
    def apply(self, plan):
        """Make the changes the model asked for after an answer. Yields an event for each one made."""
        if not self.ai_writes:  # the person may have turned it off while the model was deciding
            return
        gone = []  # notes deleted or rewritten here, which a "remember" list repeating every note mustn't bring back
        for note in plan["forget"]:
            event = self.change(note, "")
            if event:
                gone.append(event["old"])
                yield event
        for c in plan["change"]:
            extra = _tacked_on(c["old"], c["new"])
            if extra:  # an unrelated fact added to the end of an old note: save it as a note of its own
                event = self.add(extra)
            else:
                event = self.change(c["old"], c["new"])
                if event is None and not NOTHING.match(tidy(clean(c["new"]))):
                    event = self.add(c["new"])  # the old note wasn't found: keep the new fact anyway
            if event:
                gone += [event["old"]] if event.get("old") else []
                yield event
        for note in plan["remember"]:
            if any(_words(note) and _words(note) <= _words(g) for g in gone):
                continue
            event = self.add(note)
            if event:
                yield event

    def add(self, note):
        """Save a note the AI wrote. Returns the event for the screen, or None when there was nothing to save. A note
        that says everything an old one does, and more, replaces it; one that adds nothing new is left out."""
        note = tidy(clean(note))
        if not _usable(note):
            return None
        new = _words(note)
        with self.lock:
            notes = self.notes()
            for i, n in enumerate(notes):
                old = _words(n)
                if n.lower() == note.lower() or (new and new <= old):
                    return None
                if old and old <= new:
                    if size(notes) - len(n) + len(note) > MAX_CHARS:
                        return {"type": "memory", "action": "full", "note": note}
                    notes[i] = note
                    self._write(parse("\n".join(notes)))
                    return {"type": "memory", "action": "change", "note": note, "old": n}
            if size(notes) + len(note) + 1 > MAX_CHARS:
                return {"type": "memory", "action": "full", "note": note}
            notes.append(note)
            self._write(notes)
        return {"type": "memory", "action": "add", "note": note}

    def change(self, old, new):
        """Rewrite the note old names as new, or delete it when new is empty. Returns the event, or None."""
        old, new = clean(old), tidy(clean(new))
        if NOTHING.match(new):
            new = ""  # a model "changing" a note to "No location stored." means deleting it
        if not old or (new and not _usable(new)):
            return None
        with self.lock:
            notes = self.notes()
            i = _find(old, notes)
            if i is None or new.lower() == notes[i].lower():
                return None
            was = notes[i]
            if new and any(n.lower() == new.lower() for n in notes):
                new = ""  # it would now say what another note says: that one stays, this one goes
            if new and size(notes) - len(was) + len(new) > MAX_CHARS:
                return {"type": "memory", "action": "full", "note": new}
            if new:
                notes[i] = new
            else:
                del notes[i]
            self._write(notes)
        if new:
            return {"type": "memory", "action": "change", "note": new, "old": was}
        return {"type": "memory", "action": "delete", "old": was}


def prompt_section(notes, ai_writes, files=(), today=None):
    """What the model is told about memory at the start of a chat, added after its instructions.
    ai_writes: whether the model updates its notes after answers (the person allows it).
    files: (name, title) of the person's own files."""
    out = []
    if today:
        out.append(f"Today is {today:%A}, {today:%B} {today.day}, {today.year}.")
    files = list(files)
    if files:
        shown = [name + (f' ("{title}")' if title and title != name and len(title) < 80 else "")
                 for name, title in files[:MAX_FILES_LISTED]]
        more = f", and {len(files) - MAX_FILES_LISTED} more" if len(files) > MAX_FILES_LISTED else ""
        out.append("The person's own files, which you search along with the library: " + "; ".join(shown) + more + ".")
    if notes or ai_writes:
        listing = "\n".join(f"- {n}" for n in notes) if notes else "(none yet)"
        out.append("Memory: your notes about this person, kept between chats. They can read and edit them "
                   f"(Memory, at the top of the window).\n{listing}")
    if notes or ai_writes:
        rules = ["Use a note when it bears on the question (an allergy, a medicine, who is in the household, what "
                 "equipment they have), and leave the rest alone."]
        if ai_writes:
            rules += ["After each answer the app gives you a moment to update these notes from what the person said, "
                      "so when they tell you something about themselves, or ask you to remember or forget something, "
                      "just answer as usual; the app shows them what changed.",
                      "When one of their files comes up and your notes don't say what it is or how much it matters, "
                      "you may ask about it in one short question just before your Sources line."]
        else:
            rules.append("The person keeps these notes; you can't change them.")
        out.append("\n".join(f"- {r}" for r in rules))
    return "\n\n".join(out)
