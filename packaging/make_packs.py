"""Zip built library folders into release parts and write the manifest the app downloads from.

Usage: python3 packaging/make_packs.py --base-url URL [--part-mb 1900] ID=DIR [ID=DIR ...]
  e.g. --base-url https://github.com/anonnerdfrenzy/survival-llm/releases/download/v0.1.0 \
       lite=dist/library-lite full=dist/library-full
Each DIR holds index.sqlite and text/ (built with LIBRARY_OUT=DIR library/build_library.py --lite or --public).
Writes dist/packs/library-<id>.zip.NNN parts and dist/packs/library-packs.json; publish the parts in that release
and the JSON at docs/library-packs.json (the app reads it from the project's web page).
Each zip carries SOURCES.md, since several sources' licenses ask for the license to travel with every copy.
GitHub release files must be under 2 GB, hence the parts; the app joins them after downloading.
"""
import hashlib
import json
import os
import sqlite3
import sys
import zipfile

LABELS = {
    "lite": "Essentials: medical, emergency, survival, how-to and repair",
    "full": "Everything: Essentials plus about 930,000 of Wikipedia's most-read articles and iFixit",
}

args = sys.argv[1:]
base_url = args[args.index("--base-url") + 1].rstrip("/")
part_bytes = int(args[args.index("--part-mb") + 1]) * 1_000_000 if "--part-mb" in args else 1_900_000_000
packs = [a.split("=", 1) for a in args if "=" in a and not a.startswith("--")]
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dist", "packs")
os.makedirs(out_dir, exist_ok=True)

manifest = {"packs": []}
for pack_id, folder in packs:
    db = sqlite3.connect(f"file:{os.path.join(folder, 'index.sqlite')}?mode=ro", uri=True)
    articles, sources = db.execute("SELECT COALESCE(SUM(docs), 0), COUNT(*) FROM sources WHERE done=1").fetchone()
    db.close()
    zip_path = os.path.join(out_dir, f"library-{pack_id}.zip")
    print(f"zipping {pack_id}: {articles:,} articles from {sources} sources", flush=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        z.write(os.path.join(os.path.dirname(out_dir), "..", "SOURCES.md"), "SOURCES.md")
        z.write(os.path.join(folder, "index.sqlite"), "index.sqlite")
        for root, _, files in os.walk(os.path.join(folder, "text")):
            for name in sorted(files):
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, folder))
    parts = []
    with open(zip_path, "rb") as f:
        n = 0
        while chunk := f.read(part_bytes):
            name = f"library-{pack_id}.zip.{n:03d}"
            with open(os.path.join(out_dir, name), "wb") as p:
                p.write(chunk)
            parts.append({"url": f"{base_url}/{name}", "bytes": len(chunk),
                          "sha256": hashlib.sha256(chunk).hexdigest()})
            n += 1
    os.remove(zip_path)
    gb = sum(p["bytes"] for p in parts) / 1e9
    label = LABELS.get(pack_id, pack_id)
    manifest["packs"].append({"id": pack_id, "label": f"{label} ({gb:.1f} GB, {articles:,} articles)",
                              "gb": round(gb, 1), "articles": articles, "parts": parts})
    print(f"  {len(parts)} parts, {gb:.2f} GB", flush=True)

with open(os.path.join(out_dir, "library-packs.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("wrote", os.path.join(out_dir, "library-packs.json"))
