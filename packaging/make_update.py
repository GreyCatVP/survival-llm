"""Write docs/update.json, which tells installed copies of the app that a new version is out.

Usage: python3 packaging/make_update.py "What's new, one item" ["another item" ...]
Run after packaging/build.sh, once the zips are uploaded to the GitHub release named after packaging/VERSION
(v<version>), then publish docs/ (GitHub Pages). The app downloads the zip for its system from the release, checks
it against the size and SHA-256 here, and replaces itself with it.
"""
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RELEASES = "https://github.com/anonnerdfrenzy/survival-llm/releases/download"

version = open(os.path.join(HERE, "VERSION")).read().strip()
update = {"version": version, "date": datetime.date.today().isoformat(), "notes": sys.argv[1:]}
packages = [("mac", "Survival-LLM-mac.zip"), ("windows", "Survival-LLM-windows.zip")]
# Linux builds are made on their respective Linux hosts and copied into dist before publishing.
packages += [(f"linux-{arch}", f"Survival-LLM-linux-{arch}.zip") for arch in ("x86_64", "aarch64")
             if os.path.isfile(os.path.join(REPO, "dist", f"Survival-LLM-linux-{arch}.zip"))]
for key, name in packages:
    path = os.path.join(REPO, "dist", name)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    update[key] = {"url": f"{RELEASES}/v{version}/{name}", "bytes": os.path.getsize(path), "sha256": digest.hexdigest()}
with open(os.path.join(REPO, "docs", "update.json"), "w") as f:
    json.dump(update, f, indent=1)
print(json.dumps(update, indent=1))
