"""Linux packaging/runtime regressions. No model, library, display or downloads required."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from app import server


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux runtime checks")
class LinuxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="survival linux '$ ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def test_settings_locations(self):
        env = {**os.environ, "HOME": str(self.base), "XDG_CONFIG_HOME": str(self.base / "config")}
        env.pop("SURVIVAL_LLM_APP_DIR", None)
        def settings():
            result = subprocess.check_output([sys.executable, "-c", "from app.server import APP_DIR; print(APP_DIR)"],
                                             cwd=REPO, env=env, text=True)
            return result.strip()
        self.assertEqual(settings(), str(self.base / "config/survival-llm"))
        (self.base / ".survival-llm").mkdir()
        self.assertEqual(settings(), str(self.base / ".survival-llm"))
        env["SURVIVAL_LLM_APP_DIR"] = str(self.base / "override")
        self.assertEqual(settings(), env["SURVIVAL_LLM_APP_DIR"])

    def test_platform_and_models(self):
        for arch, key in (("x86_64", "linux-x86_64"), ("arm64", "linux-aarch64"),
                          ("aarch64", "linux-aarch64"), ("riscv64", None)):
            with patch.object(server.platform, "machine", return_value=arch):
                self.assertEqual(server.platform_key(), key)
        with patch.object(server, "ram_gb", return_value=32), patch.object(server, "disk_free_gb", return_value=100):
            rows = server.machine_catalog([])
        self.assertEqual(next(r["model"] for r in rows if r["recommended"]), "nemotron-3.5-lightning:30b-a3b")
        data = {"version": "1.2.3", "linux-aarch64": {"url": "https://example.com/app.zip", "bytes": 1,
                                                    "sha256": "a" * 64}}
        self.assertTrue(server.valid_update(data))
        data["linux-aarch64"]["sha256"] = "invalid"
        self.assertFalse(server.valid_update(data))

    def test_trash_never_falls_back_to_delete(self):
        path = self.base / "important.txt"
        path.write_text("keep me")
        with patch.object(server.shutil, "which", return_value=None):
            with self.assertRaises(OSError):
                server.to_trash(str(path))
        self.assertEqual(path.read_text(), "keep me")
        with patch.object(server.shutil, "which", return_value="/usr/bin/gio"), \
                patch.object(server.subprocess, "run") as run:
            server.to_trash(str(path))
            self.assertEqual(run.call_args.args[0], ["gio", "trash", "--", str(path)])
        with patch.object(server.shutil, "which", side_effect=lambda name: "/bin/trash-put" if name == "trash-put" else None), \
                patch.object(server.subprocess, "run") as run:
            server.to_trash(str(path))
            self.assertEqual(run.call_args.args[0], ["trash-put", "--", str(path)])

    def test_portable_install_detection(self):
        root = self.base / "Survival LLM/app"
        root.mkdir(parents=True)
        with patch.object(server, "ROOT", str(root)):
            self.assertIsNone(server.install_location())
            (root.parent / "survival-llm").touch()
            (root / "version.txt").write_text("1.0")
            self.assertEqual(server.install_location(), str(root.parent))
            with patch.object(server, "DEV", None):
                self.assertIsNone(server.update_blocker())

    def package(self):
        archive = self.base / "new.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for name, content in (("survival-llm", '#!/bin/sh\nprintf new > "$(dirname "$0")/started"\n'),
                                  ("python/bin/python3", "python"), ("app/app/server.py", "# server"),
                                  ("app/version.txt", "1.2.3")):
                info = zipfile.ZipInfo("Survival LLM/" + name)
                info.external_attr = (0o100755 if name in ("survival-llm", "python/bin/python3") else 0o100644) << 16
                z.writestr(info, content)
        return archive

    def test_update_swap_and_rollback(self):
        for rollback in (False, True):
            folder = self.base / ("rollback" if rollback else "app") / "Survival LLM"
            folder.mkdir(parents=True)
            launcher = folder / "survival-llm"
            launcher.write_text('#!/bin/sh\nprintf old > "$(dirname "$0")/started"\n')
            launcher.chmod(0o755)
            failure = self.base / "failed.txt"
            with patch.object(server.os, "getpid", return_value=2147483647), \
                    patch.object(server, "UPDATE_FAILED", str(failure)):
                helper = server.install_linux(str(self.package()), str(folder))
            new = Path(helper["cmd"][6])
            self.assertTrue((new / "python/bin/python3").stat().st_mode & 0o111)
            if rollback:
                shutil.rmtree(new)
            subprocess.run(helper["cmd"], check=True, capture_output=True, timeout=10)
            self.assertEqual((folder / "started").read_text(), "old" if rollback else "new")
            self.assertEqual(failure.exists(), rollback)

    def test_update_rejects_invalid_archives(self):
        archive = self.base / "bad.zip"
        for name in ("../escape", "Survival LLM/unrelated.txt"):
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(name, "bad")
            with self.assertRaises(RuntimeError):
                server.install_linux(str(archive), str(self.base / "current"))
        self.assertFalse((self.base / "escape").exists())
        self.assertEqual(list(self.base.glob(".survival-llm-update-*")), [])

    def test_launcher_and_desktop_registration(self):
        folder = self.base / 'app "quoted" %f'
        (folder / "python/bin").mkdir(parents=True)
        shutil.copyfile(REPO / "packaging/linux-launcher.sh", folder / "survival-llm")
        (folder / "survival-llm").chmod(0o755)
        # A stand-in interpreter records all arguments, including paths and flags with spaces.
        interpreter = folder / "python/bin/python3"
        interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "$PYWEBVIEW_GUI" "$@" > "$CAPTURE"\n')
        interpreter.chmod(0o755)
        capture = self.base / "args"
        env = {**os.environ, "CAPTURE": str(capture), "XDG_STATE_HOME": str(self.base / "state")}
        env.pop("PYWEBVIEW_GUI", None)
        subprocess.run([str(folder / "survival-llm"), "--no-browser", "--port", "12345"], env=env, check=True)
        self.assertEqual(capture.read_text().splitlines(),
                         ["qt", "-s", str(folder / "app/app/server.py"), "--no-browser", "--port", "12345"])
        shutil.copyfile(REPO / "packaging/linux-desktop.py", folder / "install-desktop.py")
        env["XDG_DATA_HOME"] = str(self.base / "data")
        subprocess.run([sys.executable, str(folder / "install-desktop.py")], env=env, check=True, capture_output=True)
        entry = self.base / "data/applications/survival-llm.desktop"
        self.assertIn("%%f", entry.read_text())
        if shutil.which("desktop-file-validate"):
            subprocess.run(["desktop-file-validate", str(entry)], check=True, capture_output=True)

    def test_release_manifest_includes_available_linux_architectures(self):
        (self.base / "packaging").mkdir()
        (self.base / "docs").mkdir()
        (self.base / "dist").mkdir()
        (self.base / "packaging/VERSION").write_text("1.2.3")
        shutil.copyfile(REPO / "packaging/make_update.py", self.base / "packaging/make_update.py")
        for platform in ("mac", "windows", "linux-x86_64"):
            (self.base / f"dist/Survival-LLM-{platform}.zip").write_bytes(b"package")
        subprocess.run([sys.executable, str(self.base / "packaging/make_update.py"), "Linux support"],
                       check=True, capture_output=True)
        data = json.loads((self.base / "docs/update.json").read_text())
        self.assertIn("linux-x86_64", data)
        self.assertNotIn("linux-aarch64", data)
        self.assertTrue(server.valid_update(data))


if __name__ == "__main__":
    unittest.main()
