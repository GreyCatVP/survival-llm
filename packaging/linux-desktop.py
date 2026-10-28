"""Register this portable build in the current user's application menu, without root."""
import os
from pathlib import Path


def desktop_entry(launcher):
    # Desktop Exec has its own quoting rules; percent signs introduce field codes even inside quotes.
    executable = str(launcher).replace("%", "%%")
    for char in ("\\", '"', "`", "$"):
        executable = executable.replace(char, "\\" + char)
    # Desktop string values decode backslashes before Exec is parsed.
    executable = executable.replace("\\", "\\\\")
    if "\n" in executable or "\r" in executable:
        raise ValueError("Move the app to a folder without newlines in its name before registering it.")
    return ("[Desktop Entry]\nType=Application\nName=Survival LLM\n"
            "Comment=Offline AI assistant and survival library\n"
            f'Exec="{executable}"\nTerminal=false\nIcon=applications-education\n'
            "Categories=Education;Utility;\nStartupNotify=true\n")


if __name__ == "__main__":
    launcher = Path(__file__).resolve().parent / "survival-llm"
    applications = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "applications"
    applications.mkdir(parents=True, exist_ok=True)
    target = applications / "survival-llm.desktop"
    target.write_text(desktop_entry(launcher), encoding="utf-8")
    print(f"Added Survival LLM to your application menu ({target}).")
