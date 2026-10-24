// Survival LLM's Mac launcher: runs the bundled Python inside this process, so macOS treats it as the app
// (its own Dock icon, name and menu) instead of as "Python". build.sh compiles it into Contents/MacOS.
#include <Python.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int main(int argc, char **argv) {
    char exe[PATH_MAX], real[PATH_MAX], res[PATH_MAX], home[PATH_MAX], script[PATH_MAX], log[PATH_MAX];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) != 0 || !realpath(exe, real)) return 1;
    *strrchr(real, '/') = '\0';                                  // .../Survival LLM.app/Contents/MacOS
    snprintf(res, sizeof res, "%s/../Resources", real);
    snprintf(home, sizeof home, "%s/python", res);
    snprintf(script, sizeof script, "%s/app/app/server.py", res);
    const char *user = getenv("HOME");
    if (user) {  // keep a log; there is no terminal to print to
        snprintf(log, sizeof log, "%s/Library/Logs", user);
        mkdir(log, 0755);
        snprintf(log, sizeof log, "%s/Library/Logs/Survival LLM.log", user);
        freopen(log, "a", stdout);
        freopen(log, "a", stderr);
        setvbuf(stdout, NULL, _IOLBF, 0);
    }
    setenv("PYTHONHOME", home, 1);
    setenv("PYTHONNOUSERSITE", "1", 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);  // the app folder may be read-only
    char *args[] = {argv[0], "-s", script, NULL};
    return Py_BytesMain(3, args);
}
