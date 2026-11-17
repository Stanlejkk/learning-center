"""Start the Learning Center: web app (127.0.0.1:8765) + JupyterLab for labs (127.0.0.1:8888).

Both servers bind to localhost only. Ctrl+C stops both.
"""
from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
APP_PORT = int(os.environ.get("LC_PORT", "8765"))
JUPYTER_PORT = int(os.environ.get("LC_JUPYTER_PORT", "8888"))


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    DATA.mkdir(exist_ok=True)
    tok_file = DATA / "jupyter_token.txt"
    if not tok_file.exists():
        tok_file.write_text(secrets.token_hex(16))
    token = tok_file.read_text().strip()
    env = dict(os.environ, LC_JUPYTER_PORT=str(JUPYTER_PORT), LC_API=f"http://127.0.0.1:{APP_PORT}")

    procs = []
    if port_open(JUPYTER_PORT):
        print(f"JupyterLab already running on port {JUPYTER_PORT}")
    else:
        procs.append(subprocess.Popen([
            sys.executable, "-m", "jupyterlab", "--no-browser", f"--port={JUPYTER_PORT}", "--ServerApp.ip=127.0.0.1",
            "--ServerApp.port_retries=0", f"--ServerApp.root_dir={ROOT / 'labs'}", f"--IdentityProvider.token={token}",
            "--LabApp.news_url=", "--LabApp.check_for_updates_class=jupyterlab.NeverCheckForUpdate",
        ], cwd=ROOT / "labs", env=env))
    if port_open(APP_PORT):
        print(f"Learning Center already running on port {APP_PORT}")
    else:
        procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                                       "--port", str(APP_PORT)], cwd=ROOT, env=env))
    for _ in range(60):
        if port_open(APP_PORT):
            break
        time.sleep(0.25)
    url = f"http://127.0.0.1:{APP_PORT}/"
    print(f"\n  Learning Center:  {url}\n  JupyterLab:       http://127.0.0.1:{JUPYTER_PORT}/lab?token={token}\n\n  Ctrl+C to stop.\n")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        while any(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
