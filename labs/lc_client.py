"""Notebook-side helper: grade a lab against its hidden checks and report to the local app.

Usage in the last cell of a lab notebook:

    from lc_client import grade
    grade("07_micrograd", globals())

Checks live in labs/tests/check_<lab_id>.py as functions named `check_*(ns)`, where `ns` is the
notebook's globals(). A check passes if it returns without raising (use plain `assert`).
Results are POSTed to the local app (http://127.0.0.1:8765) and also printed. Nothing leaves the PC.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
import traceback
import urllib.request
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
API = os.environ.get("LC_API", "http://127.0.0.1:8765")


def _load_checks(lab_id: str):
    path = LAB_DIR / "tests" / f"check_{lab_id}.py"
    spec = importlib.util.spec_from_file_location(f"check_{lab_id}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [(name, getattr(mod, name)) for name in sorted(dir(mod)) if name.startswith("check_")]


def run_checks(lab_id: str, ns: dict) -> dict:
    results = []
    for name, fn in _load_checks(lab_id):
        t0 = time.time()
        try:
            fn(ns)
            results.append({"check": name, "ok": True, "msg": "", "sec": round(time.time() - t0, 2)})
        except AssertionError as exc:
            results.append({"check": name, "ok": False, "msg": str(exc) or "assertion failed",
                            "sec": round(time.time() - t0, 2)})
        except Exception as exc:  # noqa: BLE001
            results.append({"check": name, "ok": False,
                            "msg": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}",
                            "sec": round(time.time() - t0, 2)})
    passed = sum(r["ok"] for r in results)
    return {"lab_id": lab_id, "passed": passed, "total": len(results), "results": results}


def grade(lab_id: str, ns: dict, submit: bool = True) -> dict:
    report = run_checks(lab_id, ns)
    for r in report["results"]:
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"[{mark}] {r['check']}" + (f" — {r['msg']}" if r["msg"] else ""))
    print(f"\n{report['passed']}/{report['total']} checks passed.")
    if submit:
        try:
            req = urllib.request.Request(f"{API}/api/labs/submit", data=json.dumps(report).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                print("Recorded in Learning Center:", json.loads(resp.read()).get("status"))
        except Exception as exc:  # noqa: BLE001
            print(f"(Could not reach the Learning Center app at {API}: {exc}. Is it running?)")
    return report
