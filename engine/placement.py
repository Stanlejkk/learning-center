"""Adaptive placement test (a simple computerised-adaptive walk over the skill graph).

For every track we binary-search over the track's skills (ordered easy -> hard):
  * start in the middle,
  * correct  -> the skill and its in-track ancestors are probably known; move up,
  * wrong    -> the skill and its in-track descendants are probably unknown; move down,
  * stop after the search interval closes or MAX_PER_TRACK items.
Finally every un-probed skill gets a prior based on which side of the boundary it lies.

State is a plain dict (stored as JSON in sessions.data) so the test can be resumed.
"""
from __future__ import annotations

import random

from engine import content

MAX_PER_TRACK = 6
P_KNOWN = 0.90       # directly answered correctly
P_ANCESTOR = 0.85    # implied known
P_BELOW = 0.65       # below the boundary but not directly implied
P_UNKNOWN = 0.12     # answered wrong / above boundary
P_SKIPPED = 0.08


def new_state(track_ids: list[str] | None = None, seed: int | None = None) -> dict:
    tracks = [t for t in content.load_tracks() if not track_ids or t.id in track_ids]
    return {
        "seed": seed if seed is not None else random.randrange(1_000_000),
        "track_order": [t.id for t in tracks],
        "current": 0,
        "tracks": {t.id: {"lo": 0, "hi": len(t.skills) - 1, "asked": [], "done": False,
                          "skipped": False} for t in tracks},
        "finished": False,
    }


def _track(state: dict) -> content.Track | None:
    while state["current"] < len(state["track_order"]):
        tid = state["track_order"][state["current"]]
        ts = state["tracks"][tid]
        if not ts["done"]:
            return next(t for t in content.load_tracks() if t.id == tid)
        state["current"] += 1
    state["finished"] = True
    return None


def next_item(state: dict) -> dict | None:
    """Return {'skill', 'level', 'seed', 'track'} for the next question, or None when finished."""
    t = _track(state)
    if t is None:
        return None
    ts = state["tracks"][t.id]
    if ts["lo"] > ts["hi"] or len(ts["asked"]) >= MAX_PER_TRACK:
        ts["done"] = True
        return next_item(state)
    mid = (ts["lo"] + ts["hi"]) // 2
    skill = t.skills[mid]
    level = 2 if mid > 0 else 1
    seed = state["seed"] * 31 + len(ts["asked"]) * 7 + mid
    return {"skill": skill.id, "level": level, "seed": seed, "track": t.id, "index": mid}


def record(state: dict, item: dict, correct: bool) -> None:
    ts = state["tracks"][item["track"]]
    ts["asked"].append({"skill": item["skill"], "index": item["index"], "correct": bool(correct)})
    if correct:
        ts["lo"] = item["index"] + 1
    else:
        ts["hi"] = item["index"] - 1


def skip_track(state: dict, track_id: str) -> None:
    ts = state["tracks"][track_id]
    ts["done"] = True
    ts["skipped"] = True


def finish(state: dict) -> None:
    for tid in state["track_order"]:
        state["tracks"][tid]["done"] = True
    state["finished"] = True


def priors(state: dict) -> dict[str, float]:
    """Translate placement results into mastery priors for every skill in the tested tracks."""
    out: dict[str, float] = {}
    for tid in state["track_order"]:
        t = next(tt for tt in content.load_tracks() if tt.id == tid)
        ts = state["tracks"][tid]
        ids = [s.id for s in t.skills]
        if ts["skipped"] and not ts["asked"]:
            for sid in ids:
                out[sid] = P_SKIPPED
            continue
        if not ts["asked"]:
            continue  # never reached: leave defaults
        known_idx = [a["index"] for a in ts["asked"] if a["correct"]]
        wrong_idx = [a["index"] for a in ts["asked"] if not a["correct"]]
        boundary = (max(known_idx) + 1) if known_idx else 0
        implied_known, implied_unknown = set(), set()
        for a in ts["asked"]:
            if a["correct"]:
                implied_known |= content.ancestors(a["skill"]) & set(ids)
            else:
                implied_unknown |= content.descendants(a["skill"]) & set(ids)
        for i, sid in enumerate(ids):
            if i in known_idx:
                p = P_KNOWN
            elif i in wrong_idx:
                p = P_UNKNOWN
            elif sid in implied_unknown:
                p = P_UNKNOWN
            elif sid in implied_known:
                p = P_ANCESTOR
            elif i < boundary:
                p = P_BELOW
            else:
                p = P_UNKNOWN
            out[sid] = p
    return out
