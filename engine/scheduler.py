"""Decides what to study next: frontier skills, gap remediation, spaced review, stretch.

Homework mix (default 8 problems):
    remediation first (weak prerequisites of the frontier),
    ~70% frontier skills (prereqs ready, not yet mastered),
    ~20% spaced review (mastered skills whose review is due),
    ~10% stretch (a harder level / the next skill up).
"""
from __future__ import annotations

import random
from datetime import date

from engine import content
from engine.mastery import FRONTIER_READY, MASTERED, WEAK, SkillState


def level_for(p: float) -> int:
    if p < 0.40:
        return 1
    if p < 0.75:
        return 2
    return 3


def frontier(states: dict[str, SkillState], track: str | None = None, limit: int = 4) -> list[str]:
    """Unmastered skills whose prerequisites are all ready, in curriculum order."""
    out = []
    for s in content.all_skills_ordered():
        if track and s.track != track:
            continue
        st = states[s.id]
        if st.p >= MASTERED:
            continue
        if all(states[p].p >= FRONTIER_READY for p in s.prereqs):
            out.append(s.id)
        if len(out) >= limit:
            break
    return out


def weak_prereqs(states: dict[str, SkillState], skill_ids: list[str]) -> list[str]:
    """Prerequisites (transitively) of the given skills that are assessed but weak."""
    weak = []
    for sid in skill_ids:
        for a in sorted(content.ancestors(sid), key=lambda x: content.skills_by_id()[x].order):
            st = states[a]
            if st.assessed and st.p < WEAK and a not in weak:
                weak.append(a)
    return weak


def due_reviews(states: dict[str, SkillState], today: date | None = None) -> list[str]:
    today = today or date.today()
    due = [s for s in states.values() if s.review_due and date.fromisoformat(s.review_due) <= today]
    due.sort(key=lambda s: s.review_due)
    return [s.skill_id for s in due]


def blocked_by(states: dict[str, SkillState], skill_id: str) -> list[str]:
    sk = content.skills_by_id()[skill_id]
    return [p for p in sk.prereqs if states[p].p < FRONTIER_READY]


def plan_homework(states: dict[str, SkillState], n: int = 8, track: str | None = None,
                  seed: int | None = None) -> list[dict]:
    """Return a list of {'skill','level','seed','role'} for a new homework set."""
    rng = random.Random(seed)
    items: list[dict] = []

    seen: dict[str, int] = {}

    def add(skill: str, role: str, level: int | None = None):
        if not content.generators_for(skill):
            return
        k = seen.get(skill, 0)
        seen[skill] = k + 1
        # repeated problems on the same skill climb in difficulty: L, L, L+1, L+1, L+2 ...
        lvl = level or min(3, level_for(states[skill].p) + k // 2)
        items.append({"skill": skill, "level": lvl, "seed": rng.randrange(10**9), "role": role})

    front = frontier(states, track=track, limit=4)
    remediation = weak_prereqs(states, front)[:3]
    for sid in remediation:
        add(sid, "remediation", level=1)
        if len(items) < n:
            add(sid, "remediation", level=2)

    n_review = max(1, round(n * 0.2))
    n_stretch = 1 if n >= 6 else 0
    n_front = max(0, n - len(items) - n_review - n_stretch)

    if front:
        k = 0
        while n_front > 0:
            add(front[k % min(len(front), 2 if n_front <= 4 else len(front))], "frontier")
            n_front -= 1
            k += 1
    reviews = due_reviews(states)
    if not reviews:  # nothing due yet: review the strongest recently-learned skills lightly
        reviews = [s.skill_id for s in sorted(states.values(), key=lambda s: -s.p)
                   if s.p >= FRONTIER_READY and s.skill_id not in front][:n_review]
    for sid in reviews[:n_review]:
        add(sid, "review", level=max(2, level_for(states[sid].p)))
    if n_stretch and front:
        add(front[0], "stretch", level=3)
    while len(items) < n and front:
        add(rng.choice(front), "frontier")
    if not items:  # nothing assessed at all: start at the very beginning
        first = content.all_skills_ordered()[0].id
        for _ in range(n):
            add(first, "frontier", level=1)
    return items[: n + 2 * len(remediation)]


def plan_remediation(states: dict[str, SkillState], skills: list[str], seed: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for sid in skills:
        for lvl in (1, 1, 2):
            if content.generators_for(sid):
                items.append({"skill": sid, "level": lvl, "seed": rng.randrange(10**9), "role": "remediation"})
    return items


def plan_checkpoint(states: dict[str, SkillState], track: str, seed: int | None = None,
                    max_items: int = 10) -> list[dict]:
    """A timed unit test over the skills of a track the learner claims to know (p >= 0.8)."""
    rng = random.Random(seed)
    t = next(tt for tt in content.load_tracks() if tt.id == track)
    cands = [s.id for s in t.skills if states[s.id].p >= FRONTIER_READY and content.generators_for(s.id)]
    if not cands:
        cands = [s.id for s in t.skills if content.generators_for(s.id)][:3]
    rng.shuffle(cands)
    cands = sorted(cands[:max_items], key=lambda x: content.skills_by_id()[x].order)
    return [{"skill": sid, "level": 3 if states[sid].p >= 0.9 else 2, "seed": rng.randrange(10**9),
             "role": "checkpoint"} for sid in cands]


def next_actions(states: dict[str, SkillState]) -> list[dict]:
    """Human-readable suggestions for the dashboard."""
    acts = []
    if not any(s.assessed for s in states.values()):
        return [{"kind": "placement", "text": "Take the placement test so the app knows where you stand."}]
    front = frontier(states, limit=3)
    weak = weak_prereqs(states, front)
    if weak:
        names = ", ".join(content.skills_by_id()[w].name for w in weak[:3])
        acts.append({"kind": "remediation", "text": f"Fix gaps first: {names}", "skills": weak[:3]})
    due = due_reviews(states)
    if due:
        acts.append({"kind": "review", "text": f"{len(due)} skill(s) due for spaced review"})
    if front:
        names = ", ".join(content.skills_by_id()[f].name for f in front)
        acts.append({"kind": "homework", "text": f"Homework on: {names}"})
    for t in content.load_tracks():
        ids = [s.id for s in t.skills]
        ready = [i for i in ids if states[i].p >= FRONTIER_READY and not states[i].certified]
        if len(ready) >= max(3, len(ids) // 2):
            acts.append({"kind": "checkpoint", "track": t.id,
                         "text": f"Ready for a checkpoint test in {t.name} ({len(ready)} skills)"})
            break
    return acts
