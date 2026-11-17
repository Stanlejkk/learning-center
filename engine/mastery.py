"""Mastery model: Bayesian Knowledge Tracing per skill, with partial credit,
prerequisite gap propagation and spaced-review intervals.

score in [0, 1]: a correct final answer is strong evidence (0.8 .. 1.0, the rubric fine-tunes it);
a wrong answer earns a little partial credit for correct working (0 .. 0.3).
If the rubric was not used at all, only the answer counts (1.0 / 0.0).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from engine import content

P_INIT = 0.10        # prior for a never-assessed skill
P_LEARN = 0.12       # chance of learning from one practice attempt
P_SLIP = 0.10        # knows it but gets it wrong
P_GUESS = 0.08       # free-response answers are hard to guess
MASTERED = 0.95
FRONTIER_READY = 0.80  # prereqs must be at least this to unlock a skill
WEAK = 0.60            # below this a prereq triggers remediation
CORRECT_BASE = 0.8       # correct answer, no rubric steps ticked
WRONG_MAX_CREDIT = 0.3   # wrong answer, every rubric step ticked
PREREQ_PENALTY = 0.88  # multiplicative evidence against direct prereqs on a wrong answer


def combined_score(correct: bool, rubric_fraction: float | None) -> float:
    if rubric_fraction is None:
        return 1.0 if correct else 0.0
    f = max(0.0, min(1.0, rubric_fraction))
    return CORRECT_BASE + (1 - CORRECT_BASE) * f if correct else WRONG_MAX_CREDIT * f


def bkt_update(p: float, score: float, learn: bool = True,
               slip: float = P_SLIP, guess: float = P_GUESS, p_learn: float = P_LEARN) -> float:
    """Posterior P(known) after an observation with partial credit `score`."""
    p = min(max(p, 1e-4), 1 - 1e-4)
    post_c = p * (1 - slip) / (p * (1 - slip) + (1 - p) * guess)
    post_w = p * slip / (p * slip + (1 - p) * (1 - guess))
    post = score * post_c + (1 - score) * post_w
    if learn:
        post = post + (1 - post) * p_learn
    return min(max(post, 0.001), 0.999)


@dataclass
class SkillState:
    skill_id: str
    p: float = P_INIT
    n_attempts: int = 0
    n_correct: int = 0
    last_seen: str | None = None
    certified: bool = False
    review_interval: float = 0.0
    review_ease: float = 2.5
    review_due: str | None = None
    source: str | None = None

    @property
    def mastered(self) -> bool:
        return self.p >= MASTERED

    @property
    def assessed(self) -> bool:
        return self.n_attempts > 0 or self.source is not None


def load_states(con) -> dict[str, SkillState]:
    rows = con.execute("SELECT * FROM mastery").fetchall()
    states = {sid: SkillState(sid) for sid in content.skills_by_id()}
    for r in rows:
        if r["skill_id"] in states:
            states[r["skill_id"]] = SkillState(
                r["skill_id"], r["p"], r["n_attempts"], r["n_correct"], r["last_seen"],
                bool(r["certified"]), r["review_interval"], r["review_ease"], r["review_due"], r["source"])
    return states


def save_state(con, s: SkillState) -> None:
    con.execute(
        """INSERT INTO mastery(skill_id,p,n_attempts,n_correct,last_seen,certified,review_interval,review_ease,review_due,source)
           VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(skill_id) DO UPDATE SET p=excluded.p, n_attempts=excluded.n_attempts,
             n_correct=excluded.n_correct, last_seen=excluded.last_seen, certified=excluded.certified,
             review_interval=excluded.review_interval, review_ease=excluded.review_ease,
             review_due=excluded.review_due, source=excluded.source""",
        (s.skill_id, s.p, s.n_attempts, s.n_correct, s.last_seen, int(s.certified),
         s.review_interval, s.review_ease, s.review_due, s.source))


def _schedule_review(s: SkillState, success: bool, today: datetime) -> None:
    """SM-2-lite: once a skill is mastered it is brought back at growing intervals."""
    if not s.mastered and s.review_interval == 0:
        return
    if success:
        s.review_interval = 1.0 if s.review_interval == 0 else max(1.0, s.review_interval * s.review_ease)
        s.review_ease = min(3.0, s.review_ease + 0.05)
    else:
        s.review_interval = 1.0
        s.review_ease = max(1.3, s.review_ease - 0.2)
    s.review_due = (today + timedelta(days=s.review_interval)).date().isoformat()


def record_attempt(con, states: dict[str, SkillState], skill_id: str, correct: bool,
                   rubric_fraction: float | None = None, learn: bool = True,
                   propagate: bool = True, today: datetime | None = None) -> dict:
    """Update mastery after one graded item. Returns a summary of changes (for gap detection)."""
    today = today or datetime.now()
    s = states[skill_id]
    before = s.p
    was_mastered = s.mastered
    score = combined_score(correct, rubric_fraction)
    s.p = bkt_update(s.p, score, learn=learn)
    s.n_attempts += 1
    s.n_correct += int(correct)
    s.last_seen = today.isoformat(timespec="seconds")
    if s.mastered or s.review_interval > 0:
        _schedule_review(s, success=score >= 0.6, today=today)
    if was_mastered and score < 0.6:
        s.certified = False
    save_state(con, s)

    weakened = []
    if propagate and score < 0.6:
        for pid in content.skills_by_id()[skill_id].prereqs:
            ps = states[pid]
            if not ps.assessed:
                continue
            old = ps.p
            ps.p = max(0.05, ps.p * PREREQ_PENALTY)
            save_state(con, ps)
            if ps.p < WEAK <= old or ps.p < WEAK:
                weakened.append(pid)
    return {"skill": skill_id, "before": before, "after": s.p, "score": score, "weakened": weakened}


def set_prior(con, states: dict[str, SkillState], skill_id: str, p: float, source: str) -> None:
    s = states[skill_id]
    s.p = p
    s.source = source
    if p >= MASTERED and s.review_interval == 0:
        s.review_interval = 3.0
        s.review_due = (datetime.now() + timedelta(days=3)).date().isoformat()
    save_state(con, s)
