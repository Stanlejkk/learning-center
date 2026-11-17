"""Problem variety: every skill-level must offer several genuinely different problem *types*,
not the same sentence with new numbers (learners notice and it feels like going in circles)."""
import re

import pytest

from engine import content

MIN_TEMPLATES = 3
SEEDS = 40


def signature(prompt: str) -> str:
    """Prompt with math, numbers and single-letter variables removed: two problems with the same
    signature are the same *type* of problem."""
    s = re.sub(r"\$\$.*?\$\$|\$[^$]*\$", "M", prompt, flags=re.S)
    s = re.sub(r"[-+]?\d+(\.\d+)?", "N", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s[:90]


def templates(skill: str, level: int) -> set[str]:
    return {signature(content.make_problem(skill, level, seed).prompt) for seed in range(SEEDS)}


CASES = [(s, l) for s in sorted(content.load_generators()) for l in (1, 2, 3)]


@pytest.mark.parametrize("skill,level", CASES, ids=[f"{s}:L{l}" for s, l in CASES])
def test_skill_level_has_varied_problem_types(skill, level):
    t = templates(skill, level)
    assert len(t) >= MIN_TEMPLATES, f"{skill} L{level}: only {len(t)} problem type(s): {sorted(t)}"
