"""Answer checking: symbolic / numeric / matrix / list / set / choice equivalence.

The user only types FINAL answers (the working is on paper). This module decides
whether a typed answer is equivalent to the reference answer.

Answer kinds
------------
expr    symbolic expression, e.g. "2*(x+1)" == "2x+2"         (sympy simplify + random numeric probing)
num     a number, tolerance-based, e.g. "0.333" ~ "1/3"
matrix  "[[1,2],[3,4]]"  (entries may be expressions)
vector  "[1, 2, 3]" or "(1,2,3)"
list    ordered list of expressions
set     unordered collection, e.g. roots "{2, -3}" or "2, -3"
choice  multiple choice letter, e.g. "B"
text    exact match after normalisation (case/space-insensitive)
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

# Names the user may type. Anything else (single letters) becomes a Symbol.
SAFE_NAMES = {
    "pi": sp.pi, "e": sp.E, "E": sp.E, "i": sp.I, "I": sp.I, "j": sp.I,
    "sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin, "acos": sp.acos,
    "atan": sp.atan, "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "abs": sp.Abs, "Abs": sp.Abs,
    "oo": sp.oo, "inf": sp.oo, "infinity": sp.oo, "binomial": sp.binomial,
    "factorial": sp.factorial, "C": sp.binomial, "sigma": lambda z: 1 / (1 + sp.exp(-z)), "sigmoid": lambda z: 1 / (1 + sp.exp(-z)),
    "log2": lambda z: sp.log(z, 2), "log10": lambda z: sp.log(z, 10),
    "Rational": sp.Rational, "Integer": sp.Integer, "Float": sp.Float, "Symbol": sp.Symbol,
    "conjugate": sp.conjugate,
}

_ALLOWED = re.compile(r"^[\w\s\.\+\-\*/\^\(\)\[\]\{\},=<>!|']*$")
_BANNED = ("__", "import", "lambda", "eval", "exec", "open", "os.", "sys.", "getattr", "globals")


class ParseError(ValueError):
    pass


def _clean(s: str) -> str:
    s = str(s).strip()
    s = s.replace("−", "-").replace("×", "*").replace("·", "*").replace("÷", "/")
    s = s.replace("√", "sqrt").replace("π", "pi").replace("∞", "oo").replace("²", "^2").replace("³", "^3")
    s = s.replace("\\cdot", "*").replace("\\pi", "pi").replace("\\sqrt", "sqrt").replace("\\infty", "oo")
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = s.replace("{", "(").replace("}", ")") if not s.startswith("{") else s
    return s


def parse(s: str) -> sp.Expr:
    """Parse a user/reference expression string into sympy, safely."""
    if isinstance(s, (int, float)):
        return sp.nsimplify(s) if isinstance(s, int) else sp.Float(s)
    if isinstance(s, sp.Basic):
        return s
    s = _clean(s)
    if not s:
        raise ParseError("empty answer")
    if not _ALLOWED.match(s) or any(b in s for b in _BANNED):
        raise ParseError(f"unsupported characters in {s!r}")
    s = s.replace("|", "")  # tolerate |x| typed loosely is ambiguous; prefer abs()
    try:
        local = dict(SAFE_NAMES)
        for name in set(re.findall(r"[A-Za-z_]\w*", s)):
            if name not in local:
                local[name] = sp.Symbol(name)
        return parse_expr(s, local_dict=local, transformations=TRANSFORMS, evaluate=True)
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"could not parse {s!r}: {exc}") from exc


def _split_top(s: str) -> list[str]:
    """Split on commas that are not nested inside brackets."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur))
    return [p.strip() for p in out]


def _strip_brackets(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] in "([{" and s[-1] in ")]}":
        inner = s[1:-1]
        # only strip if the brackets enclose the whole thing
        depth = 0
        ok = True
        for k, ch in enumerate(inner):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if depth < 0:
                ok = False
                break
        if not ok:
            break
        s = inner.strip()
    return s


def parse_seq(s) -> list[sp.Expr]:
    if isinstance(s, (list, tuple)):
        return [parse(x) if not isinstance(x, (list, tuple)) else parse_seq(x) for x in s]
    s = _clean(str(s)) if not str(s).strip().startswith("{") else str(s).strip()
    # allow a scalar factored outside the brackets: [1, 0, 0, 1]/sqrt(2)  or  1/sqrt(2)*[1, i]
    m = re.match(r"^\s*(\[[^\[\]]*\])\s*([*/])\s*(.+)$", s) or re.match(r"^\s*(.+?)\s*\*\s*(\[[^\[\]]*\])\s*$", s)
    if m:
        if m.group(1).startswith("["):
            vec, op, scal = m.group(1), m.group(2), parse(m.group(3))
            factor = scal if op == "*" else 1 / scal
        else:
            vec, factor = m.group(2), parse(m.group(1))
        return [sp.simplify(factor * v) for v in parse_seq(vec)]
    s = _strip_brackets(s)
    return [parse(p) for p in _split_top(s) if p != ""]


def parse_matrix(s) -> sp.Matrix:
    if isinstance(s, sp.MatrixBase):
        return sp.Matrix(s)
    if isinstance(s, (list, tuple)):
        rows = [[parse(x) for x in row] if isinstance(row, (list, tuple)) else [parse(row)] for row in s]
        return sp.Matrix(rows)
    txt = str(s).strip()
    txt = txt.replace(";", "],[") if ";" in txt and "[[" not in txt else txt
    if txt.startswith("[") and not txt.startswith("[["):
        txt = "[" + txt + "]"
    inner = _strip_brackets(txt) if txt.startswith("[[") else txt
    rows_txt = _split_top(inner)
    rows = [[parse(x) for x in _split_top(_strip_brackets(r))] for r in rows_txt]
    return sp.Matrix(rows)


def _numeric_close(a: sp.Expr, b: sp.Expr, rel: float, abs_tol: float) -> bool:
    try:
        va, vb = complex(sp.N(a)), complex(sp.N(b))
    except (TypeError, ValueError):
        return False
    if any(math.isnan(v) for v in (va.real, va.imag, vb.real, vb.imag)):
        return False
    return abs(va - vb) <= max(abs_tol, rel * max(abs(va), abs(vb)))


def expr_equal(a, b, rel: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    """Symbolic equivalence with numeric probing fallback."""
    a, b = parse(a), parse(b)
    if a == b:
        return True
    if a in (sp.oo, -sp.oo, sp.zoo) or b in (sp.oo, -sp.oo, sp.zoo):
        return a == b
    syms = sorted((a.free_symbols | b.free_symbols), key=str)
    if not syms:
        return _numeric_close(a, b, rel, abs_tol)
    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    # random probing at several points
    rng = random.Random(1234)
    hits = 0
    for _ in range(12):
        sub = {s: sp.Float(rng.uniform(0.3, 2.7)) for s in syms}
        try:
            va, vb = complex(a.evalf(subs=sub)), complex(b.evalf(subs=sub))
        except (TypeError, ValueError):
            continue
        if abs(va - vb) > max(1e-7, 1e-6 * max(abs(va), abs(vb))):
            return False
        hits += 1
    return hits >= 6


@dataclass
class CheckResult:
    correct: bool
    message: str = ""
    misconception: str | None = None


def check(user: str, answer, kind: str = "expr", tol: float | None = None,
          misconceptions: list | None = None) -> CheckResult:
    """Return whether `user` matches `answer`.

    misconceptions: list of (wrong_answer, tag) pairs; if the user's wrong answer
    matches one of them we report that tag (used for gap detection).
    """
    if user is None or str(user).strip() == "":
        return CheckResult(False, "no answer")
    if str(user).strip().lower() in {"?", "idk", "skip", "dont know", "don't know"}:
        return CheckResult(False, "skipped", "unknown")
    try:
        ok = _check_kind(user, answer, kind, tol)
    except ParseError as exc:
        return CheckResult(False, f"Could not read your answer: {exc}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(False, f"Could not compare answer ({type(exc).__name__}: {exc})")
    if ok:
        return CheckResult(True, "correct")
    tag = None
    for wrong, t in misconceptions or []:
        try:
            if _check_kind(user, wrong, kind, tol):
                tag = t
                break
        except Exception:  # noqa: BLE001
            continue
    return CheckResult(False, "incorrect", tag)


def _check_kind(user, answer, kind: str, tol: float | None) -> bool:
    rel = tol if tol is not None else 1e-6
    abs_t = tol / 10 if tol is not None else 1e-9
    if kind == "choice":
        return str(user).strip().upper().strip("().") == str(answer).strip().upper()
    if kind == "text":
        norm = lambda x: re.sub(r"\s+", "", str(x).lower()).lstrip("+")  # noqa: E731  ('+XZ' == 'XZ')
        options = answer if isinstance(answer, (list, tuple)) else [answer]
        return any(norm(user) == norm(o) for o in options)
    if kind == "num":
        a, b = parse(user), parse(answer)
        # with an explicit tol: relative tolerance tol, absolute floor tol/10 (so tiny answers stay strict-ish)
        abs_tol = tol / 10 if tol is not None else 1e-9
        return _numeric_close(a, b, rel=rel, abs_tol=abs_tol)
    if kind == "expr":
        return expr_equal(user, answer, rel=rel)
    if kind in ("vector", "list"):
        ua, aa = parse_seq(user), parse_seq(answer)
        return len(ua) == len(aa) and all(expr_equal(x, y, rel=rel, abs_tol=abs_t) for x, y in zip(ua, aa))
    if kind == "set":
        ua, aa = parse_seq(user), parse_seq(answer)
        if len(ua) != len(aa):
            return False
        remaining = list(aa)
        for x in ua:
            for k, y in enumerate(remaining):
                if expr_equal(x, y, rel=rel, abs_tol=abs_t):
                    remaining.pop(k)
                    break
            else:
                return False
        return True
    if kind == "matrix":
        ua, aa = parse_matrix(user), parse_matrix(answer)
        if ua.shape != aa.shape:
            return False
        return all(expr_equal(x, y, rel=rel, abs_tol=abs_t) for x, y in zip(ua, aa))
    raise ValueError(f"unknown answer kind {kind!r}")


def fmt(x) -> str:
    """Format a sympy object / python value as a canonical answer string."""
    if isinstance(x, sp.MatrixBase):
        return "[" + ", ".join("[" + ", ".join(fmt(v) for v in row) + "]" for row in x.tolist()) + "]"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(fmt(v) for v in x) + "]"
    if isinstance(x, float):
        return repr(round(x, 10))
    return str(sp.sstr(x))
