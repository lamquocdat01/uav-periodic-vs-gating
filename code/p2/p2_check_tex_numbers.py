"""Check that the paper text contains no hand-typed data numbers (rule L4).

Scans paper/main.tex and every file it pulls in with \\input / \\include (recursively), except numbers.tex.
A "numeric literal" is a digit run not preceded by a letter/digit/backslash (so E3, P1, G5, VisDrone2019,
YOLOv8 are names, not numbers). Ignored regions:
  - comments (from an unescaped % to end of line)
  - the preamble of main.tex (everything before \\begin{document})
  - math: $...$, $$...$$, \\(...\\), \\[...\\], and equation/align/gather/multline/eqnarray/IEEEeqnarray/math/
    displaymath environments (starred or not)
  - arguments (all following [..] / {..} groups) of: \\cite* \\ref \\eqref \\autoref \\cref \\label \\includegraphics
    \\input \\include \\usepackage \\documentclass \\bibliography \\bibliographystyle \\begin \\end \\setlength
    \\addtolength \\setcounter \\vspace \\hspace \\newcommand \\renewcommand \\graphicspath \\url \\rule \\fontsize
    \\cline \\cmidrule \\resizebox \\scalebox \\raisebox \\parbox \\makebox \\multicolumn(first two) \\multirow(first two)
    \\href(first) \\IfFileExists(first) \\linebreak \\pagebreak \\enlargethispage;
  - optional argument of a line break \\\\[..]
Prints file:line: context for every hit; exit code 1 if any, else 0.

Run: .venv/Scripts/python.exe code/p2/p2_check_tex_numbers.py [path/to/main.tex]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "paper" / "main.tex"
SKIP_FILES = {"numbers.tex"}

MATH_ENVS = ["equation", "align", "gather", "multline", "eqnarray", "IEEEeqnarray", "IEEEeqnarraybox",
             "math", "displaymath", "flalign", "alignat", "split", "array"]
ALL_ARGS = ["cite", "citep", "citet", "nocite", "ref", "eqref", "autoref", "cref", "Cref", "pageref", "label",
            "includegraphics", "input", "include", "usepackage", "documentclass", "bibliography",
            "bibliographystyle", "setlength", "addtolength", "setcounter", "addtocounter", "vspace",
            "hspace", "newcommand", "renewcommand", "providecommand", "graphicspath", "url", "rule", "fontsize",
            "cline", "cmidrule", "linebreak", "pagebreak", "enlargethispage", "newtheorem", "IEEEaftertitletext",
            "raisebox", "parbox", "makebox", "framebox"]
FIRST_N = {"multicolumn": 2, "multirow": 2, "href": 1, "IfFileExists": 1, "resizebox": 2, "scalebox": 1}
TABULAR_ENVS = {"tabular", "tabular*", "tabularx", "array", "minipage", "table", "table*", "figure", "figure*"}

NUM_RE = re.compile(r"(?<![A-Za-z0-9\\@])\d+(?:[.,]\d+)*")


class Masker:
    def __init__(self, text: str):
        self.t = text
        self.m = list(text)

    def blank(self, i: int, j: int) -> None:
        for k in range(max(i, 0), min(j, len(self.m))):
            if self.m[k] != "\n":
                self.m[k] = " "

    def result(self) -> str:
        return "".join(self.m)


def group_end(t: str, i: int) -> int:
    """t[i] is '{' or '['; return index after the matching close (brace-aware)."""
    open_c = t[i]
    close_c = "}" if open_c == "{" else "]"
    depth = 0
    k = i
    while k < len(t):
        c = t[k]
        if c == "\\":
            k += 2
            continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return k + 1
        elif open_c == "[" and c == "{":  # skip nested braces inside [...]
            k = group_end(t, k)
            continue
        k += 1
    return len(t)


def following_groups(t: str, i: int, limit: int | None = None) -> int:
    """From i, skip whitespace + consecutive [..]/{..} groups (at most `limit` groups); return end index."""
    n = 0
    k = i
    while k < len(t):
        j = k
        while j < len(t) and t[j] in " \t":
            j += 1
        if j < len(t) and t[j] in "[{" and (limit is None or n < limit):
            k = group_end(t, j)
            n += 1
        else:
            break
    return k


def mask_comments(M: Masker) -> None:
    t = M.t
    for mo in re.finditer(r"(?<!\\)%[^\n]*", t):
        # an escaped percent is preceded by an odd number of backslashes
        s = mo.start()
        bs = 0
        while s - 1 - bs >= 0 and t[s - 1 - bs] == "\\":
            bs += 1
        if bs % 2 == 0:
            M.blank(mo.start(), mo.end())


def mask_math_and_args(M: Masker, text: str) -> None:
    t = text
    i = 0
    n = len(t)
    env_re = re.compile(r"\\begin\{(" + "|".join(MATH_ENVS) + r")(\*?)\}")
    while i < n:
        c = t[i]
        if c == "\\":
            if t.startswith("\\\\", i):  # line break, optional [..]
                j = i + 2
                if j < n and t[j] == "[":
                    M.blank(j, group_end(t, j))
                    i = group_end(t, j)
                else:
                    i = j
                continue
            if i + 1 < n and t[i + 1] in "$%{}&#_":
                i += 2
                continue
            if t.startswith("\\(", i):
                j = t.find("\\)", i + 2)
                j = n if j < 0 else j + 2
                M.blank(i, j)
                i = j
                continue
            if t.startswith("\\[", i):
                j = t.find("\\]", i + 2)
                j = n if j < 0 else j + 2
                M.blank(i, j)
                i = j
                continue
            me = env_re.match(t, i)
            if me:
                end_tag = f"\\end{{{me.group(1)}{me.group(2)}}}"
                j = t.find(end_tag, me.end())
                j = n if j < 0 else j + len(end_tag)
                M.blank(i, j)
                i = j
                continue
            mc = re.match(r"\\([A-Za-z@]+)\*?", t[i:])
            if mc:
                name = mc.group(1)
                k = i + mc.end()
                if name in ALL_ARGS:
                    if name == "cmidrule":  # booktabs trim spec \cmidrule(lr){2-4}
                        j = k
                        while j < n and t[j] in " \t":
                            j += 1
                        if j < n and t[j] == "(":
                            close = t.find(")", j)
                            k = n if close < 0 else close + 1
                    e = following_groups(t, k)
                    M.blank(i, e)
                    i = e
                    continue
                if name in FIRST_N:
                    e = following_groups(t, k, FIRST_N[name])
                    M.blank(i, e)
                    i = e
                    continue
                if name in ("begin", "end"):
                    e = following_groups(t, k, 1)
                    env = t[k:e].strip()[1:-1] if e > k else ""
                    if name == "begin" and env in TABULAR_ENVS:
                        e = following_groups(t, e)
                    M.blank(i, e)
                    i = e
                    continue
                # plain control word (incl. \NumXxx macros): blank the name only
                M.blank(i, k)
                i = k
                continue
            i += 1
            continue
        if c == "$":
            if t.startswith("$$", i):
                j = t.find("$$", i + 2)
                j = n if j < 0 else j + 2
            else:
                j = i + 1
                while j < n and not (t[j] == "$" and t[j - 1] != "\\"):
                    j += 1
                j = min(j + 1, n)
            M.blank(i, j)
            i = j
            continue
        i += 1


def inputs_of(text: str, base: Path) -> list[Path]:
    out = []
    for mo in re.finditer(r"\\(?:input|include)\{([^}]+)\}", text):
        name = mo.group(1).strip()
        p = base / name
        if p.suffix != ".tex":
            p = p.with_suffix(".tex")
        if p.name in SKIP_FILES:
            continue
        out.append(p)
    return out


def scan_file(path: Path, is_main: bool, seen: set[Path], hits: list) -> None:
    path = path.resolve()
    if path in seen or not path.exists():
        if not path.exists():
            print(f"warning: {path} not found")
        return
    seen.add(path)
    text = path.read_text(encoding="utf-8")
    M = Masker(text)
    mask_comments(M)
    stage1 = M.result()
    if is_main:
        k = stage1.find("\\begin{document}")
        if k > 0:
            M.blank(0, k)
            stage1 = M.result()
    M2 = Masker(stage1)
    mask_math_and_args(M2, stage1)
    masked = M2.result()
    lines = text.splitlines()
    for mo in NUM_RE.finditer(masked):
        ln = masked.count("\n", 0, mo.start()) + 1
        hits.append((path, ln, mo.group(0), lines[ln - 1].strip()[:140] if ln - 1 < len(lines) else ""))
    base = MAIN.parent if not is_main else path.parent
    for sub in inputs_of(stage1, path.parent if is_main else base):
        scan_file(sub, False, seen, hits)


def main() -> int:
    global MAIN
    if len(sys.argv) > 1:
        MAIN = Path(sys.argv[1]).resolve()
    hits: list = []
    seen: set[Path] = set()
    scan_file(MAIN, True, seen, hits)
    print(f"scanned {len(seen)} file(s): " + ", ".join(p.relative_to(MAIN.parent).as_posix() for p in sorted(seen)))
    if hits:
        print(f"FOUND {len(hits)} numeric literal(s) outside macros/math/comments:")
        for p, ln, num, ctx in hits:
            print(f"  {p.relative_to(MAIN.parent).as_posix()}:{ln}: '{num}'   | {ctx}")
        return 1
    print("OK: 0 data numbers in text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
