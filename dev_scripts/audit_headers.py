#!/usr/bin/env python3
"""List functions whose header block misses the strict standard.

Usage: python3 audit_headers.py <file.c|file.cpp> [...]

A function definition is a line at column zero that looks like a
signature start (type name followed eventually by an opening brace
before a semicolon). The contiguous // block directly above it (blank
lines allowed inside the block boundary scan) must contain the literal
section labels "Parameters:" and "Returns:", and the fenced
"// ---" open line. Functions with an empty parameter list (void) are
exempt from "Parameters:"; nothing is exempt from "Returns:".
main() and PYBIND11_MODULE are skipped.

Exit 0 when every file passes; 1 otherwise.
"""
import re
import sys

SIG = re.compile(
    r"^(?:static\s+)?(?:inline\s+)?"
    r"(?:const\s+)?[A-Za-z_][A-Za-z0-9_:<>,\s\*&]*?"
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\($")
SIG1 = re.compile(
    r"^(?:static\s+)?(?:inline\s+)?"
    r"(?:const\s+)?[A-Za-z_][A-Za-z0-9_:<>,\s\*&]*?"
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(.*\)\s*\{?\s*$")
SKIP = {"main", "PYBIND11_MODULE", "if", "for", "while", "switch", "return",
        "sizeof", "defined"}

fail = 0
for path in sys.argv[1:]:
    lines = open(path).read().split("\n")
    bad = []
    for i, line in enumerate(lines):
        if line.startswith(" ") or line.startswith("\t") or not line:
            continue
        m = SIG.match(line) or SIG1.match(line)
        if not m or m.group(1) in SKIP:
            continue
        # find the real body start: the first "{" at column zero (or line
        # end) within the next 40 lines, with no ";" ending the decl first
        isdef = False
        for j in range(i, min(i + 40, len(lines))):
            t = lines[j].rstrip()
            if t.endswith(";"):
                break
            if t == "{" or t.endswith("{"):
                isdef = True
                break
        if not isdef:
            continue
        # collect the contiguous comment block directly above
        k = i - 1
        block = []
        while k >= 0 and (lines[k].lstrip().startswith("//")
                          or lines[k].strip() == ""):
            if lines[k].strip() == "" and block:
                break
            if lines[k].lstrip().startswith("//"):
                block.append(lines[k])
            k -= 1
        text = "\n".join(block)
        missing = []
        if "// ---" not in text:
            missing.append("fence")
        # void parameter list exemption
        params_needed = "(void)" not in line and not line.rstrip().endswith("(void)")
        sig_tail = "".join(lines[i:i + 12])
        if "(void)" in sig_tail.split("{")[0].replace(" ", ""):
            params_needed = False
        if params_needed and "Parameters:" not in text:
            missing.append("Parameters:")
        if "Returns:" not in text:
            missing.append("Returns:")
        if missing:
            bad.append((i + 1, m.group(1), ", ".join(missing)))
    tag = "PASS" if not bad else f"{len(bad)} FAILING"
    print(f"== {path}: {tag}")
    for ln, name, miss in bad:
        print(f"   {ln:6d}  {name:44s} missing {miss}")
    fail += len(bad)
sys.exit(1 if fail else 0)
