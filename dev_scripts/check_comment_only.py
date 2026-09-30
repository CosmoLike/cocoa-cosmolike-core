#!/usr/bin/env python3
"""Verify two C/C++ sources differ only in comments and whitespace.

Usage: python3 check_comment_only.py <original> <edited>
Exit 0 and print OK when the comment-stripped, whitespace-collapsed
token streams are identical; otherwise print the first divergence and
exit 1.
"""
import sys


def strip(src):
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            j = src.find('\n', i)
            i = n if j == -1 else j
        elif c == '/' and i + 1 < n and src[i + 1] == '*':
            j = src.find('*/', i + 2)
            i = n if j == -1 else j + 2
        elif c in ('"', "'"):
            q = c
            out.append(c)
            i += 1
            while i < n:
                out.append(src[i])
                if src[i] == '\\' and i + 1 < n:
                    out.append(src[i + 1])
                    i += 2
                    continue
                if src[i] == q:
                    i += 1
                    break
                i += 1
        else:
            out.append(c)
            i += 1
    return ' '.join(''.join(out).split())


a = strip(open(sys.argv[1]).read())
b = strip(open(sys.argv[2]).read())
if a == b:
    print(f"OK: {sys.argv[2]} is comment/whitespace-only vs {sys.argv[1]}")
    sys.exit(0)
k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
print("DIFF at token-stream offset", k)
print("  original:", a[max(0, k - 80):k + 80])
print("  edited:  ", b[max(0, k - 80):k + 80])
sys.exit(1)
