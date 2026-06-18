---
name: didactic-transparent-explanations
description: "Explain at a maximally didactic, transparent level — assume novice in ALL APIs, never hide non-obvious mechanics (name the anonymous lambda, spell out implicit arg positions, iterator-vs-list, in-place-vs-copy, hidden defaults, jargon)"
---

When explaining code or technical material to the user, default to MAXIMALLY DIDACTIC and TRANSPARENT explanations. Assume the user is a NOVICE in all programming APIs/libraries (PyTorch, numpy, or any framework) unless they say otherwise. Crucially, expertise in their own scientific domain (cosmology, CosmoLike, CAMB) does NOT transfer to terse code/API reasoning — stay step-by-step there too.

## What "maximally didactic" means, by example

These three illustrate the required GRANULARITY. The PyTorch specifics matter less than the level of detail — the point is to name every non-obvious mechanic a novice could not infer alone.

**1. Iterators / attribute access — `dev = next(model.parameters()).device`**
Don't just say "get the device." Explain: `model.parameters()` returns an ITERATOR (a lazy stream of the model's weight tensors), NOT a list — so you can't write `[0]`. `next(...)` pulls the FIRST item out of that stream (here, the first weight tensor). `.device` then reads which hardware that tensor lives on (`cpu` / `cuda` / `mps`). Since all of a model's parameters normally share one device, the first one tells you the model's device.

**2. Scoping rules that cause silent bugs — `nonlocal`**
Given an inner function that does `total += ...` where `total` belongs to the ENCLOSING function: explain WHY `nonlocal total` is needed. `total += x` is really `total = total + x` — an assignment. Assigning to a name inside a function makes Python treat that name as a BRAND-NEW LOCAL, so reading it first (`total + x`) crashes with `UnboundLocalError`. `nonlocal total` says "don't make a new local — use the `total` from the enclosing function," so `+=` updates the outer variable.

**3. Anonymous values filling named parameters — the one a novice would NEVER infer**
Given the API `saved_tensors_hooks(pack, unpack)` called as `with hooks(pack, lambda t: t):` — point at the exact token: that `lambda t: t` (an inline, unnamed one-line function meaning "take `t`, return it unchanged") IS the `unpack` argument. The novice sees no variable literally named `unpack`, so you must SAY that the anonymous lambda is filling that second slot. Always name the mapping between a positional value and the parameter it fills.

## The general rule these examples encode

Name anonymous constructs and state which named slot they fill. Flag the things that bite: iterator-vs-list, lazy-vs-eager, in-place-vs-copy, device placement, hidden defaults, implicit/positional arguments, integer-vs-float division, views-vs-copies. Define jargon in plain words BEFORE using it (e.g. pack/unpack = "store it / get it back"). Show the full chain of reasoning; never call a leap "obvious." When a token could confuse, pre-empt it.

CONSISTENCY OVER CLEVERNESS (a flagged failure mode). Reuse the patterns/APIs the user has ALREADY seen in earlier examples; do NOT reach for a different-but-equivalent tool out of thin air. The user does NOT hold the whole library API in their head — switching styles forces them to. Concrete failure: examples repeatedly used the module-style loss mse = nn.MSELoss(), then I abruptly used the functional F.mse_loss(...) for one call — which broke continuity, assumed API fluency, AND I forgot import torch.nn.functional as F so it errored. The right move was the SAME family they already knew: mse_sum = nn.MSELoss(reduction="sum"). Rules: (1) stay within the idioms already established in this artifact; (2) if a genuinely new API/concept IS warranted, INTRODUCE it explicitly — a one-line aside or a small raw block (e.g. "module vs functional") — never swap silently; (3) if you do bring in a new symbol, include its import.

**Why:** The user builds intuition through transparency, not expert-to-expert shorthand. In their words: "I am not a machine with millions of GPUs. I need transparency to build intuition." Terse, assumes-you-know explanations cause them to lose the thread — in ANY domain, including ones where they are the expert.

**Scope / how to apply:** It is the EXPLANATION-LEVEL preference and composes with the formatting rules in [[pytorch-narrow-code-for-slides]], [[pytorch-full-block-on-correction]], [[pytorch-raw-block-output]], and [[pytorch-notebook-readonly]]. Already known relevant to CosmoLike/Cocoa work, the [[axiecamb-port-project]], and anything that comes later.
