---
name: explaining-code
description: Use this skill when explaining code, APIs, libraries, frameworks, or technical mechanics — including inline commentary that comes up as part of answering another question. Triggers on phrases like "what does this do", "explain this", "how does X work", "walk me through", and on responses that include commentary on code. The skill has two rules: (1) explanations should be maximally didactic and transparent, assuming novice familiarity with the specific API and naming non-obvious mechanics (iterators vs lists, in-place vs copy, anonymous lambdas filling named slots, hidden defaults, scoping rules, `restrict` semantics, linkage of `inline`, jargon); (2) avoid all-caps words for emphasis in code comments, raw block text, or prose.
---

# Explaining code

The user's global, default preference for how code and technical material gets explained — in any project, any language, any framework. Two rules:

1. **Maximally didactic and transparent.** Assume novice familiarity with every API or library involved (PyTorch, numpy, OpenMP, FFTW, whatever). Name every non-obvious mechanic — iterator vs list, in-place vs copy, what anonymous values fill which named slots, what `restrict` actually promises, why `static inline` differs from bare `inline`. Walk through what each token does. Define jargon in plain words before using it. Never call a step "obvious."

2. **No all-caps words for emphasis.** Don't write "ALREADY computed" or "a LIST of tensors" in code comments, raw cells, or chat prose. Emphasize through bold/italic markdown or sentence structure instead. Acronyms and notation (H for the Hessian, SGD, GPU, API, JSON) stay capitalized because they are names, not emphasis.

## When triggered, read the reference

Read `references/didactic-transparent-explanations.md` before writing the explanation. It contains:

- five worked examples showing the required granularity — three from Python/PyTorch (iterator/`.device`, `nonlocal`, lambda-as-anonymous-argument) and two from CosmoLike/C (`restrict` aliasing, `static inline` linkage)
- the full catalog of "things that bite," split into higher-level (Python/NumPy/PyTorch/JAX) and lower-level (C/Fortran/OpenMP/SIMD)
- the consistency-over-cleverness rule and its concrete failure mode
- detail on what counts as a name vs an emphasis (allowed acronyms, notation, title case)
- alternatives to all-caps emphasis

The worked examples are the most important part — they convey the granularity standard better than any abstract rule.
