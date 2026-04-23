
# ml-study-lab

Retype-driven study notes for ML engineering — books, papers, and docs worked through from scratch, not skimmed.

Every chapter, paper, or doc I work through here follows the same method:
**read once, reframe in my own words, retype the code from a spec, then solve unseen exercises.**
No copy-paste. The goal is retention and building the muscle to write production code,
not collecting notes.

---

## Method

For each lesson:

1. **Concept** — the core idea, explained with an analogy that maps to the mechanics.
2. **Why it matters** — connection to production ML systems and reliability.
3. **Gap-fills** — what the source assumed, skipped, or glossed over.
4. **Retype program** — structural spec only; code is written from scratch.
   A reference implementation lives beside it for checking *after* attempting.
5. **Exercises** — concept questions and unseen code problems, solutions behind a divider.

---

## Folder convention

```
<source-type>/<source-short-name>/chNN-<topic-slug>/
├── README.md        ← the lesson
├── main.py          ← retype written from spec (no peeking)
├── reference.py     ← reference implementation
└── exercises.md     ← problems + solutions (divider-gated)
```

---

## Progress

### Books

| Source | Chapters done | Last updated | Notes |
|---|---|---|---|
| *Python for Data Analysis* — McKinney | 0 / — | — | In progress (ch. 4) |
| *AI Engineering* — Huyen | — | — | Planned |
| *Designing Machine Learning Systems* — Huyen | — | — | Planned |
| *Practical Data Quality* | — | — | Planned |

### Papers

*Added as they are worked through.*

### Docs / deep-dives

*Added as they come up in project work.*

---

## Why this repo exists

Reading technical books without building doesn't stick. This repo is the artifact of
treating each chapter as a small implementation problem instead of a passive read —
aligned with the broader goal of building ML systems that don't just work, but stay working.