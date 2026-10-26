# Ditto fork — judge modernization

Fork of [`xiaowu0162/LongMemEval`](https://github.com/xiaowu0162/LongMemEval),
MIT licensed, © 2024 Di Wu. Upstream base:
`9e0b455f4ef0e2ab8f2e582289761153549043fc` (`main` HEAD as of 2026-10-26).

This fork exists for one reason: **`gpt-4o-2024-08-06`, the official judge, is
hard-coded in the evaluator and retires on 2027-01-14 (OpenAI API) and
2027-04-14 (Azure).** After those dates no one can reproduce a LongMemEval
score with the evaluator as published.

Swapping the judge is not a model-name change, because the protocol assumes a
non-reasoning judge. That is what this fork fixes.

## The delta — three hunks in `src/evaluation/evaluate_qa.py`

**1. `model_zoo` gains `deepseek-v4.1-flash`.** Upstream knows three judges, two
of which are OpenAI models.

**2. `max_tokens` becomes per-judge (`max_tokens_zoo`), defaulting to upstream's
10.** This is the substantive fix. At `max_tokens=10` a reasoning judge spends
the entire budget on reasoning tokens and returns `content=""` with
`finish_reason="length"`. Upstream then evaluates `'yes' in ''` as `False` and
scores **every question "no"** — a run that completes, reports total failure,
and raises no error anywhere. Measured under the official convention:

| judge | empty responses |
|---|---|
| `qwen3.8-flash` | 100% |
| `mistral-large-4-0` | 99.3% |
| `deepseek-v4.1-flash` | 92% |
| `kimi-k3` | 75.9% |
| `glm-5.3-flash` | 72.4% |
| `gpt-6-luna` | 23.9% |

`reasoning: {"enabled": false}` drops every one of those to 0.0%, but it is not
a general fix: `glm-5.3-flash` and `minimax-m3.1-flash` reject that flag on
every call — *"Reasoning is mandatory for this endpoint and cannot be
disabled."* So the bound has to be raised instead.

8192 is set from measurement, not convention. Over the 76 probes that truncated
`deepseek-v4.1-flash` at a 1024 bound, re-run at 8192 × 3 replicates, the
observed completion maximum was **2,907 tokens with zero truncations**. The
bound is a cap, not a charge — mean observed completion is 62 tokens.

**3. An empty completion raises instead of being labelled "no".** Even with a
generous bound, a truncated or empty response is malformed, and silently
reading it as a negative verdict is the failure mode above in miniature.

## What is deliberately unchanged

- **The five prompt templates in `get_anscheck_prompt` are byte-identical.**
  This is the property that makes a substituted run comparable at all, and it is
  checked mechanically — not asserted — by
  `docs/longmemeval-benchmark/judge-evaluator/` in `ditto-assistant/ditto-subnet`.
- The call shape otherwise: `temperature=0`, `n=1`, one call per question.
- The `'yes' in response.lower()` labelling rule, including its
  case-insensitive substring semantics.
- The dataset, the retrieval code, and every other file in the repository.
- `gpt-4o` behaviour: it keeps the default bound of 10, so the official path
  through this fork is byte-equivalent to upstream.

## What a run of this fork is, and is not

A result produced with a substituted judge is **not an official-protocol
LongMemEval score** and must not be compared against published LongMemEval
numbers. Agreement between judges is high but not perfect, and the incumbent
does not fully honour its own rubric: `gpt-4o` respects the temporal task's
"do not penalize off-by-one errors" clause on only 31/53 (58%) of the cases
that trigger it. Picking a successor on agreement alone reproduces that defect
and calls it compatibility.

`temperature=0` is also not determinism: 20/1,893 (1.06%) of byte-identical
replicates were non-unanimous for `gpt-4o` itself, consistent with the
published audit figure of 3/500. A replacement judge should be measured against
that floor, not against perfection.

The full judge-substitution study — 102,222 judge calls, 9 models, 2 calling
conventions, 3 replicates — is published in `ditto-assistant/ditto-subnet`
under `docs/longmemeval-benchmark/judge-study/`.
