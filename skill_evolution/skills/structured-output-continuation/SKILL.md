---
name: structured-output-continuation
description: Complete oversized structured LLM stages through bounded complete calls and deterministic merging. Use when EvidenceReview, SafetyReflection, or AnswerContext must process enough independent evidence/claims that one function result risks max-output truncation; do not use to split QueryUnderstanding or to remove clinical timeline/schema fields.
metadata:
  call_stage: evidence_review|safety_reflection|answer_context
  when_to_use: a structured stage has more independent evidence or claim items than one response can safely return, or the same stage previously ended at max_output_tokens
  when_not_to_use: QueryUnderstanding semantic extraction, ordinary short outputs, or semantic evidence rejection
  trigger: oversized_repeated_structured_items
---

# Structured Output Continuation

Treat multiple calls as parts of one logical stage, not as independent clinical opinions.

## Apply

1. First attempt the stage as one complete structured call. Activate partition recovery only when that call returns incomplete/invalid function arguments or fails the stage's structured semantic validation; transport, authentication, and unavailable-model errors are not truncation signals.
2. On activation, partition only the repeated independent unit: evidence rows for EvidenceReview, eligible claims for SafetyReflection, and current claims for AnswerContext.
3. Give every recovery call the same task boundary and a numbered part. Require each part to return one complete schema-valid function result; never ask it to continue an incomplete JSON fragment.
4. Merge deterministically by stable evidence or claim ID. Preserve source order, discard duplicates, and derive any task-wide verdict only after merging all parts.
5. Validate IDs and evidence bindings after the merge. A part cannot introduce facts or IDs absent from its input.
6. Record activation, input item count, part count, batch size, initial-attempt status, and merge mode in the stage output.

## Clinical invariants

- Do not remove required schema fields merely to fit one response.
- Preserve baseline, prior treatment, current treatment, follow-up timepoints, and future outcomes as distinct temporal roles.
- Preserve event order without converting temporal sequence into drug causality.
- Later progression does not negate an earlier response window; combination outcomes do not become single-drug outcomes.
- Partial and analog evidence retain their original entity, outcome, and time scope.

## Boundaries

- QueryUnderstanding stays a single soft-schema task. If its JSON is incomplete or invalid, retry the whole task with an explicit complete-JSON instruction.
- Pre-filtering is permitted only when deterministic downstream eligibility is already known. Do not use truncation recovery as a semantic rejection rule.
- Stop when the configured stage or case call budget is exhausted; never silently replace a failed official benchmark stage with a fabricated result.
