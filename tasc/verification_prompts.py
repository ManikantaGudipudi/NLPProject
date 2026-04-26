"""
Prompts for the TASC verification loop (Critic).

Two independent audits:
  1. f(A) logic check  – is the mathematical approach correct for the question?
  2. A = C traceability – are the numeric inputs taken from the source context?

Plus a refinement template used by react_tool_verify to feed errors back.

DESIGN NOTE — small models do poorly with open-ended "is this correct?" prompts
(they either always FAIL or always PASS).  These prompts use a *structured
step-by-step checklist* that forces the model to show its work before judging.
"""

# ---------------------------------------------------------------------------
# f(A) — Logic / formula check
# ---------------------------------------------------------------------------

LOGIC_CHECK_SYSTEM = """\
You are auditing a model's mathematical reasoning for a financial question.

You must follow the steps below IN ORDER and write out each step before giving \
your verdict.

STEP 1 - QUESTION TYPE:
Read the question carefully. What specific quantity or comparison does it ask \
for? State it precisely. Examples of question types:
- "percentage change from year X to year Y"
- "ratio of A to B"
- "difference between X and Y"
- "total / sum of items"
- "yes/no: is X greater than Y?"
- "what percentage of total is item X?"

STEP 2 - MODEL'S OPERATION:
What mathematical operation did the model actually perform? Describe it \
concretely (e.g., "subtracted 400 from 500", "divided 100 by 200 and \
multiplied by 100").

STEP 3 - MATCH CHECK:
Does the operation in Step 2 produce the right TYPE of answer for Step 1? \
Common matches:
- "change/difference/increase/decrease" → subtraction ✓
- "percentage change" → (new−old)/old × 100 ✓  (just new−old ✗)
- "what percent of total" → part/total × 100 ✓  (just subtraction ✗)
- "ratio of A to B" → A / B ✓
- "total / sum" → addition ✓
If the model computed a raw difference but the question asks for a percentage, \
that is a mismatch.

Finally, write:
VERDICT: PASS (if Step 3 shows the operation matches the question type)
VERDICT: FAIL (if Step 3 shows a clear mismatch)
REASON: <one-to-two sentences>
"""

LOGIC_CHECK_USER = """\
## Source Context
{context}

## Question
{question}

## Model's Reasoning
{reasoning}

Now follow STEP 1, STEP 2, STEP 3 exactly, then give VERDICT and REASON."""


# ---------------------------------------------------------------------------
# A = C — Traceability / value-extraction check
# ---------------------------------------------------------------------------

TRACEABILITY_CHECK_SYSTEM = """\
You are auditing whether a model used the correct input numbers from the \
source context.

You must follow the steps below IN ORDER and write out each step before giving \
your verdict.

STEP 1 - INPUT NUMBERS:
List every number the model used as a DIRECT input to its calculation \
(not intermediate results or the final answer). For each, write:
  <value> = <what the model says it represents>

STEP 2 - SOURCE CHECK:
For EACH number from Step 1, find it in the source context. Quote the exact \
cell, row heading, column heading, or sentence where it appears. If a number \
does NOT appear anywhere in the source, write "NOT FOUND".

STEP 3 - CORRECT DATA POINT:
For each number, does it come from the correct row, column, year, and metric \
for what the question is asking? For example, if the question asks about \
"2019 revenue" but the model pulled a number from the "2018" column, that is \
wrong even if the number exists in the table.

Finally, write:
VERDICT: PASS (if ALL input numbers are found in the source AND correspond to \
the correct data points for the question)
VERDICT: FAIL (if ANY input number is not found, or is from the wrong \
row/column/year/metric)
REASON: <one-to-two sentences, citing any specific mismatched values>
"""

TRACEABILITY_CHECK_USER = """\
## Source Context
{context}

## Question
{question}

## Model's Reasoning
{reasoning}

Now follow STEP 1, STEP 2, STEP 3 exactly, then give VERDICT and REASON."""


# ---------------------------------------------------------------------------
# Refinement feedback (used only by react_tool_verify)
# ---------------------------------------------------------------------------

REFINEMENT_PREFIX = """\
IMPORTANT — A verification audit found issues with a previous attempt at this \
question. Read the feedback below carefully and avoid the same mistakes.

{feedback}

---
Now answer the question below. Follow the Thought / Action / Final Answer \
format exactly.
"""
