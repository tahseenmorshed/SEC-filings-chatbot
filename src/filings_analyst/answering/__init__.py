"""Grounded answering: question → retrieved chunks → verified, cited claims.

The defining invariant of this package: **no claim reaches a response without passing
mechanical verification** — its supporting quote must appear verbatim (canonical
comparison) inside the chunk it cites, and that chunk must be one actually retrieved
for the question. The LLM drafts; our code verifies; anything unverifiable is dropped,
and an answer with nothing left becomes a refusal. Refusal is a first-class outcome
with machine-readable reasons, not an error.
"""
