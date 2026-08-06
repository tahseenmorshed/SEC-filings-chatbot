"""Evaluation harness for the Grounded Filings Analyst.

This package never imports the target application's code — every probe goes through
its HTTP API (``/ask``, ``/search``, ``/chunks/{id}``, ``/health``). That is the whole
design point: a black-box harness cannot inherit the target's blind spots, and the
wire-shape expectations in ``wire_schema.py`` are an *independent* redefinition, not a
copy — if the real API drifts from what this harness expects, that drift is itself a
finding, not a shared assumption.

Test cases are plain YAML data (see ``cases/``), so the actual reference-runner
implementation here (Python + requests) is incidental — a runner in any language could
execute the same case files against the same contract.
"""
