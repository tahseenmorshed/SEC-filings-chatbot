"""HTTP API: the external contract for the language-agnostic evaluation harness.

A deliberately thin layer — it wires the existing pipeline (index → retriever →
answerer) once at startup and exposes it as structured JSON. All guarantees live in
the layers below; the API adds only transport, validation, and introspection
endpoints (/search, /chunks/{id}) so an external harness can probe retrieval and
independently re-verify cited quotes without filesystem access.
"""
