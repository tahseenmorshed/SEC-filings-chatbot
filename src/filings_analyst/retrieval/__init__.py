"""Retrieval track: hybrid search (BM25 + embeddings, RRF-fused) over parsed chunks.

Design commitments (see README / design discussion):

* Two independent full searches per query — lexical BM25 and semantic embeddings —
  merged by reciprocal-rank fusion. No query routing, no staged cutoffs.
* Exact brute-force similarity over numpy arrays; no vector database (the whole
  index is ~1.5 MB — approximation would buy nothing and cost exactness).
* The *indexed* representation of a chunk is enriched with company/form/section
  context (the "the Company" problem), but the *cited* text is always the verbatim
  stored chunk — enrichment must never leak into results.
* An index records fingerprints of the chunk files it was built from and refuses to
  load if they've changed (stale offsets must never be served).
"""
