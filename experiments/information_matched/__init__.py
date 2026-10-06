"""Information-matched baselines, empirical-mean D-EGE, and early-curve artifacts.

Builds four alternative versions of the main publication figures:

* ``uv_naive`` / ``kv_naive``: naive doubling D-EGE (``-N``), Uniform, and
  APE / TTPFTS under matched noise information (unknown or known variance).
* ``uv_both`` / ``kv_both``: additionally the empirical-mean doubling variant
  of Jun & Nowak (2016), which recommends the Pareto set of every sample
  collected so far.

Existing results are reused verbatim and are only ever read; only APE-UV and
the empirical-mean D-EGE trajectories are computed here.
"""
