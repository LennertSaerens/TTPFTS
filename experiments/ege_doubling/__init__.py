"""Doubling-trick (anytime) EGE variant of the main publication figures.

Following Jun & Nowak (ICML 2016), fixed-budget EGE-SH/EGE-SR are turned into
anytime algorithms: stage ``s`` runs the unchanged algorithm from scratch with
budget ``2**(s-1) * K`` and ends at cumulative pull ``(2**s - 1) * K``. Between
stage endpoints the recommendation of the most recently *completed* stage is
held. Uniform, APE, and TTPFTS curves are reused verbatim from the
preregistered dataset, which this package only ever reads.
"""
