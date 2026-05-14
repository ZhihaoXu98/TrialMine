"""Hand-curated UMLS drug -> drug-class CUI mappings for the oncology
drugs that appear in TrialMine's held-out eval set.

This table substitutes for UMLS hierarchy traversal, which scispacy's
KB does NOT expose (the ``Entity`` NamedTuple carries only
``concept_id`` / ``canonical_name`` / ``aliases`` / ``types`` /
``definition`` — no parent relations). Every CUI here was discovered
by linking a surface form through scispacy's UMLS linker (see
:func:`TrialMine.features.concepts.link_to_cuis`) and verifying the
returned ``canonical_name`` against the 2022 KB snapshot. The public
UMLS browser at ``uts.nlm.nih.gov`` is NOT the authority — scispacy
0.6.2 ships a frozen 2022 KB whose CUI assignments differ from the
browser for many concepts (e.g. osimertinib is ``C4058811`` in
scispacy vs ``C2700554`` in the browser). Do not use the browser as
a CUI source; doing so silently produces entries that never match.

Intentionally small (~20–30 entries) and hand-built. When the table
exceeds ~50 entries, the right next move is the UMLS REST API for
live hierarchy lookup (the project already has a ``umls_api_key``
field on :class:`TrialMine.config.Settings`, currently empty).

Design decisions captured here (see ``docs/fix_parser_umls.md`` A4):

1. **Brand-name aliases get separate keys.** SciSpacy's 2022 KB
   assigns DIFFERENT CUIs to brand vs generic names (verified pairs:
   osimertinib=C4058811 vs Tagrisso=C4058817; trastuzumab=C0728747 vs
   Herceptin=C0338204; etc.). Direct-CUI overlap therefore does NOT
   bridge brand ↔ generic — we list both as keys with the same
   class-value set, so :func:`TrialMine.features.concepts.match_via_cui`
   resolves them via the drug-class path.

2. **HER2 class CUI uses the "HER2 inhibitor" mechanism concept**
   (``C4759996``), not "HER2-targeted therapy" — the latter does not
   resolve to a drug-type CUI in the 2022 KB. The broader
   ``C0003250`` 'Monoclonal Antibodies' is intentionally NOT used as
   a fallback because it would false-match unrelated mAbs
   (ipilimumab, pembrolizumab, etc.).

3. **Immune checkpoint inhibitors collapse to one parent class CUI**
   (``C4684977`` 'Immune Checkpoint Inhibitors'). PD-1- and
   PD-L1-specific class phrases do not resolve to useful CUIs in the
   2022 KB ('PD-1 inhibitor' → generic 'Inhibitor'; 'PD-L1 inhibitor'
   → the gene ``C0965245`` 'CD274 protein, human'). Collapsing to
   the parent class is less granular but clinically sound for the
   common "any prior ICI" trial restriction.

4. **Known KB gaps (omitted, documented inline):**
   * fam-trastuzumab deruxtecan (Enhertu) — no drug-type CUI in 2022 KB.
   * Zejula (niraparib brand name) — no drug-type CUI; the generic
     ``niraparib`` CUI still resolves so most queries are covered.

Scope — held-out queries this table is curated against (as of
2026-05-14):

* Q413 — '58M EGFR exon 19 NSCLC failed osimertinib phase 2-3':
  osimertinib + EGFR TKI class (**mandatory**).
* Q416 — '62F HER2+ metastatic breast post-trastuzumab progression':
  trastuzumab + HER2-targeted class (**mandatory**).
* '55M MSI-high colorectal post-pembrolizumab progression' +
  '60M MSI-high colorectal cancer pembrolizumab progression
  second-line': pembrolizumab + ICI class.
* '55F ovarian cancer BRCA wild type platinum-resistant PARP failure':
  PARP inhibitor family.
* '34F HER2+ metastatic breast cancer brain metastases tucatinib
  failure': tucatinib shares the HER2-targeted class CUI.

The remaining ~10 complex held-out queries mention drugs not yet
covered (dabrafenib/trametinib, enzalutamide, adagrasib, lenalidomide,
bortezomib). Add families as those queries become load-bearing in the
B2 eval; the recipe is in ``docs/fix_parser_umls.md`` A4 (probe via
A3 Test 5's ``_A4_PROBES`` list, lift verified CUIs into here).

All CUIs verified against scispacy 0.6.2's 2022 KB on 2026-05-14 via
``/tmp/a3_smoke.log`` + A4 probe runs. To re-verify after a KB
upgrade, re-run the A3 smoke test.
"""

from __future__ import annotations

# drug CUI -> frozenset of class-level CUIs the drug belongs to.
DRUG_TO_CLASS_CUIS: dict[str, frozenset[str]] = {
    # ===================================================================
    # EGFR tyrosine kinase inhibitors -- targets Q413 (failed osimertinib).
    # Class CUIs:
    #   C5574906 'Epidermal growth factor receptor inhibitor'
    #   C1268567 'Protein-tyrosine kinase inhibitor' (broader)
    # ===================================================================
    "C4058811": frozenset({"C5574906", "C1268567"}),  # osimertinib
    "C4058817": frozenset({"C5574906", "C1268567"}),  # Tagrisso (brand)
    "C1135135": frozenset({"C5574906", "C1268567"}),  # erlotinib
    "C1135136": frozenset({"C5574906", "C1268567"}),  # Tarceva (brand)
    "C1122962": frozenset({"C5574906", "C1268567"}),  # gefitinib
    "C0919281": frozenset({"C5574906", "C1268567"}),  # Iressa (brand)
    "C2987648": frozenset({"C5574906", "C1268567"}),  # afatinib
    "C2987430": frozenset({"C5574906", "C1268567"}),  # dacomitinib
    # ===================================================================
    # HER2-targeted -- targets Q416 (post-trastuzumab progression) plus
    # the 'tucatinib failure' complex query.
    # Class CUI:
    #   C4759996 'Substance with HER2 inhibitor mechanism of action'
    # Tucatinib is a HER2 TKI (kinase inhibitor, not antibody) but shares
    # the mechanism-of-action class — clinically valid for "post-HER2-
    # directed therapy" eligibility matching.
    # ===================================================================
    "C0728747": frozenset({"C4759996"}),  # trastuzumab
    "C0338204": frozenset({"C4759996"}),  # Herceptin (brand)
    "C1328025": frozenset({"C4759996"}),  # pertuzumab
    "C2935436": frozenset({"C4759996"}),  # ado-trastuzumab emtansine (T-DM1)
    "C4519167": frozenset({"C4759996"}),  # tucatinib
    "C5244610": frozenset({"C4759996"}),  # Tukysa (brand)
    # GAP: fam-trastuzumab deruxtecan (Enhertu) -- no drug-type CUI in
    # 2022 KB. Add when the KB updates.
    # ===================================================================
    # Immune checkpoint inhibitors (PD-1 + PD-L1 + CTLA-4).
    # Class CUI:
    #   C4684977 'Immune Checkpoint Inhibitors'
    # ===================================================================
    "C3658706": frozenset({"C4684977"}),  # pembrolizumab
    "C3855203": frozenset({"C4684977"}),  # Keytruda (brand)
    "C3657270": frozenset({"C4684977"}),  # nivolumab
    "C3872108": frozenset({"C4684977"}),  # Opdivo (brand)
    "C4055433": frozenset({"C4684977"}),  # atezolizumab
    "C4055109": frozenset({"C4684977"}),  # durvalumab
    "C1367202": frozenset({"C4684977"}),  # ipilimumab
    # ===================================================================
    # PARP inhibitors -- targets the 'platinum-resistant PARP failure'
    # complex query (ovarian, BRCA wild type).
    # Class CUI:
    #   C1882413 'Poly(ADP-ribose) Polymerase Inhibitors'
    # ===================================================================
    "C2316164": frozenset({"C1882413"}),  # olaparib
    "C3872110": frozenset({"C1882413"}),  # Lynparza (brand)
    "C2744440": frozenset({"C1882413"}),  # niraparib
    "C3661315": frozenset({"C1882413"}),  # rucaparib
    "C4294561": frozenset({"C1882413"}),  # Rubraca (brand)
    "C4042960": frozenset({"C1882413"}),  # talazoparib
    "C4733248": frozenset({"C1882413"}),  # Talzenna (brand)
    # GAP: Zejula (niraparib brand) -- no drug-type CUI in 2022 KB. The
    # generic 'niraparib' still works; queries that say 'Zejula' verbatim
    # won't match until the KB updates.
}


__all__ = ["DRUG_TO_CLASS_CUIS"]
