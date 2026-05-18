# Phase C3b — Before / after comparison

Side-by-side of C3's pre-fix A/B output (`agent_vs_rule_v1.json`) vs C3b.5's post-fix re-eval (`agent_vs_rule_v2.json`).

### Headline before / after

| Signal | C3 (v1) | C3b (v2) | Δ | Verdict |
|---|---:|---:|---:|:---:|
| Agent success rate | 57.5% | 100.0% | +42.5 pts | ✅ ≥80% |
| Strict NDCG@5 | 0.725 | 0.839 | +0.114 | ✅ ≥ rule (0.79) |
| Fallback NDCG@5 | 0.840 | 0.839 | -0.001 | ✅ lift |
| Mean cost / query | $0.0195 | $0.0066 | -12.84 m¢ | ✅ ≤ $0.012 |
| p95 latency | 60000 ms | 32829 ms | -27171 ms | ⚠ > 30s |
| Routing rate | 47.1% | 42.4% | -4.7 pts | — |

### Per-category fallback NDCG@5 — v1 vs v2

| Category | n | v1 fallback NDCG@5 | v2 fallback NDCG@5 | Δ | v2 agent success |
|---|---:|---:|---:|---:|---:|
| common | 5 | 0.958 | 0.941 | -0.017 | 0.0% |
| complex | 15 | 0.704 | 0.752 | +0.048 | 100.0% |
| existing | 30 | 0.925 | 0.926 | +0.001 | 100.0% |
| geographic | 3 | 0.796 | 0.796 | +0.000 | 0.0% |
| pediatric | 3 | 0.789 | 0.789 | +0.000 | 100.0% |
| rare | 5 | 1.000 | 0.985 | -0.015 | 100.0% |
| rare_explicit | 5 | 0.892 | 0.836 | -0.056 | 100.0% |
| treatment | 4 | 0.933 | 0.933 | +0.000 | 0.0% |
| vague | 15 | 0.692 | 0.663 | -0.029 | 100.0% |

### Cost per successful agent query

- C3 (v1, no caching): $0.0416/query
- C3b (v2, with caching): $0.0134/query
- Savings: 67.8% per successful agent query

### Agent recovery / regression breakdown

- **Recovered** (failed v1, succeeded v2): 18 qids → 17, 413, 415, 417, 420, 421, 424, 425, 602, 603, 604, 606, 610, 611, 613, 615, 616, 619
- **Regressed** (succeeded v1, failed v2): 0 qids → none
- **Still failing** (no agent submit in either run): 0 qids → 

### C4 decision-gate input — 6 criteria

| Criterion | C2/C3 result | C3b.5 result | Target | Verdict |
|---|---:|---:|---:|:---:|
| Agent success rate | 57.5% | 100.0% | ≥ 80% | ✅ |
| Strict NDCG@5 | 0.725 | 0.839 | ≥ 0.792 | ✅ |
| Fallback NDCG@5 | 0.840 | 0.839 | ≥ 0.842 | ⚠ |
| Complex fallback lift | +0.177 | +0.226 | ≥ +0.10 | ✅ |
| Mean cost / query | $0.0195 | $0.0066 | ≤ $0.012 | ✅ |
| p95 latency | 60000 ms | 32829 ms | ≤ 30,000 ms | ⚠ |

**Pass rate: 4 / 6**

→ **C4 verdict: TUNE** (one more iteration)