## Phase C3 — Agent-arm A/B comparison (offline)

Production NDCG is shown two ways: ``strict`` (agent failures score 0 — honest cost of failure) and ``with_fallback`` (failures fall back to rule, matching the A5 retry-as-rule design once the agent's degraded-result bubble bug is fixed).

### Headline
- **Queries**: 85 (routed to agent: 40, to rule: 45)
- **Routing rate (heuristic)**: 47.1%
  (Production with the 50/50 AB router: 23.5%)
- **Agent success rate** (of agent-routed): 57.5%
- **Rule-only baseline NDCG@5**: 0.792
- **Production NDCG@5 (strict)**: 0.725 [0.636, 0.805] → Δ = -0.067 [-0.137, -0.007]
- **Production NDCG@5 (with rule fallback)**: 0.840 [0.791, 0.883] → Δ = +0.048 [+0.024, +0.076]
- **Mean cost/query**: $0.0195  (rule-only baseline: $0.0017)
- **p50 / p95 latency**: 6000 ms / 60000 ms

### Per-category

| Category | n | Routed to agent | Agent success | Rule NDCG@5 | Agent NDCG@5 (where ran) | Prod NDCG@5 (strict) | Prod NDCG@5 (fallback) | Δ strict vs rule | Δ fallback vs rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| common | 5 | 1 | 100.0% | 0.941 | 1.000 | 0.958 | 0.958 | +0.017 | +0.017 |
| complex | 15 | 14 | 50.0% | 0.526 | 0.865 | 0.464 | 0.704 | -0.062 | +0.177 |
| existing | 30 | 3 | 100.0% | 0.919 | 1.000 | 0.925 | 0.925 | +0.005 | +0.005 |
| geographic | 3 | 1 | 0.0% | 0.796 | 0.000 | 0.462 | 0.796 | -0.333 | +0.000 |
| pediatric | 3 | 1 | 100.0% | 0.789 | 1.000 | 0.789 | 0.789 | +0.000 | +0.000 |
| rare | 5 | 4 | 100.0% | 0.974 | 1.000 | 1.000 | 1.000 | +0.026 | +0.026 |
| rare_explicit | 5 | 4 | 100.0% | 0.747 | 0.883 | 0.892 | 0.892 | +0.145 | +0.145 |
| treatment | 4 | 0 | 0.0% | 0.933 | 0.000 | 0.933 | 0.933 | +0.000 | +0.000 |
| vague | 15 | 12 | 25.0% | 0.673 | 0.925 | 0.347 | 0.692 | -0.326 | +0.019 |

### Cost + latency per category

| Category | Mean cost / query | p50 latency | p95 latency |
|---|---:|---:|---:|
| common | $0.0070 | 6000 ms | 27821 ms |
| complex | $0.0433 | 56007 ms | 60000 ms |
| existing | $0.0044 | 6000 ms | 25669 ms |
| geographic | $0.0134 | 6000 ms | 54600 ms |
| pediatric | $0.0159 | 6000 ms | 37565 ms |
| rare | $0.0255 | 21784 ms | 36702 ms |
| rare_explicit | $0.0343 | 36136 ms | 40888 ms |
| treatment | $0.0017 | 6000 ms | 6000 ms |
| vague | $0.0298 | 60000 ms | 60000 ms |
