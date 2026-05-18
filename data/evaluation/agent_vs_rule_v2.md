## Phase C3 — Agent-arm A/B comparison (offline)

Production NDCG is shown two ways: ``strict`` (agent failures score 0 — honest cost of failure) and ``with_fallback`` (failures fall back to rule, matching the A5 retry-as-rule design once the agent's degraded-result bubble bug is fixed).

### Headline
- **Queries**: 85 (routed to agent: 36, to rule: 49)
- **Routing rate (heuristic)**: 42.4%
  (Production with the 50/50 AB router: 21.2%)
- **Agent success rate** (of agent-routed): 100.0%
- **Rule-only baseline NDCG@5**: 0.792
- **Production NDCG@5 (strict)**: 0.839 [0.790, 0.883] → Δ = +0.046 [+0.009, +0.079]
- **Production NDCG@5 (with rule fallback)**: 0.839 [0.790, 0.883] → Δ = +0.046 [+0.009, +0.079]
- **Mean cost/query**: $0.0066  (rule-only baseline: $0.0017)
- **p50 / p95 latency**: 6000 ms / 32829 ms

### Per-category

| Category | n | Routed to agent | Agent success | Rule NDCG@5 | Agent NDCG@5 (where ran) | Prod NDCG@5 (strict) | Prod NDCG@5 (fallback) | Δ strict vs rule | Δ fallback vs rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| common | 5 | 0 | 0.0% | 0.941 | 0.000 | 0.941 | 0.941 | +0.000 | +0.000 |
| complex | 15 | 14 | 100.0% | 0.526 | 0.741 | 0.752 | 0.752 | +0.226 | +0.226 |
| existing | 30 | 4 | 100.0% | 0.919 | 0.972 | 0.926 | 0.926 | +0.006 | +0.006 |
| geographic | 3 | 0 | 0.0% | 0.796 | 0.000 | 0.796 | 0.796 | +0.000 | +0.000 |
| pediatric | 3 | 1 | 100.0% | 0.789 | 1.000 | 0.789 | 0.789 | +0.000 | +0.000 |
| rare | 5 | 4 | 100.0% | 0.974 | 0.981 | 0.985 | 0.985 | +0.011 | +0.011 |
| rare_explicit | 5 | 3 | 100.0% | 0.747 | 0.898 | 0.836 | 0.836 | +0.089 | +0.089 |
| treatment | 4 | 0 | 0.0% | 0.933 | 0.000 | 0.933 | 0.933 | +0.000 | +0.000 |
| vague | 15 | 10 | 100.0% | 0.673 | 0.579 | 0.663 | 0.663 | -0.009 | -0.009 |

### Cost + latency per category

| Category | Mean cost / query | p50 latency | p95 latency |
|---|---:|---:|---:|
| common | $0.0017 | 6000 ms | 6000 ms |
| complex | $0.0130 | 30979 ms | 38688 ms |
| existing | $0.0045 | 6000 ms | 20625 ms |
| geographic | $0.0017 | 6000 ms | 6000 ms |
| pediatric | $0.0131 | 6000 ms | 25937 ms |
| rare | $0.0138 | 18548 ms | 23494 ms |
| rare_explicit | $0.0169 | 21099 ms | 22951 ms |
| treatment | $0.0017 | 6000 ms | 6000 ms |
| vague | $0.0015 | 9937 ms | 18007 ms |
