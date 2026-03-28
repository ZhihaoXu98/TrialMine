# TrialMine Fair Ablation Evaluation Report


## Fair Ablation Table (50 held-out test queries)


| Method                            |    NDCG@5     |    NDCG@10    |      MRR      |  Latency   |
|-----------------------------------|---------------|---------------|---------------|------------|
| BM25 only                         | 0.617±0.09  | 0.614±0.08  | 0.768±0.10  |     21ms   |
| Semantic only                     | 0.606±0.07  | 0.603±0.06  | 0.815±0.08  |     36ms   |
| Hybrid (BM25 + Semantic)          | 0.636±0.08  | 0.644±0.06  | 0.807±0.09  |     71ms   |
| + Cross-Encoder Re-ranking        | 0.651±0.08  | 0.657±0.06  | 0.825±0.09  |   6166ms   |
| + Metadata Blender                | 0.670±0.08  | 0.657±0.07  | 0.806±0.09  |   6134ms   |

*Bootstrap 95% CI from 1000 samples over 50 held-out queries.*

*LightGBM trained on 150 queries — true held-out test.*


## Key Differences from Original Evaluation

1. **Held-out queries**: 50 new queries never seen during LightGBM training (150 training queries)
2. **Pooled labeling**: Labels from BM25 + Semantic + Hybrid union (no method bias)
3. **Expanded training**: LightGBM trained on 150 queries (~6,200 labeled pairs) vs original 20
