# TrialMine Ablation Evaluation Report

## Ablation Table

| Method                            |    NDCG@5     |    NDCG@10    |      MRR      | Median Latency |
|-----------------------------------|---------------|---------------|---------------|----------------|
| BM25 only                         |  0.274±0.11   |  0.292±0.10   |  0.454±0.17   |      21ms      |
| Semantic only                     |  0.241±0.09   |  0.240±0.08   |  0.517±0.19   |      32ms      |
| Hybrid (BM25 + Semantic)          |  0.372±0.11   |  0.363±0.09   |  0.531±0.15   |      71ms      |
| + Cross-Encoder Re-ranking        |  0.417±0.11   |  0.426±0.10   |  0.597±0.15   |     5609ms     |
| + Metadata Blender                |  0.450±0.11   |  0.526±0.11   |  0.612±0.16   |     5806ms     |

*Bootstrap 95% confidence intervals from 1000 samples over 20 queries.*

## Methodology

- **Labels**: 990 graded relevance labels (0-3 scale) from Claude Haiku, pooled from both embedding models
- **Queries**: 20 diverse oncology queries (clinical, patient-language, misspelled)
- **Bootstrap CI**: 95% confidence intervals from 1000 bootstrap resamples of query-level metrics
- **Latency**: Median per-query wall-clock time including all pipeline stages
- **NDCG**: Uses graded relevance (0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant)

## Per-Method Details

### BM25 Only
Elasticsearch multi-match with field boosting (title 3x, conditions 2x).

### Semantic Only
Fine-tuned BioLinkBERT bi-encoder + FAISS IndexFlatIP (cosine similarity).

### Hybrid (BM25 + Semantic)
Reciprocal Rank Fusion (k=60) combining 200 candidates from each retriever.

### + Cross-Encoder Re-ranking
Fine-tuned BioLinkBERT cross-encoder scores top-50 hybrid candidates.
Blended scoring: 0.7 * RRF_normalized + 0.3 * CE_sigmoid.

### + Metadata Blender
LightGBM LambdaRank re-ranks using retrieval scores + metadata features
(phase, status, enrollment, condition match, title overlap, eligibility).
