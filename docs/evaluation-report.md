# TrialMine Ablation Evaluation Report

## Ablation Table

| Method                            |    NDCG@5     |    NDCG@10    |      MRR      | Median Latency |
|-----------------------------------|---------------|---------------|---------------|----------------|
| BM25 only                         |  0.789±0.12   |  0.756±0.13   |  0.912±0.09   |      22ms      |
| Semantic only                     |  0.703±0.12   |  0.700±0.10   |  0.807±0.12   |      37ms      |
| Hybrid (BM25 + Semantic)          |  0.816±0.10   |  0.796±0.08   |  0.917±0.08   |      72ms      |
| + Cross-Encoder Re-ranking        |  0.829±0.09   |  0.817±0.07   |  0.950±0.06   |     6295ms     |
| + Metadata Blender                |  0.980±0.02   |  0.921±0.03   |  1.000±0.00   |     6472ms     |

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

## Caveats

**Overfitting warning on blender results:** The LightGBM model was trained on all 20 queries
and evaluated on those same 20 queries. The ablation numbers for "+ Metadata Blender" are
optimistic. The honest number is the leave-one-query-out CV result: **NDCG@5=0.844, NDCG@10=0.840**.
This is still a significant improvement over hybrid-only (0.816/0.796) but not as dramatic
as the ablation table suggests (0.980/0.921).

**20 queries is thin.** One query swings the average by 5%. The wide bootstrap CIs reflect
this — e.g., BM25 NDCG@5 = 0.789 +/- 0.12 spans from 0.67 to 0.91. Results are directional,
not conclusive.

**Label source.** All labels are from Claude Haiku (LLM-as-judge), not human annotators.
No inter-annotator agreement or human calibration has been performed.

## Feature Importance

Top features by LightGBM gain:
1. **rrf_score** (393) — hybrid retrieval score is the strongest signal
2. **cross_encoder_score** (243) — CE adds value as a feature despite failing as standalone ranker
3. **bm25_score** (132) — raw BM25 score adds signal beyond rank
4. **enrollment_log** (97) — larger trials tend to be more relevant
5. **semantic_score** (93) — cosine similarity adds complementary signal
6. **phase_numeric** (57) — Phase 3 trials preferred by patients
7. **title_query_overlap** (45) — simple lexical match is still useful
8. **is_recruiting** (39) — active enrollment status matters
