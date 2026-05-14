# Building TrialMine: An ML-Powered Clinical Trial Search Engine for Oncology

*A complete walkthrough of every step, every function, and every decision — written for learning.*

---

## Table of Contents

1. [Introduction: What Are We Building and Why?](#1-introduction-what-are-we-building-and-why)
2. [Foundations of Machine Learning for Search](#2-foundations-of-machine-learning-for-search)
3. [How We Measure Search Quality (Read This First)](#3-how-we-measure-search-quality-read-this-first)
4. [Data Pipeline: Getting 140,000 Clinical Trials](#4-data-pipeline-getting-140000-clinical-trials)
5. [Stage 1: BM25 Keyword Search with Elasticsearch](#5-stage-1-bm25-keyword-search-with-elasticsearch)
6. [Stage 2: Semantic Search — Embeddings, FAISS, and the Anisotropy Problem](#6-stage-2-semantic-search--embeddings-faiss-and-the-anisotropy-problem)
7. [Stage 3: Hybrid Search — Combining BM25 and Semantic with RRF](#7-stage-3-hybrid-search--combining-bm25-and-semantic-with-rrf)
8. [Training Data: Teaching the Model to Search](#8-training-data-teaching-the-model-to-search)
9. [Fine-Tuning the Bi-Encoder](#9-fine-tuning-the-bi-encoder)
10. [Evaluation: LLM-as-Judge, Bias Discovery, and the Pooling Fix](#10-evaluation-llm-as-judge-bias-discovery-and-the-pooling-fix)
11. [Stage 4: Cross-Encoder Re-Ranking](#11-stage-4-cross-encoder-re-ranking)
12. [Stage 5: LightGBM Metadata Blender](#12-stage-5-lightgbm-metadata-blender)
13. [Fair Evaluation: Testing on Queries We've Never Seen](#13-fair-evaluation-testing-on-queries-weve-never-seen)
14. [The Full Pipeline: How Everything Connects](#14-the-full-pipeline-how-everything-connects)
15. [What's Planned But Not Yet Built](#15-whats-planned-but-not-yet-built)
16. [Current Problems: What's Wrong and Why](#16-current-problems-whats-wrong-and-why)
17. [What We Can Do Next](#17-what-we-can-do-next)
18. [Lessons Learned](#18-lessons-learned)
19. [MLE Interview Guide: Questions This Project Prepares You For](#19-mle-interview-guide-questions-this-project-prepares-you-for)

---

## 1. Introduction: What Are We Building and Why?

### The Problem

There are over 140,000 oncology (cancer) clinical trials registered on ClinicalTrials.gov. A cancer patient looking for a trial faces a massive needle-in-a-haystack problem. They might type something like:

> "I have stage 3 breast cancer and the chemo stopped working, what trials can I join?"

But the clinical trials on ClinicalTrials.gov use clinical language:

> "A Phase 3 Study of Pembrolizumab in Participants With Triple-Negative Breast Cancer Who Have Failed Prior Anthracycline-Based Chemotherapy"

This is called the **vocabulary mismatch problem**. The patient says "chemo stopped working" — the trial says "failed prior anthracycline-based chemotherapy." A simple keyword search would not connect these two, even though they mean the same thing.

### What Is Information Retrieval?

**Information retrieval (IR)** is the task of finding relevant documents from a large collection given a user's query. Google Search is the most famous IR system. Our system is a specialized IR system: given a patient's description of their situation, find the clinical trials that are most relevant to them.

The core challenge of IR is **ranking**: not just finding relevant documents, but putting the most relevant ones at the top of the list. A system that returns 100 relevant trials buried among 1000 irrelevant ones is useless. A system that puts the 5 most relevant trials at positions 1-5 is excellent.

### Our 5-Stage Pipeline

We solve this by building a **multi-stage retrieval pipeline** where each stage addresses a different weakness:

| Stage | What It Does | Why We Need It |
|-------|-------------|----------------|
| 1. BM25 | Keyword matching (like Ctrl+F on steroids) | Fast, handles exact terms like drug names |
| 2. Semantic | Meaning matching (understands "chemo stopped working" ≈ "failed chemotherapy") | Bridges vocabulary gap |
| 3. Hybrid (RRF) | Combines BM25 and semantic results | Each method finds different relevant trials |
| 4. Cross-Encoder | A more powerful AI model re-scores the top candidates | Catches errors the first two stages missed |
| 5. LightGBM | Uses trial metadata (phase, recruiting status, enrollment) to finalize ranking | Text relevance alone isn't enough — a recruiting Phase 3 trial is more useful than a completed Phase 1 |

**Key result:** On 50 held-out test queries we never trained on, this pipeline achieves NDCG@5 = 0.670, compared to 0.617 for BM25 alone. Every stage makes the results a little bit better.

---

## 2. Foundations of Machine Learning for Search

This section teaches the core ML theory that underpins every component of the pipeline. If you're preparing for an MLE interview, these are the concepts you'll be asked about directly. Every subsection here connects to a specific part of our system — we'll point out those connections as we go.

### How Neural Networks Learn: Gradient Descent

A neural network is a mathematical function with millions of adjustable numbers called **parameters** (or **weights**). Our BioLinkBERT model has 110 million parameters. The model's behavior is entirely determined by these values. Training means finding the values that make the model produce the outputs we want.

**The training loop has three steps, repeated thousands of times:**

**Step 1: Forward pass.** Feed an input (e.g., a query-trial pair) through the network. The network produces an output (e.g., a relevance score of 0.3).

**Step 2: Compute loss.** Compare the output to the correct answer (e.g., the true label is 1.0). The **loss function** quantifies how wrong the prediction is. If we're using Mean Squared Error: `loss = (0.3 - 1.0)² = 0.49`.

**Step 3: Backward pass (backpropagation).** Calculate how much each parameter contributed to the error, then adjust each parameter to reduce the loss.

**What is a gradient?** A gradient tells you the direction and magnitude of steepest increase for a function. If you're standing on a hill and want to go downhill (minimize loss), you walk in the direction opposite to the gradient. For each parameter, the gradient says "if you increase this parameter by a tiny amount, the loss changes by X." To reduce the loss, we adjust the parameter in the direction that decreases the loss.

**The update rule:**
```
new_parameter = old_parameter - learning_rate × gradient
```

The **learning rate** controls the step size:
- Too large (e.g., 0.1): parameters overshoot the minimum, training is unstable, loss oscillates wildly
- Too small (e.g., 1e-8): training takes forever, might get stuck in a poor local minimum
- Just right (e.g., 2e-5 for fine-tuning BERT): converges steadily to a good solution

**Why 2e-5 specifically?** Pre-trained models like BERT already have useful parameter values. A large learning rate would destroy what the model learned during pre-training (called "catastrophic forgetting"). A small learning rate like 2e-5 makes gentle adjustments — fine-tuning rather than retraining.

**Stochastic Gradient Descent (SGD) and mini-batches.** Computing the gradient over the entire dataset (586K triplets) for every parameter update is expensive. Instead, we compute it on a small **mini-batch** (e.g., 32 triplets), update the parameters, and repeat. This is noisier but much faster, AND the noise actually helps escape bad local minima. Our bi-encoder uses `batch_size=32`, meaning each parameter update is based on 32 triplets.

**Concrete example from our pipeline:**
```
Epoch 1 (586K triplets / 32 per batch = 18,312 steps):
  Step 1:    Forward 32 triplets → Loss = 4.2 → Backward → Update 110M params
  Step 2:    Forward 32 triplets → Loss = 4.1 → Backward → Update
  ...
  Step 18312: Forward 32 triplets → Loss = 1.5 → Backward → Update

Epoch 2: Same 586K triplets, reshuffled → Loss drops from 1.5 to 1.2
Epoch 3: Same again → Loss drops from 1.2 to 1.1 → Training complete
```

**Backpropagation** is the algorithm that computes gradients efficiently. A neural network is a chain of operations: input → layer 1 → layer 2 → ... → layer 12 → output. The **chain rule** from calculus lets us compute ∂loss/∂parameter for every parameter by working backward from the output. In our BERT model, backpropagation computes 110 million gradients in one pass — this is what makes training feasible.

**Learning rate warmup.** Our config uses `warmup_ratio=0.1`, meaning the learning rate starts at 0 and linearly increases to 2e-5 over the first 10% of training steps. Why? At the start of fine-tuning, the model's representations are from pre-training. Large initial updates could destroy useful pre-trained features. The warmup lets the model gently transition to the new task.

**Adam and AdamW optimizers.** We don't use plain SGD — we use **AdamW**, which maintains per-parameter adaptive learning rates and momentum. Adam tracks two running averages for each parameter: (1) the mean gradient (momentum — keeps moving in a consistent direction) and (2) the squared gradient (scales down learning rate for parameters with large gradients). "W" stands for **weight decay** — a regularization technique that slightly shrinks all parameters toward zero each step, preventing any single parameter from growing too large.

### The Transformer Architecture and BERT

Every neural model in our pipeline (bi-encoder, cross-encoder) is based on **BERT**, which uses the **Transformer** architecture. Understanding this is essential for MLE interviews.

**The problem Transformers solved.** Before Transformers (2017), the dominant approach for processing text was **Recurrent Neural Networks (RNNs)**, which read text one word at a time, left to right. This had two problems: (1) **slow** — you can't parallelize sequential reading, and (2) **forgets** — by the time the RNN reaches word 500, it has largely forgotten word 1. Long-range dependencies (e.g., connecting "patient" at the start to "eligibility" at the end of a paragraph) were lost.

**The key innovation: self-attention.** Instead of reading sequentially, Transformers process all words simultaneously and learn which words to "pay attention to" for each word.

For the sentence "The patient has EGFR-mutated lung cancer":
- When processing "EGFR-mutated": pay attention to "lung" and "cancer" (they form a meaningful unit)
- When processing "cancer": pay attention to "lung" (lung cancer, not brain cancer) and "EGFR-mutated" (specific subtype)
- When processing "patient": pay attention to "has" and "cancer" (what the patient has)

**How self-attention works:**

For each word, the model creates three vectors from the word's embedding:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What information do I contain?"
- **Value (V):** "What should I pass along if someone attends to me?"

The attention score between word_i and word_j:
```
attention(i, j) = softmax( Q_i · K_j / sqrt(d) )
```

Where `d = 768` (the embedding dimension). The `sqrt(d)` scaling prevents dot products from growing too large, which would make softmax output near-0/near-1 values and kill gradient flow.

The output for word_i is a weighted sum of all Value vectors:
```
output_i = Σ_j  attention(i, j) × V_j
```

Words with high attention scores contribute more. This lets the model dynamically decide, for each word, which other words are important for understanding it.

**Multi-head attention.** Instead of one set of Q/K/V vectors, BERT uses **12 parallel attention heads**. Each head learns to attend to different linguistic aspects:
- Head 1 might learn syntactic relationships ("cancer" ← modified by "lung")
- Head 3 might learn semantic type matching ("pembrolizumab" is a drug → connects to "treatment")
- Head 7 might learn coreference ("it" refers to "the trial")

The 12 heads' outputs are concatenated and projected back to 768 dimensions. This gives the model 12 different "perspectives" on the relationships between words.

**Worked numerical example of attention.** Interviewers frequently ask "walk me through self-attention with actual numbers." Here's a minimal example using 4-dimensional embeddings (real BERT uses 768, but the math is identical).

Input: three words — "lung", "cancer", "treatment" — with initial embeddings:
```
lung    = [1.0, 0.0, 1.0, 0.0]
cancer  = [0.0, 1.0, 1.0, 0.0]
treatment = [0.5, 0.5, 0.0, 1.0]
```

**Step 1: Compute Q, K, V** by multiplying each embedding by learned weight matrices. For simplicity, say the learned W_Q, W_K, W_V are identity matrices with slight adjustments:
```
Q_lung = [1.0, 0.0, 1.0, 0.0]    K_lung = [1.0, 0.1, 0.9, 0.0]    V_lung = [0.9, 0.0, 1.0, 0.1]
Q_cancer = [0.0, 1.0, 1.0, 0.0]  K_cancer = [0.1, 1.0, 1.0, 0.0]  V_cancer = [0.0, 0.9, 1.0, 0.1]
Q_treatment = [0.5, 0.5, 0.0, 1.0] K_treatment = [0.5, 0.5, 0.1, 0.9] V_treatment = [0.5, 0.5, 0.0, 0.9]
```

**Step 2: Compute attention scores** via dot products Q_i · K_j, then scale by `sqrt(d) = sqrt(4) = 2`:

For "lung" attending to each word (Q_lung · K_j / 2):
```
lung → lung:      (1.0×1.0 + 0.0×0.1 + 1.0×0.9 + 0.0×0.0) / 2 = 1.9/2 = 0.95
lung → cancer:    (1.0×0.1 + 0.0×1.0 + 1.0×1.0 + 0.0×0.0) / 2 = 1.1/2 = 0.55
lung → treatment: (1.0×0.5 + 0.0×0.5 + 1.0×0.1 + 0.0×0.9) / 2 = 0.6/2 = 0.30
```

**Step 3: Softmax** to convert to probabilities (must sum to 1):
```
exp(0.95) = 2.586    exp(0.55) = 1.733    exp(0.30) = 1.350
sum = 5.669

attention weights for "lung":
  → lung: 2.586/5.669 = 0.456
  → cancer: 1.733/5.669 = 0.306
  → treatment: 1.350/5.669 = 0.238
```

"Lung" pays most attention to itself (0.456) and to "cancer" (0.306) — this is the model learning that "lung" and "cancer" form a meaningful unit. "Treatment" gets less attention (0.238) because it's less directly related to "lung."

**Step 4: Weighted sum** of Value vectors using attention weights:
```
output_lung = 0.456 × V_lung + 0.306 × V_cancer + 0.238 × V_treatment
            = 0.456 × [0.9, 0.0, 1.0, 0.1]
            + 0.306 × [0.0, 0.9, 1.0, 0.1]
            + 0.238 × [0.5, 0.5, 0.0, 0.9]
            = [0.410, 0.000, 0.456, 0.046]
            + [0.000, 0.275, 0.306, 0.031]
            + [0.119, 0.119, 0.000, 0.214]
            = [0.529, 0.394, 0.762, 0.290]
```

Notice: the output for "lung" is no longer just about "lung." It now contains information from "cancer" — dimension 2 went from 0.0 (lung's original V value) to 0.394, with V_cancer contributing 0.275 of that (70%). After this attention step, the representation of "lung" *knows about* "cancer" — it's been contextualized. This is the core mechanism that lets BERT understand that "lung cancer" is a compound concept, not two independent words.

In real BERT, this happens with 768-dimensional vectors, 12 heads in parallel, and learned W_Q/W_K/W_V matrices with millions of parameters. The math is identical — just higher-dimensional.

**BERT's full architecture:**
```
Input: "breast cancer immunotherapy"
  |
  v
Tokenization: [CLS] breast cancer immuno ##therapy [SEP]
  |
  v
Token embeddings + Position embeddings → 6 vectors of 768 dims
  |
  v
+-- Transformer Layer 1 ---------------------+
|  Multi-head self-attention (12 heads)       |
|  Add & Layer Normalization                  |
|  Feed-forward network (768 → 3072 → 768)   |
|  Add & Layer Normalization                  |
+---------------------------------------------+
  |
  v
+-- Transformer Layer 2 (same structure) ----+
  ...
+-- Transformer Layer 12 --------------------+
  |
  v
6 contextualized vectors of 768 dimensions
```

After 12 layers of self-attention, each word's vector captures its meaning **in context**. The word "bank" gets different vectors in "river bank" vs "bank account" because the attention layers incorporate surrounding words. This is called a **contextual embedding** — unlike static word vectors (Word2Vec), the same word gets different representations in different contexts.

**Position embeddings: teaching word order.** Self-attention is fundamentally **order-agnostic**. Look at the attention formula: `attention(i,j) = softmax(Q_i · K_j / sqrt(d))`. It computes a dot product between vectors — there's nothing in this formula that knows word *i* comes before word *j*. The sentence "patient has cancer" and "cancer has patient" would produce identical attention scores if we only used token embeddings.

BERT solves this with **learned position embeddings**. Each position in the sequence (position 0, 1, 2, ..., up to 512) has its own 768-dimensional learned vector. Before the first Transformer layer, BERT adds these to the token embeddings:

```
final_embedding[i] = token_embedding[word_i] + position_embedding[i] + segment_embedding
```

For "The patient has cancer":
```
Position 0: embedding("The")     + position_vec_0  → input to layer 1
Position 1: embedding("patient") + position_vec_1  → input to layer 1
Position 2: embedding("has")     + position_vec_2  → input to layer 1
Position 3: embedding("cancer")  + position_vec_3  → input to layer 1
```

These position vectors are learned during pre-training — the model discovers that certain positions have certain roles. Position 0 is always [CLS], position 1 is typically the first content word, and the model learns positional patterns like "early positions often hold the main subject." The addition means the attention computation now sees different Q/K vectors for the same word at different positions, breaking the order-agnostic symmetry.

Why addition instead of concatenation? Concatenation would double the embedding dimension (768 token + 768 position = 1536), increasing all downstream parameters. Addition keeps the dimension at 768. It works because the model learns token and position embeddings in a shared space where addition combines their information.

The **segment embedding** is a binary vector indicating "sentence A" or "sentence B" — used during pre-training for NSP, and by our cross-encoder to distinguish query from trial text.

**Residual connections and layer normalization.** The architecture diagram shows "Add & Layer Normalization" after each sub-layer. These are critical for training deep networks, and interviewers ask about both.

**Residual connections** (the "Add" part): Each sub-layer's output is added to its input:
```
sublayer_output = MultiHeadAttention(x)
residual_output = x + sublayer_output     ← this is the residual connection
```

Why? In a 12-layer network, gradients from the loss must travel backward through all 12 layers during backpropagation. Without residual connections, each layer multiplies the gradient by its weight matrix. After 12 multiplications, if any weight matrix has eigenvalues < 1, the gradient shrinks exponentially — this is the **vanishing gradient problem**. The lower layers receive near-zero gradients and stop learning.

The residual `+ x` creates a **gradient highway**: a direct additive path from the loss all the way back to layer 1. Even if the sublayer's gradient vanishes, the gradient through the `+ x` path is exactly 1.0 (the derivative of `x + f(x)` with respect to `x` includes a constant term of 1). This ensures every layer receives meaningful gradient signal.

Intuition: the residual connection lets each layer learn a *correction* to its input rather than a complete transformation. The model starts by passing information through unchanged (the identity path) and gradually learns what adjustments each layer should make. This is much easier to optimize than learning 12 complete transformations from scratch.

**Layer normalization** (the "Normalization" part): After each residual addition, layer norm normalizes the values across the 768 dimensions:
```
LayerNorm(x) = γ × (x - mean(x)) / sqrt(var(x) + ε) + β
```

Where γ (scale) and β (shift) are learnable parameters, and ε (usually 1e-12) prevents division by zero. This serves two purposes:
1. **Stabilizes activation magnitudes.** Without normalization, values can grow exponentially through 12 layers (each attention and FFN layer can amplify). Layer norm resets the scale at each layer.
2. **Smooths the loss landscape.** Normalized inputs to each layer have consistent statistics, making gradient descent more stable and enabling higher learning rates.

The complete computation flow through one Transformer layer:
```
x                              ← input (768-dim per token)
├→ Multi-Head Attention(x) → a
├→ a + x                      ← residual connection
├→ LayerNorm(a + x) → n       ← layer normalization
├→ FFN(n) → f
├→ f + n                      ← second residual connection
└→ LayerNorm(f + n) → output  ← second layer normalization → next layer
```

**The feed-forward network (FFN): per-token processing.** Each Transformer layer has a 2-layer FFN applied to each token independently:
```
FFN(x) = GELU(x × W₁ + b₁) × W₂ + b₂

Where: W₁ is 768×3072, W₂ is 3072×768
```

This creates a **bottleneck → expansion → bottleneck** pattern: 768 → 3072 → 768. The expansion to 3072 (4× wider) gives the model more capacity for non-linear transformations.

The attention mechanism and the FFN serve complementary roles:
- **Attention** captures relationships *between* tokens — "lung" gathers context from "cancer" and "treatment." It's about *communication* between positions.
- **FFN** processes each token *independently* — it takes the context-enriched representation of "lung" (which now contains information about "cancer") and applies non-linear transformations to it. It's about *computation* within each position.

Think of attention as "gather information from neighbors" and FFN as "think about what you gathered." Without the FFN, the model could only compute linear combinations of other tokens' representations. The FFN adds the non-linear expressiveness needed to learn complex features like "this token represents a drug name that treats the cancer type mentioned earlier."

**GELU activation** (Gaussian Error Linear Unit) is a smooth version of ReLU. ReLU(x) = max(0, x) hard-clips negative values to 0. GELU(x) ≈ x × Φ(x) (where Φ is the standard normal CDF) smoothly suppresses negative values. BERT uses GELU because the smooth gradient helps with optimization — no sudden gradient discontinuity at x=0.

**Parameter count breakdown: where do 110M parameters live?** Understanding parameter counts helps you reason about memory usage, inference speed, and fine-tuning feasibility. Here's where BERT-base's 110M parameters go:

**Embeddings (23.8M parameters):**
```
Token embeddings:    30,522 vocab × 768 dims  = 23.4M
Position embeddings:     512 positions × 768  =  0.4M
Segment embeddings:        2 segments × 768   =  0.002M
Layer norm (embedding):    768 × 2 (γ and β)  =  0.002M
                                        Total ≈ 23.8M
```

**Per Transformer layer (7.1M parameters):**
```
Multi-head attention:
  Q projection: 768 × 768 + 768 bias  = 0.59M
  K projection: 768 × 768 + 768 bias  = 0.59M
  V projection: 768 × 768 + 768 bias  = 0.59M
  Output proj:  768 × 768 + 768 bias  = 0.59M
  Layer norm:   768 × 2               = 0.002M
                            Subtotal  = 2.36M

Feed-forward network:
  W₁: 768 × 3072 + 3072 bias  = 2.36M
  W₂: 3072 × 768 + 768 bias   = 2.36M
  Layer norm: 768 × 2          = 0.002M
                    Subtotal   = 4.72M

Total per layer: 2.36M + 4.72M = 7.08M
```

**All 12 layers:** 12 × 7.08M = **85.0M**

**Pooler layer:** 768 × 768 + 768 = **0.6M** (transforms [CLS] token for classification)

**Grand total:** 23.8M + 85.0M + 0.6M ≈ **109.5M** (rounds to 110M)

**Why this matters for our pipeline:**
- The cross-encoder runs a full 110M-parameter forward pass for EACH (query, trial) pair. With 50 candidates: 50 × 110M multiplications = 5.5 billion operations → ~4 seconds on CPU. This is the bottleneck.
- The bi-encoder runs ONE forward pass per query (110M ops, ~15ms) and finds candidates via FAISS (matrix multiplication, ~22ms). Pre-computed trial embeddings mean 0 forward passes for trials at serving time.
- The FFN dominates parameter count (4.72M/7.08M = 67% per layer). This is why quantizing FFN weights gives the biggest speedup.
- Embedding parameters (23.8M) are shared across all positions and only looked up (not multiplied), so they contribute to memory but not to inference latency.

**How BERT was pre-trained (two tasks on massive text):**

1. **Masked Language Modeling (MLM):** Randomly hide 15% of words, make the model predict them. Given "The patient has [MASK] lung cancer," predict "EGFR-mutated." This forces deep understanding of word relationships — you can't predict the missing word without understanding the sentence.

2. **Next Sentence Prediction (NSP):** Given two sentences, predict whether the second logically follows the first. Teaches inter-sentence relationships.

**BioLinkBERT** extends BERT pre-training to biomedical text from PubMed, with a key innovation: it uses **citation links** between papers to create training pairs. If Paper A cites Paper B, sentences from A and B are used as positive pairs. This means BioLinkBERT learns that "glioblastoma" and "temozolomide" are related (because glioblastoma papers frequently cite temozolomide papers), even though the words share no characters.

**Tokenization: WordPiece.** BERT doesn't process raw words — it uses **WordPiece** tokenization, which breaks rare words into subword pieces. Examples:
- "immunotherapy" → ["immuno", "##therapy"]
- "pembrolizumab" → ["pe", "##mb", "##ro", "##li", "##zu", "##mab"]
- "cancer" → ["cancer"] (common word, kept intact)

The `##` prefix means "continuation of previous token." This is why BERT can handle words it has never seen: "pembrolizumab" gets broken into subwords, and the model recognizes "-mab" as the monoclonal antibody suffix from pre-training on biomedical text. BERT's vocabulary has ~30,000 subword tokens — enough to represent any English word as a sequence of pieces.

### Encoder vs Decoder: Why BERT for Search?

The original 2017 Transformer paper had both an **encoder** and a **decoder**. Since then, three distinct architectures have emerged, each suited to different tasks. Understanding which to use and why is a common interview question.

**Encoder-only (BERT, BioLinkBERT) — what we use:**
```
Input: "lung cancer immunotherapy"
  → Bidirectional self-attention (every token sees every other token)
  → Output: one vector per token (768-dim each)
  → Use the vectors for classification, similarity, or ranking
```

The key property is **bidirectional attention**: when processing "cancer," the model sees both "lung" (to the left) and "immunotherapy" (to the right) simultaneously. This produces rich contextual representations because every token has full context. The output is a set of fixed-size vectors — one per input token — which can be pooled into a single embedding for the entire input.

**Decoder-only (GPT, Claude, LLaMA) — what Claude Haiku is:**
```
Input: "The patient has"
  → Causal (left-to-right) self-attention (each token only sees previous tokens)
  → Output: probability distribution over the NEXT token
  → Sample: "stage" → "3" → "breast" → "cancer" → ...
```

The key property is **causal attention**: when processing "has," the model sees "The" and "patient" but NOT any future tokens. This is enforced by an **attention mask** — a triangular matrix that sets attention scores to -infinity for future positions, making their softmax weights exactly 0. This constraint is necessary for autoregressive generation: the model can't peek at the answer while generating it.

Claude Haiku, which we use as our LLM judge for labeling relevance, is decoder-only. It generates text token by token, left to right. When we prompt it with a rubric and a (query, trial) pair, it generates a relevance score and explanation.

**Encoder-decoder (T5, BART, the original Transformer):**
```
Input (encoder): "Translate: lung cancer immunotherapy"
  → Bidirectional attention over input
  → Output: contextualized representations of the input

Input (decoder): generates output token by token
  → Causal attention over generated tokens
  → Cross-attention to encoder representations
  → Output: "Lungenkrebs-Immuntherapie"
```

The encoder reads the full input bidirectionally, then the decoder generates output one token at a time while attending to the encoder's representations via **cross-attention**. Best for sequence-to-sequence tasks: translation, summarization, question answering with extractive output.

**Why encoder-only for search:**

Our search pipeline needs to do two things: (1) produce a fixed-size embedding for every trial so we can store it in FAISS, and (2) compare query and trial embeddings via dot product to find similar ones.

| Requirement | Encoder | Decoder | Encoder-Decoder |
|---|---|---|---|
| Fixed-size vector per input | Yes (pool token vectors) | No (outputs token probabilities) | Partially (encoder side only) |
| Pre-compute trial embeddings offline | Yes (embed once, store in FAISS) | No (no embeddings to store) | Over-engineered for this |
| Fast similarity comparison | Yes (dot product, ~1ms) | N/A | N/A |
| Understands bidirectional context | Yes | No (left-to-right only) | Yes (encoder side) |

A decoder-only model like GPT *could* be used for search — you'd prompt it with "Rate the relevance of this trial to this query" and have it generate a score. But this requires a full forward pass per (query, trial) pair (like our cross-encoder), making it impractical for 140K candidates. The encoder's ability to produce pre-computable, fixed-size embeddings is what makes sub-100ms retrieval over 140K trials possible.

**The role of each architecture in our pipeline:**

| Component | Architecture | Why |
|---|---|---|
| Bi-encoder (retrieval) | Encoder-only (BioLinkBERT) | Need pre-computed embeddings for FAISS |
| Cross-encoder (re-ranking) | Encoder-only (BioLinkBERT) | Need bidirectional attention over query+trial jointly |
| LLM judge (labeling) | Decoder-only (Claude Haiku) | Need text generation for relevance scores + explanations |
| Query parser (planned) | Decoder-only (Claude via LangGraph) | Need text generation for structured query output |

Notice that the two BERT models (bi-encoder and cross-encoder) use the same encoder architecture but in fundamentally different ways. The bi-encoder processes query and trial *separately* (enabling pre-computation), while the cross-encoder processes them *together* as a single input (enabling richer interaction but requiring per-pair inference). This distinction is explored in detail in Section 11.

### Transfer Learning: Why Pre-train Then Fine-tune?

**Transfer learning** is the idea that knowledge learned on one task can help with a different task. It's one of the most important ideas in modern ML.

**The paradigm:**
```
Phase 1: Pre-training (done by researchers, costs millions of dollars)
  - Train on billions of words of text
  - Model learns: grammar, word meanings, world knowledge, biomedical concepts
  - Result: a general-purpose language model

Phase 2: Fine-tuning (done by us, costs ~$10 on Colab)
  - Take the pre-trained model
  - Train further on our 586K clinical trial triplets
  - Model adjusts: now produces embeddings optimized for trial search
  - Result: a specialized search model
```

**Why it works — the layer hierarchy:**
- **Lower layers** (1-4): learn general features — word boundaries, part-of-speech, basic syntax. These barely change during fine-tuning because language itself doesn't change.
- **Middle layers** (5-8): learn semantic relationships — synonyms, entity types, biomedical concept associations. These adjust moderately.
- **Upper layers** (9-12): learn task-specific features — what makes a trial relevant to a query. These change the most during fine-tuning.

This is why fine-tuning works with "only" 586K examples while pre-training needed billions of words. The lower layers already know language — we only need to teach the upper layers about clinical trial ranking.

**Why not train from scratch?** Training BERT from scratch on our 586K triplets would fail catastrophically. 586K examples is enough to learn "query X matches trial Y" but nowhere near enough to learn "how English works." Pre-training provides the linguistic foundation; fine-tuning provides the task-specific skill.

**Analogy:** Pre-training is like medical school (years of general medical education). Fine-tuning is like a radiology residency (specialized training for a specific task). You wouldn't try to become a radiologist without first going to medical school.

### When to Fine-tune vs When to Prompt

With the rise of powerful language models, there are now two ways to adapt a model to your task: **fine-tuning** (change the model's parameters) or **prompting** (change the input, keep the model frozen). Knowing when to use which is a critical practical skill and a frequent interview question.

**Fine-tuning** changes the model's **behavior**. You feed it training examples and adjust millions of parameters via backpropagation. After fine-tuning, the model inherently "thinks differently" — its internal representations are reshaped.

**Prompting** changes the model's **instructions**. You prepend a detailed prompt (instructions, rubric, examples) to the input and let the frozen model follow it. The model's parameters don't change — you're leveraging capabilities it already has.

**The decision framework — when to use each:**

| Factor | Fine-tune | Prompt |
|---|---|---|
| **What you're changing** | Model's internal representations | Model's instructions |
| **Data needed** | 1K-1M labeled examples | 0-50 examples (in-context) |
| **Cost** | $10-$1000 (GPU compute) | $0.001-$0.10 per inference (API cost) |
| **Latency** | Same as base model | Higher (longer input context) |
| **When it's appropriate** | Need to change embedding geometry, learn new similarity patterns, teach domain-specific ranking | Need text generation, classification with a rubric, one-off judgments |

**How this played out in our pipeline:**

| Task | Approach | Why |
|---|---|---|
| **Trial embeddings** (bi-encoder) | Fine-tune | No prompt can fix the anisotropy problem. Base BioLinkBERT produced embeddings with a cosine spread of only 0.047 — every trial looked equally similar to every query. This is a problem with the model's *internal geometry*, not its instructions. We needed contrastive fine-tuning (586K triplets) to reshape the embedding space so similar trials cluster together and dissimilar ones separate. |
| **Relevance labeling** (LLM-as-judge) | Prompt | Claude Haiku already understands clinical trial relevance — it was trained on medical text. We just need to tell it our rubric: "Score 0 = wrong cancer type, Score 3 = highly relevant." A detailed prompt with the grading criteria and a few examples is sufficient. Fine-tuning Claude would be overkill, expensive, and we don't have enough labeled data for it. |
| **Cross-encoder scoring** | Fine-tune | The cross-encoder needs to learn a specific scoring function: "given this query concatenated with this trial, output a relevance score." Base BERT has no concept of clinical trial relevance scoring. We train on 200K (query, trial, label) pairs to teach this behavior. |
| **LightGBM ranking** | Train from scratch | LightGBM is a traditional ML model, not a neural network. There's no pre-trained LightGBM to fine-tune or prompt. We train on 6,018 feature vectors with LambdaRank. |
| **Query parsing** (planned) | Prompt (via LangGraph) | Extracting structured information from patient queries ("cancer type: breast, stage: 3, prior treatments: chemotherapy") is a task that Claude can handle with a well-structured prompt. No fine-tuning needed — the model already understands medical concepts. |

**The anisotropy story illustrates the boundary clearly.** When we first tried semantic search with base BioLinkBERT, the top 1000 results for any query had cosine similarities ranging from only 0.817 to 0.864 — a spread of 0.047. Three "hub" trials appeared in 33% of all result slots regardless of the query. This is not a prompting problem. There's no text you can prepend to the model's input that would reshape its 768-dimensional embedding space. The geometry is baked into the parameters. Fine-tuning with contrastive loss (MNRL) restructured the space: after training, the top-5 cosine spread expanded to 0.10, hub trials disappeared, and semantically relevant trials clustered correctly.

**Rule of thumb:** If the problem is about *what the model knows* (it doesn't understand your rubric), prompt it. If the problem is about *how the model represents information* (its embeddings, its scoring function, its internal geometry), fine-tune it.

### Vector Spaces, Similarity Metrics, and Nearest Neighbor Search

**What is a vector space?** Our embeddings live in a 768-dimensional space. Each dimension is an axis. Each trial's embedding is a point in this space. Trials about similar topics cluster together — breast cancer trials form a cluster, lung cancer trials form another.

**Three similarity/distance metrics you must know:**

| Metric | Formula | Range | When to Use |
|--------|---------|-------|-------------|
| **Euclidean (L2)** | `sqrt(Σ(a_i - b_i)²)` | [0, ∞) | When magnitude matters (coordinates, pixel distances) |
| **Cosine similarity** | `(a · b) / (‖a‖ × ‖b‖)` | [-1, 1] | When direction matters, not magnitude (text embeddings) |
| **Dot product (IP)** | `Σ(a_i × b_i)` | (-∞, ∞) | Equivalent to cosine when vectors are L2-normalized |

**Why cosine for text?** We care about *what concept* the embedding represents (its direction in 768-dim space), not *how strongly* it represents it (its length, which depends on sentence length and word frequency). Cosine similarity measures direction only. After L2 normalization (scaling every vector to length 1.0), **dot product = cosine similarity**, which is why FAISS `IndexFlatIP` (inner product) computes cosine similarity for us.

**Exact vs Approximate Nearest Neighbor (ANN):**

Our FAISS index uses **exact search** (brute-force scan of all 140K vectors, ~37ms). At larger scale, exact search becomes impractical:

| Algorithm | How It Works | Complexity | Accuracy | Use When |
|-----------|-------------|------------|----------|----------|
| **Flat (exact)** | Compare query to every vector | O(n) | 100% | n < 1M |
| **IVF** | Cluster vectors into groups, only search nearby clusters | O(n/k) | 95-99% | 1M-100M |
| **HNSW** | Build a navigable graph linking similar vectors | O(log n) | 95-99% | 1M-100M (best speed/accuracy) |
| **PQ** | Compress each 768-float vector to ~64 bytes | O(n) but faster | 90-95% | Memory-constrained |
| **IVF-PQ** | Cluster + compress | O(n/k), fast | 85-95% | 100M+ (billion-scale) |

**HNSW (Hierarchical Navigable Small World)** is the most common production choice. It builds a multi-layer graph where each node is a vector. Layer 0 has all vectors connected to their nearest neighbors. Layer 1 is a sparser "highway" with fewer nodes. Searching starts at the top layer (fast, coarse navigation) and descends to lower layers (slow, precise). Think of it like searching a map: start at the country level, zoom into the state, then the city, then the street.

**Interview answer for "How would you scale TrialMine to 10M trials?":** Replace `IndexFlatIP` with `IndexHNSWFlat`. Set `M=32` (connections per node) and `efSearch=128` (search beam width). Build time increases from minutes to hours, but search time stays under 5ms. Accuracy drops to ~97% (acceptable — we're retrieving 200 candidates for re-ranking anyway, so missing 6 isn't critical).

### The BM25 Formula

Section 5 explains BM25 intuitively. Here's the actual math for your interview.

**TF-IDF (the predecessor):**
```
TF-IDF(term, doc) = TF(term, doc) × IDF(term)

  TF(term, doc) = count of term in doc
  IDF(term) = log(N / df(term))
  N = total documents,  df = docs containing term
```

**Problem:** Raw TF is unbounded. A doc mentioning "cancer" 100 times scores 100x higher than one mentioning it once. Also, no length normalization — longer documents get unfairly high scores because they contain more word occurrences.

**BM25 fixes both:**
```
BM25(query, doc) = Σ      IDF(t) × TF(t, doc) × (k1 + 1)
                  t in q         ─────────────────────────────────────
                                 TF(t, doc) + k1 × (1 - b + b × |doc| / avgdl)
```

Two key parameters:
- **k1 = 1.2** (default): Controls TF saturation. At k1=1.2: TF=1 scores 1.0, TF=2 scores 1.33, TF=10 scores 1.67. Diminishing returns — mentioning "cancer" 10 times isn't 10x better than once.
- **b = 0.75** (default): Controls length normalization. b=1.0 means full length penalty (long docs score lower), b=0.0 means no length adjustment.

**BM25 IDF** (Robertson-Sparck Jones):
```
IDF(t) = log( (N - df + 0.5) / (df + 0.5) )
```

**Worked example.** Query: "EGFR lung cancer", corpus of 140,000 trials, avg length 500 words:

| Term | df (docs with term) | IDF |
|------|---------------------|-----|
| EGFR | 2,100 | log((140000 - 2100 + 0.5) / (2100 + 0.5)) = **4.18** |
| lung | 11,000 | log((140000 - 11000 + 0.5) / (11000 + 0.5)) = **2.46** |
| cancer | 135,000 | log((140000 - 135000 + 0.5) / (135000 + 0.5)) = **-3.30** |

"Cancer" gets **negative IDF** — it appears in almost every document, so it's anti-informative. "EGFR" has the highest IDF because it's rare and specific. This is exactly right: "EGFR" distinguishes relevant trials, "cancer" does not.

For a doc with TF(EGFR)=3, TF(lung)=5, TF(cancer)=12, doc length=600:
```
BM25 for "EGFR":
  = 4.18 × (3 × 2.2) / (3 + 1.2 × (1 - 0.75 + 0.75 × 600/500))
  = 4.18 × 6.6 / (3 + 1.2 × 1.15)
  = 4.18 × 6.6 / 4.38  =  6.30

BM25 for "lung":
  = 2.46 × (5 × 2.2) / (5 + 1.2 × 1.15)
  = 2.46 × 11.0 / 6.38  =  4.24

BM25 for "cancer":
  = -3.30 × ...  (negative — this term HURTS the score)
  Elasticsearch clamps this to 0.

Total BM25 ≈ 6.30 + 4.24 + 0 = 10.54
```

**Key takeaway:** Rare, specific terms (drug names, gene mutations) dominate BM25 scores. Common terms ("cancer", "treatment") contribute nothing. This is exactly why BM25 is strong for clinical trial search — it rewards specificity.

### Learning to Rank: Pointwise, Pairwise, and Listwise

**Learning to Rank (L2R)** is the ML subfield dedicated to producing ranked lists. It's a classic MLE interview topic and the theoretical foundation for our cross-encoder and LightGBM stages. There are three paradigms:

**Pointwise — predict absolute relevance independently:**
```
Input:  (query, single document)  →  Model  →  predicted relevance = 2.7
```
- Treats ranking as regression or classification on individual documents
- **In our pipeline:** The cross-encoder is pointwise — it predicts binary relevance (0 or 1) for each (query, trial) pair independently
- **Strength:** Simple to implement and train
- **Weakness:** Doesn't consider relative ordering. If docs score 2.7 and 2.8, the model doesn't know that their ORDER matters

**Pairwise — predict which document is MORE relevant:**
```
Input:  (query, doc_A, doc_B)  →  Model  →  "doc_A is more relevant"
```
- Learns from pairs: "for this query, document A should rank above document B"
- **In our pipeline:** MNRL (the bi-encoder loss) is pairwise — it learns that the positive trial should rank higher than the negatives
- **Classic algorithms:** RankNet, RankSVM, LambdaMART
- **Strength:** Directly optimizes relative ordering
- **Weakness:** Treats all position swaps equally. Swapping positions 1↔2 is penalized the same as swapping 98↔99, even though users only see the top results

**Listwise — optimize a metric over the entire ranked list:**
```
Input:  (query, [doc_1, ..., doc_50])  →  Model  →  ranking that maximizes NDCG
```
- Directly optimizes the evaluation metric we care about
- **In our pipeline:** LambdaRank (used by LightGBM) is listwise — it weights gradient updates by how much each swap would change NDCG
- **Classic algorithms:** LambdaRank, LambdaMART, ListNet, SoftRank
- **Strength:** Focuses learning on the swaps that matter most (top positions, big relevance differences)

**How LambdaRank works in detail:**

Standard gradient descent says: "adjust parameters to reduce the loss." LambdaRank modifies the gradients by multiplying by |ΔNDCG| — the absolute NDCG change from swapping two documents:

```
lambda_gradient(doc_i, doc_j) = base_gradient × |ΔNDCG(swap i ↔ j)|
```

Example: if swapping #1 and #2 would change NDCG by 0.15, but swapping #8 and #9 would change it by only 0.002, the first swap gets **75x more gradient weight**. The model learns to focus on getting the top results right — which is exactly what NDCG measures.

**Why our pipeline uses all three:**

| Stage | L2R Paradigm | Why This Paradigm |
|-------|-------------|-------------------|
| Bi-encoder (MNRL) | Pairwise | Fast retrieval needs pairwise contrast — "this trial, not that one" |
| Cross-encoder (BCE) | Pointwise | Re-ranking needs per-document scores, binary labels limit us to pointwise |
| LightGBM (LambdaRank) | Listwise | Final stage directly optimizes the metric we report (NDCG) |

Each stage uses a more sophisticated ranking paradigm. This is not an accident — it reflects the cost/accuracy tradeoff. Pairwise is cheapest (can pre-compute embeddings), listwise is most expensive (needs all candidates scored together), so we use them in order of computational budget.

**Interview question: "Why not use LambdaRank for everything?"** Answer: LambdaRank needs numeric features as input. The retrieval scores (BM25, semantic, CE) ARE those features. You need the upstream stages to produce the signals that LambdaRank combines. Also, LambdaRank over all 140K docs per query would be impractical — the multi-stage pipeline narrows from 140K → 200 → 50 → 20, making expensive stages tractable.

### Loss Functions: Choosing the Right Training Signal

A **loss function** tells the model how wrong its prediction is. The choice of loss fundamentally shapes what the model can learn. Here's every loss relevant to our pipeline:

**Mean Squared Error (MSE):** `loss = (predicted - actual)²`
- Used for: regression (predict a continuous number)
- Example: predict relevance 2.3 when true label is 3.0 → loss = 0.49
- **Not used in our pipeline**, but would be the natural choice for retraining the CE on graded (0-3) labels

**Binary Cross-Entropy (BCE):** `loss = -[y × log(p) + (1-y) × log(1-p)]`
- Used for: binary classification (yes/no)
- **Used by our cross-encoder:** y=1 for relevant pairs, y=0 for irrelevant
- The loss goes to infinity when the model is confidently wrong (predicts p≈0 for y=1)
- **Why it's limiting:** binary labels can't express graded relevance. A marginal trial (score 1) and a perfect match (score 3) both get y=1.

**Categorical Cross-Entropy:** `loss = -Σ_c y_c × log(p_c)`
- Used for: multi-class classification (which category?)
- Softmax inside MNRL uses this form internally

**Contrastive Losses (what our bi-encoder uses):**

These train embeddings by pulling similar items together and pushing different items apart:

| Loss | Negatives per step | Used in our pipeline? |
|------|-------------------|----------------------|
| **Triplet Loss:** `max(0, d(q,pos) - d(q,neg) + margin)` | 1 | No (too slow) |
| **InfoNCE:** `-log(exp(sim(q,pos)/τ) / Σ exp(sim(q,neg_i)/τ))` | N (in-batch) | This is MNRL's foundation |
| **MNRL (ours):** InfoNCE + hard negatives | 31 in-batch + 1 hard = **32** | Yes — bi-encoder training |

MNRL gives us **33 contrasts per query per step** (1 positive + 31 in-batch negatives + 1 hard negative). Triplet loss gives only 2 (1 positive + 1 negative). This 16x efficiency is why MNRL trains much faster.

The **temperature parameter** `τ` (or equivalently, `scale = 1/τ`): Our `scale=20` means `τ = 0.05`. Lower temperature → sharper softmax → model must be more confident. Without scaling: similarities [0.9, 0.3] → softmax → [0.65, 0.35]. With scale=20: [18, 6] → softmax → [0.9999, 0.0001]. The model gets penalized severely for being even slightly uncertain.

**LambdaRank (what LightGBM uses):**
Not a traditional loss — it modifies gradients directly based on NDCG impact (see Learning to Rank section above). This is how LightGBM can optimize a non-differentiable metric like NDCG.

**The key insight — your loss defines your ceiling:**

| Loss | What it teaches | What it CAN'T teach |
|------|----------------|---------------------|
| BCE | "Is this the right disease?" | "How relevant on a 0-3 scale?" |
| MNRL | "This trial is more relevant than that one" | "This should be at position 1, not position 3" |
| LambdaRank | "Get the top of the list right" | (Limited by feature quality, not loss design) |

This explains why our cross-encoder achieves 0.992 validation NDCG but hurts results when used as a standalone ranker — binary training labels create a binary model, no matter how powerful the architecture. **The training signal defines the ceiling.**

### Overfitting, Underfitting, and the Bias-Variance Tradeoff

**Overfitting** = the model memorizes training data instead of learning general patterns. It scores perfectly on training queries but fails on new ones. A student who memorizes the answer key but doesn't understand the material.

**Underfitting** = the model is too simple to capture the patterns. A model that predicts "relevance = 0.5" for every trial ignores all useful information. A student who hasn't studied at all.

**The Bias-Variance Tradeoff:**
- **Bias** = error from oversimplifying. A model using only BM25 score to predict relevance has high bias — it assumes relevance is purely about keyword matching, ignoring phase, status, enrollment.
- **Variance** = error from oversensitivity to training data. A complex model trained on 20 queries memorizes those specific queries — add one new query and predictions change wildly.
- **The tradeoff:** Simple models → high bias, low variance. Complex models → low bias, high variance. The goal is the sweet spot.

**How this appeared in our pipeline:**

| Problem | Diagnosis | Fix |
|---------|-----------|-----|
| LightGBM NDCG@5 = 0.980 on train, 0.844 on LOOCV | Overfitting (20 queries is too few) | More data: 20 → 145 queries |
| CE val NDCG = 0.992, but real-world ranking is poor | Overfitting to binary task | Need graded labels (future work) |
| Base BioLinkBERT: cosine range 0.047 | Underfitting (never trained to rank) | Contrastive fine-tuning (586K triplets) |
| Fair test NDCG=0.670 vs LOOCV=0.843 | Not overfitting — test queries are genuinely harder | This is expected generalization gap |

**Regularization — techniques to prevent overfitting:**

| Technique | How it works | Where we use it |
|-----------|-------------|----------------|
| **Early stopping** | Stop training when val metric stops improving | CE training stopped at epoch 1 |
| **Weight decay (L2)** | Penalize large params: `loss + λ × Σ(w²)` | AdamW optimizer in bi-encoder fine-tuning |
| **Dropout** | Randomly zero 10% of neurons during training — forces redundancy | Inside BERT layers (default 0.1) |
| **Data augmentation** | Create more training examples | Hard negative mining (730K triplets) |
| **Cross-validation** | Evaluate on held-out subsets | LOOCV for LightGBM |
| **Warmup** | Small LR early prevents destroying pre-trained features | `warmup_ratio=0.1` for bi-encoder |

**Interview tip:** Don't just list regularization techniques — explain which ones you used and why they were appropriate. For example: "We used early stopping for the cross-encoder because additional epochs on binary labels wouldn't teach graded relevance — the model achieves 0.992 in one epoch, so more training just memorizes the training set without learning new capabilities."

### Model Compression: Distillation, Quantization, and Pruning

Our cross-encoder takes ~4 seconds to score 50 candidates on CPU. For a user-facing search engine, this is unacceptable — users expect results in under 1 second. **Model compression** is the family of techniques for making models smaller and faster while preserving accuracy. This is one of the most practical MLE interview topics because every production ML system faces this tradeoff.

**Knowledge Distillation: teach a small model to mimic a large one.**

The core idea: train a small "student" model to reproduce the outputs of a large "teacher" model, rather than training the student from scratch on the original labels.

```
Teacher (BioLinkBERT-base, 110M params):
  Input: (query, trial) → Output: relevance score 0.87 (soft label)

Student (MiniLM, 22M params):
  Input: (query, trial) → Trained to predict: 0.87 (not just "relevant/irrelevant")
```

Why does distillation work better than training the student directly on the original labels?

1. **Soft labels carry more information.** A hard label says "relevant" (1) or "irrelevant" (0). The teacher's soft prediction of 0.87 says "very relevant but not perfect." The soft label also reveals relationships: if the teacher scores trial A at 0.91 and trial B at 0.87, the student learns that A is slightly better than B — information the binary labels don't contain.

2. **Dark knowledge.** When the teacher assigns small probabilities to "wrong" classes, it reveals which mistakes are "almost right." A teacher that gives "irrelevant" a probability of 0.05 vs 0.001 is telling the student which irrelevant trials are at least in the right ballpark.

3. **Temperature scaling.** During distillation, we soften the teacher's output probabilities using a temperature parameter:
   ```
   soft_probs = softmax(logits / T)
   ```
   At T=1 (normal), a confident teacher outputs [0.95, 0.04, 0.01]. At T=5: [0.45, 0.30, 0.25]. Higher temperature reveals more about the teacher's uncertainty, giving the student richer signal.

**Concrete plan for our pipeline:** Use our fine-tuned BioLinkBERT cross-encoder (110M params) as teacher. Train MiniLM (22M params, 5× smaller) as student on our 6,018 labeled pairs plus the teacher's soft scores. Expected improvement: ~4s → ~500ms (roughly proportional to param reduction), with <2% NDCG loss based on published distillation results.

Common distilled models: DistilBERT (66M → 40% faster), MiniLM (22M → 5× faster), TinyBERT (14M → 7× faster). The speedup comes from fewer layers (6 vs 12) and smaller hidden dimensions (384 vs 768).

**Quantization: reduce numerical precision.**

Neural networks train in **fp32** (32-bit floating point: 4 bytes per parameter). But do you really need 32 bits of precision for inference? Quantization reduces precision to save memory and compute:

```
fp32 (4 bytes):  sign(1) + exponent(8) + mantissa(23) = 32 bits
                 Range: ±3.4×10³⁸, Precision: ~7 decimal digits

fp16 (2 bytes):  sign(1) + exponent(5) + mantissa(10) = 16 bits
                 Range: ±65,504, Precision: ~3 decimal digits

int8 (1 byte):   8 bits, values -128 to 127
                 No exponent, must scale values to fit range
```

| Precision | Memory per param | Relative speed | Accuracy loss | Our usage |
|---|---|---|---|---|
| fp32 | 4 bytes | 1× (baseline) | 0% | Default inference |
| fp16 | 2 bytes | ~1.5-2× on GPU | <0.1% | We use fp16 during training (`fp16=True`) |
| int8 | 1 byte | ~2-4× on CPU | 0.5-2% | Not yet used; best opportunity for speedup |
| int4 | 0.5 bytes | ~4-8× | 2-5% | Too aggressive for ranking accuracy |

**How int8 quantization works:**
1. Find the min/max of each weight tensor (e.g., min=-0.8, max=1.2)
2. Map this range linearly to [-128, 127]: `quantized = round(value × 127 / max_abs)`
3. During inference, dequantize: `value ≈ quantized × max_abs / 127`
4. Use integer arithmetic (much faster on CPUs) for matrix multiplications

The key insight: weight values in trained models cluster around zero with small variance. Most values are between -0.5 and 0.5 — you don't need 7 decimal digits of precision to represent them. The small quantization error averages out across millions of operations.

**Concrete plan:** Export our cross-encoder to ONNX format, apply dynamic int8 quantization via ONNX Runtime. Expected: ~2× speedup (4s → ~2s) with <1% NDCG drop. Combined with distillation: MiniLM + int8 could reach ~250ms — within the real-time budget.

**Pruning: remove unnecessary parameters.**

Pruning sets near-zero parameters to exactly zero, creating a **sparse** model. The theory (the **Lottery Ticket Hypothesis**) suggests that dense networks contain sparse subnetworks that can match the full network's performance.

Two types:
- **Unstructured pruning:** Zero out individual weights. Can remove 50-90% of weights. Problem: sparse matrices need special hardware/software to be fast. Standard matrix multiplication doesn't benefit from zeros scattered randomly.
- **Structured pruning:** Remove entire attention heads, neurons, or layers. Removes 20-50% of structure. Works with standard hardware because you literally have a smaller matrix, not a sparse one.

```
Unstructured (90% sparse):            Structured (remove 2 of 12 heads):
[0, 0.3, 0, 0, 0]                    Attention: 12 heads → 10 heads
[0, 0, 0, 0.7, 0]                    Params: 2.36M → 1.97M per layer
[0.2, 0, 0, 0, 0]                    Works on any hardware
↑ needs sparse matrix support         ↑ standard dense operations
```

**Practical relevance:** Pruning is less commonly used than distillation and quantization for Transformer models. Distillation gives more predictable speedups with well-studied accuracy tradeoffs. We mention pruning for completeness and because interviewers ask about it, but our compression roadmap prioritizes distillation first, then quantization.

**Summary — compression techniques for our pipeline:**

| Technique | How | Speedup | Accuracy cost | Effort | Priority |
|---|---|---|---|---|---|
| Distillation (→ MiniLM) | Train small model on teacher's soft labels | ~5× | 1-3% NDCG | Medium (need to train) | High |
| Quantization (int8) | Reduce weight precision | ~2× | <1% NDCG | Low (export + config) | High |
| Distill + Quantize | Both together | ~8-10× | 2-4% NDCG | Medium | Highest |
| Structured pruning | Remove attention heads | ~1.3× | 1-2% NDCG | High (need to identify which heads) | Low |

The realistic path: distill to MiniLM (4s → ~800ms), then quantize to int8 (~800ms → ~400ms). This brings cross-encoder re-ranking from "deal-breaker slow" to "acceptable for production."

### Training-Serving Skew

**Training-serving skew** (also called train-serve skew) occurs when the data or features a model sees during training differ from what it sees during serving (inference). This is one of the most insidious production ML bugs because it doesn't cause crashes or errors — it silently degrades model quality.

**Common causes:**

1. **Different preprocessing code.** Training script normalizes text one way, serving API normalizes it differently. Example: training lowercases input, serving doesn't — the model sees "EGFR" at serving time but was trained on "egfr."

2. **Stale features.** Model was trained on features computed from last month's data. At serving time, features are computed from today's data. If distributions shifted, predictions degrade.

3. **Feature leakage during training.** A feature that's available during training but not during serving (e.g., using the "correct answer" as a feature). The model learns to rely on it, then fails without it.

4. **Different software versions.** Training uses numpy 1.24, serving uses numpy 1.26. A change in float rounding produces subtly different features.

**How our pipeline avoids training-serving skew:**

**1. Single `compute_features()` function.** This is the most important defense. The `compute_features()` function in `src/TrialMine/models/ranker.py` is called in both places:

```python
# During training (scripts/train_ranker.py):
features = compute_features(query, candidate, trial_doc)

# During serving (RankingBlender.rerank()):
feats = compute_features(query, c)
```

The same function, the same code path, the same feature engineering. If we had separate training and serving feature pipelines (a common mistake), any divergence would silently corrupt predictions. For example, if training computed `enrollment_log` as `log1p(enrollment)` but serving used `log(enrollment + 1)` (mathematically identical, but different floating-point results for edge cases), the LightGBM model would receive slightly different inputs than it was trained on.

**2. NCT ID-level data splitting.** When splitting data into train/test, we split by *query*, not by individual (query, trial) pairs. If query Q has 40 labeled trials, ALL 40 go into either train or test — never split across both. This prevents a subtle form of leakage: if the same trial appears in both train and test (paired with different queries), the model might memorize trial-specific patterns rather than learning generalizable ranking features.

Our evaluation scripts enforce this:
- Training: 145 queries (IDs 0-19, 100-149, 200-274) — all labels for these queries
- Test: 50 queries (IDs 300-349) — completely disjoint set

**3. Features computed from live retrieval scores.** The LightGBM model's input features (`bm25_score`, `semantic_score`, `cross_encoder_score`, `rrf_score`) are computed from the actual retrieval pipeline at both training and serving time. We don't cache training-time scores and serve from the cache. This means:
- If the Elasticsearch index is updated with new trials, BM25 scores naturally reflect the new corpus statistics
- If the FAISS index is rebuilt with updated embeddings, semantic scores reflect the new embedding space

**How training-serving skew could still bite us:**

1. **Index staleness.** If the BM25 index is updated with new trials but the FAISS index isn't rebuilt, the score distributions diverge. BM25 IDF values change (a term that was rare becomes common), but semantic scores don't reflect the new trials. The LightGBM model was trained on correlated BM25/semantic scores — if they become decorrelated, predictions degrade.

2. **Model version mismatch.** If the bi-encoder model is retrained but the FAISS index was built with the old model's embeddings, semantic scores become meaningless — the query encoder and the stored trial embeddings speak different "languages."

3. **Feature distribution shift.** Our `is_recruiting` feature is based on trial status. If we retrain LightGBM using a snapshot where 60% of trials are recruiting, but serve on a corpus where only 30% are recruiting (because many trials completed), the feature distribution shifts. LightGBM's learned thresholds for `is_recruiting` interactions may no longer be appropriate.

**The general defense:** Use the same code path for training and serving feature computation, version all artifacts (model, index, features) together, and monitor feature distributions in production. When any component is updated, validate that the full pipeline's metrics haven't degraded before deploying.

---

## 3. How We Measure Search Quality (Read This First)

Before we dive into the pipeline, you need to understand how we measure whether search results are good. This section explains every metric we use. Come back to this section whenever you see a number you don't understand.

### Relevance Labels: The 0-3 Scale

To measure search quality, we need to know which trials are actually relevant to a given query. We label each (query, trial) pair on a 0-3 scale:

| Score | Meaning | Example |
|-------|---------|---------|
| 0 | **Wrong cancer type entirely** | Query about lung cancer, trial is for leukemia |
| 1 | **Marginal** — same area, wrong specifics | Query about EGFR lung cancer, trial is for general lung cancer with wrong drug |
| 2 | **Relevant** — patient could be eligible | Query about breast cancer immunotherapy, trial is a breast cancer immunotherapy study |
| 3 | **Highly relevant** — strong match | Query about Phase 3 EGFR lung cancer, trial is exactly a Phase 3 EGFR inhibitor trial |

Why not just binary (relevant/irrelevant)? Because in search, **position matters**. If we have two relevant trials, one that's a perfect match (score 3) and one that's marginal (score 1), we want the perfect match at position 1. Binary labels can't express this preference. Graded labels can.

### NDCG (Normalized Discounted Cumulative Gain)

NDCG is our primary metric. It answers: **"How good is the ranking?"** It rewards putting highly relevant results near the top.

Let's walk through a concrete example. Suppose we search for "breast cancer immunotherapy" and get 5 results:

| Position | Trial | Relevance |
|----------|-------|-----------|
| 1 | Trial A (wrong cancer) | 0 |
| 2 | Trial B (perfect match) | 3 |
| 3 | Trial C (marginal) | 1 |
| 4 | Trial D (relevant) | 2 |
| 5 | Trial E (wrong cancer) | 0 |

**Step 1: Calculate DCG (Discounted Cumulative Gain)**

For each result, we calculate a "gain" and discount it by position:

```
DCG@5 = (2^0 - 1)/log2(2) + (2^3 - 1)/log2(3) + (2^1 - 1)/log2(4) + (2^2 - 1)/log2(5) + (2^0 - 1)/log2(6)
      = 0/1.0         + 7/1.585       + 1/2.0         + 3/2.322        + 0/2.585
      = 0              + 4.416          + 0.5           + 1.292          + 0
      = 6.208
```

Why `2^rel - 1`? This makes the difference between scores exponential. A score-3 result is worth 7 gain, while a score-1 result is worth only 1 gain. This strongly rewards putting the best results first.

Why `log2(position + 1)`? This is the "discount." Results further down the list get discounted more. Position 1 has no discount (log2(2) = 1), but position 10 has a big discount (log2(11) = 3.46). This captures the intuition that users mostly look at the top few results.

**Step 2: Calculate IDCG (Ideal DCG)**

The ideal ranking would put results in order: 3, 2, 1, 0, 0.

```
IDCG@5 = (2^3 - 1)/log2(2) + (2^2 - 1)/log2(3) + (2^1 - 1)/log2(4) + (2^0 - 1)/log2(5) + (2^0 - 1)/log2(6)
       = 7/1.0 + 3/1.585 + 1/2.0 + 0/2.322 + 0/2.585
       = 7 + 1.893 + 0.5 + 0 + 0
       = 9.393
```

**Step 3: NDCG = DCG / IDCG**

```
NDCG@5 = 6.208 / 9.393 = 0.661
```

A perfect ranking would score 1.0. Our ranking scores 0.661, meaning it's decent but not ideal — the perfect match (Trial B) should have been at position 1, not position 2.

**NDCG@5 vs NDCG@10:** The number after @ is the "cutoff." NDCG@5 only considers the top 5 results. NDCG@10 considers the top 10. We use NDCG@5 as our primary metric because most users only look at the first page of results.

### MRR (Mean Reciprocal Rank)

MRR is much simpler: **"How far down do you have to scroll to find the first relevant result?"**

```
MRR = 1 / (position of first relevant result)
```

Examples:
- First result is relevant → MRR = 1/1 = 1.0
- Second result is first relevant one → MRR = 1/2 = 0.5
- Fifth result is first relevant one → MRR = 1/5 = 0.2

We average this across all queries. An MRR of 0.917 means that on average, a relevant result appears at position 1 for almost every query. MRR captures "time to first useful result."

### Bootstrap Confidence Intervals

When we report "NDCG@5 = 0.670 ± 0.08", the ± 0.08 is a **confidence interval**. Here's what it means:

We have 50 test queries. Each query has its own NDCG score. The average is 0.670, but if we picked a slightly different set of 50 queries, the average might be different. To estimate this uncertainty, we use **bootstrapping**:

1. Randomly pick 50 queries from our 50 (with replacement, so some queries get picked twice, some not at all)
2. Compute the average NDCG on this random sample
3. Repeat 1000 times
4. The middle 95% of those 1000 averages gives us the 95% confidence interval

A confidence interval of ± 0.08 means: "We're 95% confident the true NDCG is between 0.590 and 0.750." When two methods have overlapping confidence intervals, we can't be sure one is truly better — the difference might just be random noise.

### Why We Use an LLM as Judge

Labeling 990 (query, trial) pairs by hand would take 8+ hours of expert time. Instead, we used Claude Haiku (a fast, cheap AI model) to read each query-trial pair and assign a 0-3 score. This costs about $2 for 990 labels.

**Trade-off:** We don't know if Claude Haiku's "score 2" matches what a human doctor would say. 46% of labels are score 3 (highly relevant), which might mean Haiku is too generous. Without comparing to human labels (called "Cohen's kappa" — a measure of agreement between two raters), we can't be sure of the absolute numbers. But we CAN trust **relative comparisons** — if Method A scores higher than Method B using the same labels, that comparison is valid even if the absolute scores are inflated.

### Offline vs Online Metrics: What We Measure vs What Matters

Everything we've discussed so far — NDCG, MRR, precision, recall — are **offline metrics**: computed on a static dataset without any real users. Understanding the distinction between offline and online metrics is critical for MLE interviews because every real search system must bridge this gap.

**Offline metrics** (what we have):
- Computed on a fixed set of labeled (query, trial) pairs
- Evaluated after training, before deployment
- Tell you: "Given these labels, how well does the model rank?"
- Our examples: NDCG@5=0.670, NDCG@10=0.657, MRR=0.806 on 50 test queries

**Online metrics** (what we'd need in production):
- Computed from live user behavior in a deployed system
- Measured continuously during serving
- Tell you: "Are users actually finding what they need?"

| Online Metric | What It Measures | How to Compute | What It Reveals |
|---|---|---|---|
| **Click-through rate (CTR)** | Did the user click any result? | clicks / impressions | Whether results look relevant from the title/snippet |
| **Mean Reciprocal Rank from clicks** | Which position did they click first? | 1/rank of first click, averaged | Whether the best result is near the top |
| **Dwell time** | How long did they spend on the trial page? | Time between click and return to results | Whether the trial content was actually useful (vs. misleading title) |
| **Query refinement rate** | Did they rephrase and search again? | Proportion of queries followed by a modified query | Whether the first search failed — high refinement = poor results |
| **Zero-click rate** | Did they not click anything? | Proportion of queries with 0 clicks | Whether results are so poor that users give up |
| **Conversion actions** | Did they take a meaningful action? | "Save trial" / "Contact site" / "Check eligibility" clicks | Ultimate measure of search utility |

**Why offline metrics don't equal online metrics:**

1. **Labels vs behavior.** Our NDCG is based on LLM-generated labels ("Claude Haiku thinks this trial is relevant"). But a real user might disagree. A trial labeled "score 3" (highly relevant) might have impenetrable eligibility criteria — the user clicks, reads for 3 seconds, bounces. High NDCG, low dwell time. The label says "relevant"; the user says "useless."

2. **Relevance vs utility.** A completed trial (status: COMPLETED) can be textually relevant — it studied exactly the patient's cancer type with promising results. Our LLM judge scores it 3. But for a patient seeking to *enroll*, a completed trial is useless. Our offline metrics don't capture this because relevance ≠ utility. This is partly why the LightGBM blender helps — it incorporates `is_recruiting` as a feature, pushing completed trials down even when text relevance is high.

3. **Label inflation.** 46% of our labels are score 3 (highly relevant). If the LLM judge is generous, our NDCG is inflated — but inflated *consistently* across all methods, so relative comparisons are still valid. Online metrics would reveal the true baseline: if users only click 20% of our top-3 results, absolute quality is lower than NDCG suggests.

4. **Presentation bias.** In a live system, users are more likely to click results at position 1 simply because they see it first — not because it's the best result. This creates a feedback loop: position 1 gets more clicks → appears more "relevant" → gets trained to stay at position 1. Offline evaluation doesn't have this bias because labels are assigned independently of presentation order (our LLM judge doesn't know what position the trial was in).

**How to bridge the gap — A/B testing and interleaving:**

**A/B testing:** Route 50% of users to pipeline A (e.g., hybrid + CE only) and 50% to pipeline B (hybrid + CE + LightGBM). Compare online metrics between groups. If B's CTR and dwell time are higher, the LightGBM blender genuinely helps users — not just LLM judges.

**Interleaving:** For each query, merge results from both pipelines into a single interleaved list. If users consistently click pipeline B's results more often than pipeline A's (controlling for position), B is better. Interleaving is more statistically efficient than A/B testing — it detects smaller differences with fewer queries.

**Our specific gap:** We have no online metrics because TrialMine isn't deployed to real users. Our entire evaluation chain is offline: LLM-generated labels → NDCG/MRR → ablation tables. The fair evaluation (Section 13) is the strongest offline evidence we have — 50 held-out queries with pooled labeling — but it still can't tell us whether real patients find the results useful. Any claim about absolute quality (e.g., "our search is good enough for patients") would require online validation.

**Interview answer for "How would you evaluate this system in production?":** "Our offline metrics (NDCG, MRR) tell us relative ranking quality, but they're based on LLM labels, not user behavior. In production, I'd add click-through tracking, dwell time measurement, and query refinement detection. For launch, I'd run an interleaving experiment: serve both the old pipeline (BM25-only) and the full pipeline, measure which results get clicked more. The offline metrics suggest 5-8% NDCG improvement per stage — but the online lift might be larger or smaller depending on whether LLM labels correlate with user preferences."

---

## 4. Data Pipeline: Getting 140,000 Clinical Trials

### What Is ClinicalTrials.gov?

ClinicalTrials.gov is a database run by the U.S. National Library of Medicine. Every clinical trial conducted in the U.S. (and many international ones) must be registered here. Each trial has a record with an ID (like "NCT01234567") and structured fields: title, conditions being studied, interventions (drugs/treatments), eligibility criteria (who can join), phase, recruitment status, and more.

### How We Downloaded the Data

We used the ClinicalTrials.gov REST API v2. An API (Application Programming Interface) is like a structured way to ask a website for data. Instead of loading a web page, we send a request like:

```
GET https://clinicaltrials.gov/api/v2/studies?query.cond=cancer&pageSize=1000
```

This returns 1000 cancer trials as JSON (a structured data format). The API uses **pagination** — since there are 140,000+ results, we can't get them all at once. Each response includes a `pageToken` that we include in the next request to get the next page.

The search query is defined as a constant:

```python
ONCOLOGY_QUERY = "cancer OR oncology OR tumor OR carcinoma OR lymphoma OR leukemia OR melanoma OR sarcoma"
```

This broad query captures all oncology trials. We use OR to match any of 8 cancer-related terms.

**Key functions we wrote:**

**`src/TrialMine/data/models.py` — The `Trial` and `Location` classes**

These are **Pydantic models**. Pydantic is a Python library for data validation — it ensures that every trial we process has the right field types. If the API returns a string where we expect an integer, Pydantic catches the error immediately instead of letting it cause a bug later.

```python
class Location(BaseModel):
    facility: str | None = None    # Hospital/center name
    city: str | None = None
    state: str | None = None
    country: str | None = None
    zip_code: str | None = None

class Trial(BaseModel):
    nct_id: str = Field(..., description="ClinicalTrials.gov unique identifier")
    title: str = Field(default="")
    brief_summary: str | None = None
    detailed_description: str | None = None
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)   # names only, not full objects
    eligibility_criteria: str | None = None
    min_age: str | None = None      # raw string, e.g. "18 Years"
    max_age: str | None = None
    sex: str | None = None          # "ALL", "FEMALE", or "MALE"
    phase: str | None = None        # e.g. "Phase 1", "Phase 3"
    status: str | None = None       # e.g. "RECRUITING", "COMPLETED"
    enrollment: int | None = None
    start_date: str | None = None
    completion_date: str | None = None
    sponsor: str | None = None
    locations: list[Location] = Field(default_factory=list)
    url: str | None = None          # ClinicalTrials.gov URL
```

**18 fields total.** Most are `None`-able because many trials have missing data. We made a deliberate decision to **keep trials with missing fields** rather than throwing them away. A Phase 3 breast cancer trial with no eligibility text is still a valid search result — the patient should know it exists. Note that `min_age` and `max_age` are stored as raw strings (e.g., "18 Years") rather than parsed to numbers — this preserves the exact API format and avoids edge cases like "N/A" or "1 Month."

**`src/TrialMine/data/download.py` — `download_oncology_trials()` and `fetch_page()`**

The download system has two key functions:

**`fetch_page(client, params)`** fetches one page from the API with **exponential backoff retry logic**:

```python
def fetch_page(client, params):
    for attempt in range(3):          # 3 retry attempts
        try:
            response = client.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
            if attempt < 2:
                delay = 2 ** attempt * 2   # 2s, 4s, 8s
                logger.warning("Retry %d after %.1fs: %s", attempt + 1, delay, e)
                time.sleep(delay)
            else:
                raise                      # give up after 3 tries
```

**Exponential backoff** means each retry waits longer: 2 seconds, then 4, then 8. This handles transient network errors and server overload without hammering the API.

**`download_oncology_trials(output_dir, query)`** is the main download orchestrator. It supports **resumable downloads** via a state file (`.download_state.json`) that saves the current `pageToken`, pages completed, and trials downloaded after every successful page fetch. If the download crashes at page 95 of 150, rerunning the script picks up at page 95 instead of starting over. The state file looks like:

```json
{"next_page_token": "CAoQAg...", "pages_saved": 95, "trials_downloaded": 95000}
```

**`src/TrialMine/data/parse.py` — `parse_study()` and `_get()`**

The API returns deeply nested JSON. A trial's title is buried at `protocolSection.identificationModule.officialTitle`. To safely navigate this without crashing on missing fields, we wrote a **safe nested access helper**:

```python
def _get(d, *keys, default=None):
    """Safe nested dict access: _get(obj, 'a', 'b', 'c') = obj['a']['b']['c']"""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, {})
    return d if d != {} else default
```

This is used throughout `parse_study()` to extract fields. For example:
```python
nct_id = _get(raw, "protocolSection", "identificationModule", "nctId")
title = _get(raw, "protocolSection", "identificationModule", "officialTitle", default="")
conditions = _get(raw, "protocolSection", "conditionsModule", "conditions", default=[])
enrollment = _get(raw, "protocolSection", "designModule", "enrollmentInfo", "count")
```

If any intermediate level is missing (e.g., a trial has no `designModule`), `_get` returns the default instead of crashing. The function returns `None` if `nct_id` is missing — that trial is skipped entirely.

`parse_raw_files(raw_dir)` processes all saved JSON page files, calls `parse_study()` on each trial, and tracks statistics: how many trials had missing eligibility criteria, missing conditions, or failed to parse. It continues through errors rather than crashing.

**`src/TrialMine/data/store.py` — `store_trials()` and the SQLite ORM**

Saves the parsed `Trial` objects to a SQLite database. **SQLite** is a lightweight database that stores everything in a single file (our `data/trials.db` is 912 MB). We chose SQLite because our data flow is "write once, read many" — we download all the trials once, then read them repeatedly for indexing and evaluation. SQLite is perfect for this pattern: zero configuration, no server to manage, and the whole database is one portable file.

The SQLAlchemy ORM maps `Trial` objects to a `trials` table. List fields (conditions, interventions, locations) can't be stored directly in SQL columns, so they're **JSON-serialized**:

```python
def _to_row(trial):
    return TrialRow(
        nct_id=trial.nct_id,
        conditions=json.dumps(trial.conditions),           # ["breast cancer"] -> '["breast cancer"]'
        interventions=json.dumps(trial.interventions),
        locations=json.dumps([loc.model_dump() for loc in trial.locations]),
        # ... other fields mapped directly
    )
```

And deserialized back when loading:
```python
def _from_row(row):
    return Trial(
        nct_id=row.nct_id,
        conditions=json.loads(row.conditions or "[]"),     # '["breast cancer"]' -> ["breast cancer"]
        locations=[Location(**loc) for loc in json.loads(row.locations or "[]")],
        # ...
    )
```

The `store_trials()` function pre-fetches all existing `nct_id`s in one query and filters out duplicates before inserting, so it's safe to re-run without creating duplicates. It inserts in batches of 500 with per-batch commits for crash safety.

The SQLite table has **indexes** on `nct_id` (unique, for fast lookups), `status` (for filtering), and `phase` (for filtering).

**`scripts/download_data.py` — The orchestration script**

This is the script you'd actually run (`make download` or `python scripts/download_data.py`). It orchestrates the full pipeline:

1. **Download**: Call `download_oncology_trials()` (skippable with `--skip-download`)
2. **Parse**: Call `parse_raw_files()` to convert JSON pages to Trial objects
3. **Store**: Call `store_trials()` to persist in SQLite
4. **Summary**: Call `print_summary()` which shows: total trials, by-status breakdown, by-phase breakdown, eligibility coverage (% with criteria text), and top-20 most common conditions

### The Corpus

After downloading, we have 140,723 oncology clinical trials. The distribution is heavily skewed:

| Cancer Type | Number of Trials |
|-------------|-----------------|
| Breast cancer | ~15,000 |
| Lung cancer | ~11,000 |
| Colorectal cancer | ~5,000 |
| Leukemia | ~4,500 |
| Mesothelioma | 292 |
| Neuroblastoma | 361 |

This skew matters for training — if we naively train on all trials, the model will learn "breast cancer" extremely well but fail on rare cancers. We address this later with stratified sampling.

---

## 5. Stage 1: BM25 Keyword Search with Elasticsearch

### What Is BM25?

BM25 (Best Match 25) is a formula for ranking documents by keyword relevance. Think of it as a sophisticated version of "count how many times the search words appear in the document." It has three key ideas:

1. **Term Frequency (TF):** If a query word appears more often in a document, that document is probably more relevant. But there are diminishing returns — appearing 10 times isn't 10x better than appearing once.

2. **Inverse Document Frequency (IDF):** Words that appear in fewer documents are more informative. "Pembrolizumab" (a specific drug) appearing in a trial is a strong signal. "Cancer" appearing in a trial tells us almost nothing (every trial in our corpus is about cancer).

3. **Document Length Normalization:** A 5000-word trial that mentions "immunotherapy" once is less focused on immunotherapy than a 500-word trial that mentions it once. BM25 adjusts for document length.

### What Is Elasticsearch?

Elasticsearch is a search engine that implements BM25 and many other search features. When you add documents to Elasticsearch, it builds an **inverted index**: a data structure that maps each word to the list of documents containing it. When you search for "breast cancer immunotherapy," it instantly looks up which documents contain each of those words and scores them with BM25. This is extremely fast — our searches take about 22 milliseconds for 140,000 documents.

### What Is Field Boosting?

Our clinical trials have multiple text fields: title, conditions, interventions, summary, eligibility criteria. If the word "immunotherapy" appears in the trial's **title**, that's a stronger signal than if it appears in the eligibility criteria. **Field boosting** lets us tell Elasticsearch this:

- `title` gets **3x boost** — a match in the title counts 3 times as much
- `conditions` gets **2x boost** — matching the condition is important
- Everything else gets **1x** (no boost)

### What Is Stemming?

Elasticsearch applies **stemming**: reducing words to their root form. "Running," "runs," and "ran" all become "run." This means a search for "treatment" will also match documents containing "treating" or "treatments." We use the standard English stemmer.

It also removes **stop words** — extremely common words like "the," "is," "and" that don't carry meaning for search.

### Key Functions

**`src/TrialMine/retrieval/bm25.py` — `ElasticsearchIndex` class**

This class wraps all our Elasticsearch operations. Let's start with the Elasticsearch configuration:

```python
INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,      # only 1 machine, don't split the data
        "number_of_replicas": 0,     # no backup copies (we can rebuild from SQLite)
        "analysis": {
            "analyzer": {
                "english_custom": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "english_stemmer", "english_stop"],
                },
            },
            "filter": {
                "english_stemmer": {"type": "stemmer", "language": "english"},
                "english_stop": {"type": "stop", "stopwords": "_english_"},
            },
        },
    },
    "mappings": {
        "properties": {
            "nct_id": {"type": "keyword"},       # exact match only (no stemming)
            "title": {"type": "text", "analyzer": "english_custom"},
            "brief_summary": {"type": "text", "analyzer": "english_custom"},
            "conditions": {"type": "text", "analyzer": "english_custom"},
            "interventions": {"type": "text", "analyzer": "english_custom"},
            "eligibility_criteria": {"type": "text", "analyzer": "english_custom"},
            "all_text": {"type": "text", "analyzer": "english_custom"},  # catch-all
            "phase": {"type": "keyword"},         # exact match for filtering
            "status": {"type": "keyword"},        # exact match for filtering
            "enrollment": {"type": "integer"},
        }
    },
}
```

**Reading this config:** The `"analyzer"` section defines how Elasticsearch processes text before indexing. Each word goes through three steps: (1) `"standard"` tokenizer splits text into words, (2) `"lowercase"` converts to lowercase, (3) `"english_stemmer"` reduces words to roots ("treatments" -> "treatment"), and (4) `"english_stop"` removes common words like "the" and "is". The `"mappings"` section tells Elasticsearch which fields are full-text searchable (`"type": "text"`) vs exact-match only (`"type": "keyword"`). Phase and status are keywords because you'd never search for "Phase 3-ish" — it's either Phase 3 or it's not. The `"all_text"` field is a catch-all: it concatenates all text fields into one blob so a term mentioned only in eligibility criteria still gets found.

**Elasticsearch runs in Docker.** You start it with `docker start es` (or `docker compose up elasticsearch`). It runs on `http://localhost:9200` and stores the index data in a Docker volume.

**`create_index()`** — Creates the Elasticsearch index with the settings above. If the index already exists, it deletes it first and rebuilds from scratch.

**`index_trials(trials: list[Trial])`** — Takes all 140,723 Trial objects and sends them to Elasticsearch in batches of 5,000 using the `bulk()` helper. Each trial is converted to a document dict by the `_trial_to_action()` helper method, which maps `Trial` fields to Elasticsearch fields and builds the `all_text` catch-all by concatenating title + summary + conditions + interventions + eligibility. Batching is important: sending 140K documents one at a time would mean 140K individual HTTP requests, each with network overhead. Batching groups 5,000 into one request, reducing the total to 28 requests.

**`search(query, filters, top_k)`** — The main search function. It builds a `multi_match` query with `best_fields` type and `tie_breaker=0.3`. Here's what each part means:

- `multi_match`: search across all text fields simultaneously
- `best_fields`: the final score comes from whichever field gave the highest score (so a title match beats a summary match)
- `tie_breaker=0.3`: if a document matches in multiple fields, add 30% of the non-best field scores. This rewards documents that match across multiple fields without letting weak matches override strong ones
- Field boosting is done by listing the fields as: `["title^3", "conditions^2", "interventions", "brief_summary", "eligibility_criteria", "all_text"]` — the `^3` means "multiply this field's score by 3"

It also supports `filters` — you can restrict results to only "RECRUITING" trials or only "Phase 3" trials. Filters are implemented as Elasticsearch `"term"` queries inside a `bool.filter` clause, which means they remove non-matching documents AFTER scoring. They don't affect BM25 relevance — they just exclude.

**`_trial_to_action(trial)`** — A private helper that converts a `Trial` object to an Elasticsearch bulk-insert document. It:
- Joins `conditions` list with `" ; "` (semicolons) for the text field
- Joins `interventions` list with `" ; "`
- Constructs the `all_text` catch-all field by concatenating: `title + conditions + brief_summary + eligibility_criteria + interventions`
- Sets the `_id` to `nct_id` for unique identification

**`get_trial(nct_id)`** — Fetches a single trial by its NCT ID. This is used later in the pipeline when semantic search finds a trial and we need its metadata (title, conditions, phase, etc.) from Elasticsearch. Since semantic search only stores NCT IDs in the FAISS index (not metadata), we need Elasticsearch to look up the full trial details.

### Why BM25 Is the Unsung Hero

Throughout all our experiments, BM25 consistently delivers strong first-result quality. The MRR (how quickly users find a relevant result) is 0.768-0.912 across all evaluations, meaning a relevant trial almost always appears in the first 1-2 positions. BM25 handles exact term matching perfectly — if a patient searches for "pembrolizumab," BM25 finds every trial containing that exact word. The later stages improve the ranking of positions 2-10, but the first relevant result usually comes from BM25.

---

## 6. Stage 2: Semantic Search — Embeddings, FAISS, and the Anisotropy Problem

### What Is an Embedding?

An **embedding** is a way to represent text as a list of numbers (a "vector") such that texts with similar meanings have similar number lists. For example:

- "breast cancer chemotherapy" → [0.12, -0.34, 0.78, ..., 0.45] (768 numbers)
- "breast carcinoma treatment" → [0.11, -0.35, 0.77, ..., 0.44] (very similar numbers!)
- "basketball scores" → [-0.89, 0.12, -0.03, ..., -0.67] (very different numbers)

An **embedding model** is a neural network that has been trained to produce these number lists. We use **BioLinkBERT-base**, which was pre-trained on millions of biomedical research papers AND the citation links between them. This means it knows that "glioblastoma" and "temozolomide" are related (because papers about glioblastoma frequently cite papers about temozolomide), even though the words look nothing alike.

### What Is Cosine Similarity?

Once we have embeddings (lists of numbers) for both a query and a trial, we need to measure how similar they are. **Cosine similarity** measures the angle between two vectors:

- **1.0** = identical direction = maximum similarity
- **0.0** = perpendicular = no similarity
- **-1.0** = opposite direction = opposite meaning

We normalize all our vectors to length 1.0 (called "L2 normalization"), which means the cosine similarity equals the **dot product** (multiply corresponding numbers and add them up). This is important because it allows us to use FAISS's inner product search.

### What Is FAISS?

**FAISS** (Facebook AI Similarity Search) is a library for fast nearest-neighbor search on vectors. Given a query vector, it finds the most similar vectors in a collection.

We use **`IndexFlatIP`** (Flat Index, Inner Product). "Flat" means it does a brute-force scan through all 140,000 vectors — no shortcuts, no approximation. "Inner Product" means it computes the dot product between the query and every stored vector. At 140K vectors, this takes about 37 milliseconds, which is fast enough for our purposes. (At 1 million+ vectors, we'd need approximate methods, but 140K is fine for exact search.)

### Key Functions

**`src/TrialMine/models/embeddings.py` — `TrialEmbedder` class**

This class wraps the embedding model.

**`__init__(model_name)`** — Loads the BioLinkBERT model. But there's a critical subtlety here. Let's look at the actual code:

```python
def __init__(self, model_name="michiyasunaga/BioLinkBERT-base", device="cpu"):
    if self._needs_explicit_modules(model_name):
        # Raw HuggingFace model — wire up manually to avoid crash
        from sentence_transformers.models import Pooling, Transformer

        word_model = Transformer(model_name)              # loads the BERT weights
        pooling = Pooling(
            word_model.get_word_embedding_dimension(),     # 768 for BioLinkBERT
            pooling_mode_mean_tokens=True,                 # average all token embeddings
        )
        self.model = SentenceTransformer(
            modules=[word_model, pooling], device=device   # explicit pipeline
        )
    else:
        self.model = SentenceTransformer(model_name, device=device)  # normal load
```

**Why this matters:** BioLinkBERT is a raw HuggingFace model — it was not packaged for the `sentence-transformers` library. Sentence-transformers expects a `modules.json` file that tells it how to turn a model into an embedding pipeline (which layer to use, how to pool token embeddings into a sentence embedding). Without `modules.json`, sentence-transformers tries to auto-detect, but for BioLinkBERT, this auto-detection **silently misconfigures** the model. The result is a **SIGSEGV** — a segmentation fault that kills the entire Python process instantly. You can't catch it with try/except because it happens at the C level, below Python's exception handling.

**The fix:** `_needs_explicit_modules()` checks whether `modules.json` exists (locally or on HuggingFace Hub). If it doesn't, we bypass auto-detection entirely by manually creating the two components: (1) `Transformer` — loads the BERT weights and converts text to 768-dimensional token-level embeddings (one vector per word), and (2) `Pooling` with `pooling_mode_mean_tokens=True` — averages all the token vectors into a single 768-dimensional sentence vector. This explicit wiring produces the same result as a properly-configured model, just without the buggy detection step.

**When our fine-tuned model loads:** After fine-tuning, our model DOES have `modules.json` (sentence-transformers saves it), so it takes the `else` branch. The fix only activates for the off-the-shelf BioLinkBERT.

**`embed_text(text)`** — Takes a string and returns a 768-dimensional normalized vector. Here's the actual code:

```python
def embed_text(self, text):
    embedding = self.model.encode(
        text,
        normalize_embeddings=True,   # L2 normalize: scale to length 1.0
        show_progress_bar=False,
    )
    return np.asarray(embedding, dtype=np.float32)
```

**What happens inside `encode()`:**
1. **Tokenize:** The text "breast cancer" becomes token IDs `[101, 7318, 4456, 102]` (numbers the model understands)
2. **Forward pass:** These tokens go through 12 layers of the BERT transformer, producing 768 numbers per token
3. **Mean pooling:** Average all token vectors into a single 768-number vector
4. **Normalize:** Scale the vector so its length (sqrt of sum of squares) equals exactly 1.0

After normalization, the dot product between any two vectors equals their cosine similarity. This is why we use FAISS `IndexFlatIP` (inner product) — it's computing cosine similarity.

**`embed_batch(texts, batch_size=64)`** — Same as `embed_text` but processes 64 texts at once. We use this when building the FAISS index — embedding all 140,000 trials. Processing 64 at a time is efficient because the model does matrix multiplication internally, and large matrices are faster per-element than small ones (GPU/CPU parallelism). Building the full index takes about 2-3 hours on CPU.

**`prepare_trial_text(trial)`** — Combines a trial's title, conditions, and summary into a single string. Here's the actual code:

```python
def prepare_trial_text(self, trial):
    parts = []
    if trial.title:
        parts.append(trial.title)
    if trial.conditions:
        parts.append(" ".join(trial.conditions))   # list -> single string
    if trial.brief_summary:
        parts.append(trial.brief_summary)

    text = " [SEP] ".join(parts) if parts else ""

    max_chars = 2048  # ~512 tokens (BERT's max input length)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text
```

**Example output:** For a trial about EGFR lung cancer:
```
"A Phase 3 Study of Osimertinib in EGFR-Mutated NSCLC [SEP] Non-Small Cell Lung Cancer
 [SEP] This study evaluates the efficacy of osimertinib as first-line treatment..."
```

The `[SEP]` token is a special marker that BERT models recognize as a separator between segments. It tells the model "these are related but distinct pieces of text." We truncate at 2048 characters because BERT has a hard limit of 512 tokens (roughly 4 characters per token), and exceeding it causes an error.

**`src/TrialMine/retrieval/semantic.py` — `FAISSIndex` class**

**`build(embeddings, trial_ids)`** — Takes the 140K embedding vectors (each 768 numbers) and the corresponding NCT IDs, L2-normalizes the vectors, and adds them to the FAISS index:

```python
def build(self, embeddings, trial_ids):
    faiss.normalize_L2(embeddings)              # normalize all vectors to length 1.0
    self.index = faiss.IndexFlatIP(dimension)   # create index for inner product search
    self.index.add(embeddings)                  # add all 140K vectors
    self.trial_ids = trial_ids                  # store NCT IDs for lookup
```

`IndexFlatIP` means: **Flat** = exact search (no approximation), **IP** = Inner Product (which equals cosine similarity for normalized vectors). The index stores all 140K vectors in a flat array and does a brute-force scan for each query. This is simple but sufficient — scanning 140K 768-dim vectors takes only ~37ms.

**`search(query_embedding, top_k=200)`** — Given a query embedding, finds the top 200 most similar trial vectors:

```python
def search(self, query_embedding, top_k=200):
    query = np.expand_dims(query_embedding, 0)  # FAISS expects a 2D array
    faiss.normalize_L2(query)
    scores, indices = self.index.search(query, top_k)  # inner product search
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:  # FAISS returns -1 for missing results
            results.append((self.trial_ids[idx], float(score)))
    return results
```

Returns a list of `(nct_id, similarity_score)` tuples, sorted by score descending. The scores range from ~0.0 to ~1.0 (after fine-tuning; before fine-tuning, they were all clustered around 0.87).

**`save()` / `load()`** — Persists the index to disk and loads it back. The FAISS index file is 412 MB (140K vectors x 768 dimensions x 4 bytes per float32). Without saving, we'd need to re-embed all 140K trials every time we restart the system (which takes hours).

### The Anisotropy Discovery

When we first ran semantic search with the off-the-shelf (un-fine-tuned) BioLinkBERT, the results looked terrible. We diagnosed the problem by measuring the **cosine similarity range** — the difference between the highest-scoring and lowest-scoring results:

| Metric | Value | What It Means |
|--------|-------|--------------|
| Cosine range across 1000 results | **0.047** | The most relevant and least relevant trials score almost identically |
| Score range | 0.8517 to 0.8988 | Everything clusters around 0.87 |
| Hub trials (appear in >25% of queries) | **3 trials** | 3 generic trials monopolize 33% of all result slots |
| Top-3 overlap with BM25 | **0%** | BM25 and semantic find completely different trials |

**What is anisotropy?** In plain English: all the embeddings are pointing in nearly the same direction. Imagine 140,000 arrows in a 768-dimensional space. Instead of pointing in all different directions (which would let cosine similarity distinguish them), they're all clustered in a narrow cone. The cosine similarity between ANY two embeddings is around 0.87, whether they're relevant to each other or not.

**What is a hub trial?** When all embeddings cluster in one direction, the embedding that's closest to the center of the cluster appears as the "nearest neighbor" for almost every query. Three trials — "Mebendazole in Glioblastoma," "R2 Follicular Lymphoma," and "Efficacy Prediction Model" — appeared in the top results for 8, 6, and 6 of our 20 test queries respectively. That's a third of all result slots occupied by three generic trials, regardless of what the patient searched for.

**The diagnosis:** BioLinkBERT **understands** biomedical concepts (we verified that "come back" and "recurrent" map to similar regions, and there's 30% overlap in the top-200 between BM25 and semantic for paraphrase queries). But it was never trained to **rank documents** by relevance. Pre-training gives knowledge; contrastive fine-tuning (which we do in Section 9) gives ranking behavior.

---

## 7. Stage 3: Hybrid Search — Combining BM25 and Semantic with RRF

### Why Combine Two Search Methods?

BM25 and semantic search find **completely different** relevant trials (0% top-3 overlap). This means they're complementary:

- BM25 excels at **exact terms**: drug names, gene mutations (EGFR, BRCA1), specific cancer types
- Semantic excels at **meaning**: "chemo stopped working" matches "failed prior chemotherapy"

By combining them, we get the best of both worlds.

### What Is Reciprocal Rank Fusion (RRF)?

RRF is a method for combining two ranked lists into one. The formula is simple:

```
RRF_score(document) = sum( 1 / (k + rank_in_list) ) for each list containing the document
```

Where `k = 60` is a smoothing constant (from Cormack et al., 2009 — the original RRF paper). Let's walk through an example:

**Suppose Trial X is:**
- Rank 1 in BM25
- Rank 5 in semantic

```
RRF_score = 1/(60 + 1) + 1/(60 + 5) = 1/61 + 1/65 = 0.01639 + 0.01538 = 0.03177
```

**Suppose Trial Y is:**
- Rank 1 in BM25
- Not found by semantic at all

```
RRF_score = 1/(60 + 1) = 0.01639
```

Trial X (found by both methods) gets a higher score than Trial Y (found by only one). This is the core idea: **trials found by both methods are boosted.**

**Why RRF instead of just averaging scores?** BM25 scores range from about 5 to 50. Cosine similarity ranges from about 0.85 to 0.90. If we tried to average them, the BM25 scores would completely dominate. We'd need to normalize both to [0, 1], but the right normalization is hard to pick and changes with different queries. RRF avoids this problem entirely because it uses **ranks** (positions 1, 2, 3, ...) instead of raw scores. Ranks are always comparable regardless of the score scale.

### Key Functions

**`src/TrialMine/retrieval/hybrid.py` — `reciprocal_rank_fusion()`**

This is the core fusion algorithm. Let's look at the actual code:

```python
def reciprocal_rank_fusion(bm25_results, semantic_results, k=60):
    scores = {}          # nct_id -> accumulated RRF score
    bm25_ranks = {}      # nct_id -> rank in BM25 list
    semantic_ranks = {}   # nct_id -> rank in semantic list

    # Pass 1: BM25 contributions
    for rank, result in enumerate(bm25_results, start=1):
        nct_id = result["nct_id"]
        scores[nct_id] = scores.get(nct_id, 0.0) + 1.0 / (k + rank)
        bm25_ranks[nct_id] = rank

    # Pass 2: Semantic contributions
    for rank, (nct_id, _score) in enumerate(semantic_results, start=1):
        scores[nct_id] = scores.get(nct_id, 0.0) + 1.0 / (k + rank)
        semantic_ranks[nct_id] = rank

    # Build fused list with source tags
    fused = []
    for nct_id, rrf_score in scores.items():
        bm25_rank = bm25_ranks.get(nct_id)
        semantic_rank = semantic_ranks.get(nct_id)

        if bm25_rank is not None and semantic_rank is not None:
            source = "both"           # found by both methods
        elif bm25_rank is not None:
            source = "bm25_only"      # only found by keywords
        else:
            source = "semantic_only"  # only found by meaning

        fused.append({
            "nct_id": nct_id,
            "rrf_score": rrf_score,
            "bm25_rank": bm25_rank,
            "semantic_rank": semantic_rank,
            "source": source,
        })

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused
```

**Walking through it step by step:**

1. **Pass 1** loops through BM25 results. If a trial is BM25 rank 1, it gets score `1/(60+1) = 0.01639`. If it's rank 10, it gets `1/(60+10) = 0.01429`. The `scores.get(nct_id, 0.0)` means "get the existing score, or start at 0."

2. **Pass 2** does the same for semantic results. Here's the magic: if a trial already got a score in Pass 1 (because BM25 found it), Pass 2 **adds** to that score. A trial at BM25 rank 1 and semantic rank 5 gets: `1/61 + 1/65 = 0.01639 + 0.01538 = 0.03177`.

3. A trial found by ONLY BM25 at rank 1 gets just `0.01639` — almost half the score of one found by both methods. This is how RRF rewards agreement between the two methods.

4. The **source tags** are used later by the UI to show users whether a result was found by "keywords," "meaning," or "both."

**Concrete example:** BM25 returns [NCT001, NCT002, NCT003] and semantic returns [NCT002, NCT004, NCT001]:

| Trial | BM25 Rank | Semantic Rank | RRF Score | Source |
|-------|-----------|---------------|-----------|--------|
| NCT002 | 2 | 1 | 1/62 + 1/61 = 0.03253 | both |
| NCT001 | 1 | 3 | 1/61 + 1/63 = 0.03226 | both |
| NCT003 | 3 | — | 1/63 = 0.01587 | bm25_only |
| NCT004 | — | 2 | 1/62 = 0.01613 | semantic_only |

NCT002 ranks first because it was found early by both methods.

**`HybridRetriever` class**

This class orchestrates the full hybrid search.

**`__init__(bm25, semantic, embedder)`** — Takes three components: the Elasticsearch index, the FAISS index, and the embedding model. These are the building blocks of stages 1-3.

**`search(query, top_k=50, candidate_k=200)`** — The standard hybrid search:
1. Get 200 candidates from BM25 (`self.bm25.search()`)
2. Embed the query, get 200 candidates from FAISS (`self.semantic.search()`)
3. Merge with RRF (`reciprocal_rank_fusion()`)
4. Take the top 50 results
5. For each result, attach metadata (title, conditions, phase, status) from the BM25 results
6. For semantic-only results that lack metadata, fetch it from Elasticsearch using `get_trial()`

**`full_pipeline(query, reranker, blender, top_k=20, rerank_top_k=50)`** — This is **the most important function in the entire codebase.** It runs ALL 5 stages in sequence. We'll walk through it in detail in Section 14. For now, just know it exists.

### Results

Hybrid search with RRF achieves NDCG@5 = 0.816 on the original 20 evaluation queries, compared to 0.789 for BM25 alone and 0.703 for semantic alone. The combination is better than either method individually.

---

## 8. Training Data: Teaching the Model to Search

### Why Do We Need Training Data?

As we discovered in Section 6, BioLinkBERT understands biomedical concepts but can't rank documents (the anisotropy problem). We need to **fine-tune** the model — adjust its internal weights — so it produces embeddings that are spread out and meaningful for search.

To fine-tune, we need **training data**: examples of "this query should match this trial, not that trial." This is called **contrastive learning** — teaching the model by showing it contrasts (good match vs. bad match).

### What Is a Triplet?

Each training example is a **triplet**: `(query, positive_trial, negative_trial)`.

- **query**: a search query like "breast cancer immunotherapy"
- **positive_trial**: a trial that IS relevant to this query
- **negative_trial**: a trial that is NOT relevant to this query

The model learns: "make the query embedding closer to the positive trial embedding and farther from the negative trial embedding."

### The Three Sources

We created training data from three complementary sources. Each source teaches the model something different:

**Source 1: Metadata Pairs (242,000 pairs)**

We automatically extracted queries from trial metadata. A trial with `conditions=["breast cancer"]` and `interventions=["pembrolizumab"]` generates queries like "breast cancer," "pembrolizumab," and "breast cancer pembrolizumab." These are robotic-sounding queries, but there are 242,000 of them — enough for the model to learn basic concept-trial associations.

**Source 2: Synthetic Patient Queries (1,500 via Claude Haiku)**

We used Claude Haiku (a fast AI model) to generate realistic patient queries from trial descriptions. Instead of the robotic "breast cancer pembrolizumab," Claude generates:

> "My mother was just diagnosed with triple-negative breast cancer — are there clinical trials for new immunotherapy drugs she could try?"

These 1,500 queries teach the model the vocabulary gap between clinical language and patient language. They cost about $1.50 total to generate.

**Source 3: Hard Negatives (730,000 triplets)**

A **hard negative** is a trial that looks similar to the positive trial but isn't actually relevant to the query. For example:

- **Query**: "breast cancer pembrolizumab"
- **Positive**: A breast cancer trial studying pembrolizumab
- **Hard negative**: A breast cancer trial studying tamoxifen (same cancer, different drug)

Without hard negatives, the model only sees random negatives (a lung cancer trial as a negative for a breast cancer query), which is too easy. The model just learns "is this about the right cancer?" Without hard negatives, it never learns "is this the right TREATMENT for the right cancer?"

We mine hard negatives by building an in-memory index. Here's how:

```
Step 1: Build a condition-word index
  "breast" → [NCT001, NCT002, NCT003, NCT004, ...]
  "lung"   → [NCT010, NCT011, NCT012, ...]
  "cancer" → [NCT001, NCT002, NCT003, NCT010, ...]  (almost all trials)

Step 2: For query "breast cancer pembrolizumab" + positive trial NCT001:
  - Find trials sharing "breast" AND "cancer" keywords → [NCT002, NCT003, NCT004]
  - Filter out trials with the SAME intervention as NCT001 (pembrolizumab)
  - Keep trials with DIFFERENT interventions → [NCT003 (tamoxifen), NCT004 (trastuzumab)]
  - Randomly sample 3 hard negatives from this pool
```

The key insight: NCT003 (breast cancer + tamoxifen) is a much harder negative than a random leukemia trial. The model must learn "not just right cancer, but right treatment for that cancer."

### Key Functions

**`scripts/generate_training_data.py`** — The script that generates all training data.

- **Stratified sampling**: Uses a taxonomy of 23 cancer types defined in `configs/training_data.yaml`. Each cancer type is defined by keywords (e.g., "lung" → lung cancer trials). A 2,000-trial cap per group prevents breast cancer (15K trials) from dominating. Without this cap, the model would learn "breast cancer" embeddings perfectly but fail on rare cancers like neuroblastoma (361 trials).

- **Metadata pair generation**: Loops through sampled trials, extracts conditions, interventions, and phases, and creates query-trial pairs.

- **Synthetic queries via Claude Haiku API**: For each trial in a random sample, sends a prompt to Claude asking it to write a realistic patient query. The API calls are **resumable** — if the process crashes, it picks up where it left off using a checkpoint file. Rate limited to avoid hitting API limits.

- **Hard negative mining**: Builds a `condition_word → [nct_ids]` index. For each (query, positive_trial) pair, finds trials sharing condition keywords but with different interventions. Samples 3 negatives per positive.

**Output**: 586,000 training triplets in `data/training/train_pairs.jsonl` and 145,000 validation triplets in `data/training/val_pairs.jsonl` (an 80/20 split).

**Critical design: splitting by NCT ID, not by row.** A naive random split would put some triplets from the same trial in training and others in validation. The model could memorize trial-specific patterns and score well on validation without actually learning to generalize. Instead, `split_and_save()` collects all NCT IDs, randomly assigns each ID to either train or val, and ensures ALL triplets for a given trial go to the same split:

```python
nct_ids = list({t["nct_id"] for t in triplets})
rng.shuffle(nct_ids)
val_ncts = set(nct_ids[:int(len(nct_ids) * val_fraction)])
val_triplets = [t for t in triplets if t["nct_id"] in val_ncts]
train_triplets = [t for t in triplets if t["nct_id"] not in val_ncts]
```

This prevents **data leakage** — a common pitfall in ML that inflates validation scores.

**The cancer type taxonomy** in `configs/training_data.yaml` defines 23 cancer groups, each with multiple keyword aliases:

```yaml
cancer_types:
  breast: ["breast", "mammary"]
  lung: ["lung", "NSCLC", "SCLC", "non-small cell", "small cell lung"]
  melanoma: ["melanoma"]
  sarcoma: ["sarcoma", "GIST", "Ewing"]
  neuroblastoma: ["neuroblastoma"]
  # ... 23 groups total
```

The `classify_cancer_type()` function checks each trial's conditions against these keywords (case-insensitive). Trials that don't match any group go into an "other" bucket. This taxonomy drives the stratified sampling that ensures rare cancers (neuroblastoma, 361 trials) are represented proportionally alongside common ones (breast, 15,000 trials).

---

## 9. Fine-Tuning the Bi-Encoder

### What Is Fine-Tuning?

**Fine-tuning** means taking a model that was pre-trained on a general task (BioLinkBERT was pre-trained to understand biomedical language) and training it further on your specific task (in our case, ranking clinical trials by relevance to patient queries).

The model's architecture doesn't change — the same neural network structure with the same 110 million parameters. But the parameter values (the numbers inside the model that determine its behavior) get adjusted through training on our 586K triplets.

### What Is a Bi-Encoder?

There are two ways to use a model for comparing queries and documents:

- **Bi-encoder**: Encode the query and the document **separately** into embeddings, then compare them with cosine similarity. Fast because you can pre-compute all document embeddings once and reuse them for every query.

- **Cross-encoder**: Feed the query AND document **together** into the model and get a single relevance score. Slower because you can't pre-compute anything — every query-document pair requires a fresh forward pass. But more accurate because the model can see the interaction between query words and document words.

We fine-tune BioLinkBERT as a bi-encoder here (and as a cross-encoder in Section 11).

### The Two-Tower Model Pattern

Our bi-encoder is an instance of the **two-tower model** — the canonical architecture name used in recommendation systems (recsys) and information retrieval at companies like Google, Meta, and Spotify. If you interview for any ranking or recommendation role, you'll be expected to know this pattern.

**The two towers:**

```
       Query Tower                    Item Tower
       (runs at query time)           (runs offline, once per trial)
       ┌──────────────┐              ┌──────────────┐
       │ BioLinkBERT  │              │ BioLinkBERT  │
       │ (same model) │              │ (same model) │
       └──────┬───────┘              └──────┬───────┘
              │                              │
              v                              v
       768-dim query                  768-dim trial
       embedding                      embedding
              │                              │
              └──────────┬───────────────────┘
                         │
                    dot product
                   (cosine sim)
                         │
                         v
                  relevance score
```

**Tower 1 — Query tower** (online): Encodes the user's query into a 768-dimensional vector. Runs once per search request at serving time. In our pipeline: `self.embedder.embed_text(query)` in `semantic.py` — takes ~15ms.

**Tower 2 — Item tower** (offline): Encodes each trial's text into a 768-dimensional vector. Runs once per trial during index building, NOT at query time. In our pipeline: `scripts/build_index.py` pre-computes all 140,723 trial embeddings and stores them in the FAISS index. This takes 2-3 hours but only happens when trials are added or the model is retrained.

**The key asymmetry — why two-tower is fast:**

The item tower has already run by the time a user searches. All 140,723 trial embeddings are sitting in FAISS, ready for comparison. At query time, we:
1. Run the query tower ONCE (15ms for one forward pass through BioLinkBERT)
2. Do FAISS search: compare the query embedding to all 140,723 stored trial embeddings via dot product (~22ms)
3. Return the top 200 most similar trials

Total: ~37ms for 140,723 candidates. This is possible because dot product is just multiplication and addition — no neural network inference needed for the items.

**Contrast with the cross-encoder (NOT a two-tower model):**

The cross-encoder processes query and trial TOGETHER in a single BERT pass. There's no way to pre-compute anything — each (query, trial) pair is a unique input:

```
Cross-encoder (single tower, must run for every pair):
  Input: "[CLS] lung cancer treatment [SEP] A Phase 3 Study of... [SEP]"
    → Single BERT forward pass → relevance score

  50 candidates × 80ms per forward pass = ~4,000ms
```

| Property | Two-Tower (bi-encoder) | Cross-Encoder |
|---|---|---|
| Pre-computation | Yes (item tower runs offline) | No (every pair is unique) |
| Scoring 140K items | ~37ms (dot products) | ~3 hours (140K forward passes) |
| Scoring 50 items | <1ms (50 dot products) | ~4,000ms (50 forward passes) |
| Can use FAISS/ANN | Yes (embeddings stored in vector index) | No (no pre-computed embeddings) |
| Accuracy | Lower (query and trial don't interact) | Higher (full attention between query and trial) |

This is exactly why the pipeline uses two-tower for retrieval (scan 140K trials fast) and cross-encoder for re-ranking (score 50 candidates accurately). The two-tower model sacrifices accuracy for speed; the cross-encoder sacrifices speed for accuracy. The multi-stage pipeline gets both.

**Two-tower in production recsys:**

At Google, YouTube uses a two-tower model for candidate generation: the query tower encodes user features (watch history, demographics), the item tower encodes video features (title, creator, topics), and ANN search finds the top ~1000 candidate videos from billions. Then a more expensive ranking model scores those 1000 candidates. Our pipeline follows the same pattern at clinical trial scale.

**Training a two-tower model:** Both towers share the same BioLinkBERT model (shared weights). During training, contrastive loss (MNRL) adjusts the model so that query and matching trial embeddings have high dot product, while query and non-matching trial embeddings have low dot product. After training, the query tower and item tower produce embeddings in a shared space where similarity corresponds to relevance.

### What Is MultipleNegativesRankingLoss (MNRL)?

MNRL is the loss function we use for training. A **loss function** tells the model how wrong it is — the model adjusts its parameters to minimize the loss.

Here's how MNRL works with a batch of 32 triplets:

1. Encode all 32 queries → 32 query embeddings
2. Encode all 32 positive trials → 32 positive embeddings
3. Encode all 32 hard negatives → 32 negative embeddings
4. For **each** query, compute cosine similarity with:
   - Its own positive trial (should be HIGH)
   - The other 31 queries' positive trials (these are **in-batch negatives** — they should be LOW because they're someone else's relevant trial)
   - Its hard negative (should be LOW)
5. Apply softmax to get a probability distribution over all 33 options (1 positive + 31 in-batch + 1 hard)
6. The loss penalizes when the positive trial doesn't have the highest probability

**What is softmax?** Softmax converts a list of raw similarity scores into probabilities that add up to 1.0. If your query has cosine similarity [0.9, 0.3, 0.1, 0.2] with 4 trials, softmax turns this into something like [0.65, 0.12, 0.08, 0.15]. The highest score dominates. The model's goal is to make the positive trial's probability as close to 1.0 as possible.

**What does `scale=20.0` do?** Before softmax, we multiply all similarities by 20. This sharpens the distribution. Without scaling:
- Similarities [0.9, 0.3] → softmax → [0.65, 0.35] (not very decisive)
- Scale by 20: [18.0, 6.0] → softmax → [0.9999, 0.0001] (very decisive)

A sharper distribution means the model gets penalized more harshly when the positive trial isn't clearly the top choice. This forces it to learn stronger distinctions.

This gives us **33 contrasts per training step** — far more efficient than triplet loss, which only sees 1 positive and 1 negative per step. More contrasts = faster learning = better model.

### Key Functions

**`scripts/finetune_embeddings.py`** — The training script. Let's walk through its key parts:

- **Device detection**: The `detect_device()` function checks for GPU availability:
  ```python
  def detect_device():
      if torch.cuda.is_available():
          return "cuda"         # NVIDIA GPU (Colab, cloud)
      elif torch.backends.mps.is_available():
          return "mps"          # Apple Silicon GPU (M1/M2 Mac)
      return "cpu"              # fallback (very slow for training)
  ```
  If only CPU is detected, the script automatically reduces epochs to 1 and prints a warning with estimated training time (~1.5s/step on CPU vs ~0.03s/step on A100).

- **Data loading**: Reads the 586K triplets from JSONL files and loads them as a HuggingFace `Dataset` (a library for efficient data handling that supports memory mapping — it doesn't load the entire 1GB file into RAM at once). **Important filtering step**: rows with empty `negative` fields are removed, and columns are renamed from `query` to `anchor` (sentence-transformers convention).

- **Model initialization**: Loads `michiyasunaga/BioLinkBERT-base` as a `SentenceTransformer`, using our explicit `Transformer + Pooling` wiring to avoid the SIGSEGV bug.

- **Loss function**: `MultipleNegativesRankingLoss(model, scale=20.0)` with 3-column input (anchor, positive, negative).

- **Evaluator**: The `build_evaluator()` function sets up an `InformationRetrievalEvaluator` from the validation data. It subsamples to 5,000 examples for speed and configures it to compute NDCG@10, MRR@10, Recall@10, and accuracy@{1,3,5,10} during training. The model checkpoint with the best `val-retrieval_cosine_ndcg@10` is saved as the final model.

- **`SentenceTransformerTrainer`**: A wrapper around HuggingFace's `Trainer` class that handles training loops, checkpointing, mixed precision (fp16), and logging to MLflow.

- **Metadata saving**: After training, `save_metadata()` saves a JSON file alongside the model with: training date, base model name, dataset sizes, all hyperparameters, device used, and final evaluation metrics. This makes the model self-documenting — you can always check what settings produced it.

**`configs/training/embeddings.yaml`** — The training configuration:

| Parameter | Value | What It Means |
|-----------|-------|--------------|
| `learning_rate` | 2e-5 | How big each parameter update step is. Too high = unstable, too low = slow |
| `batch_size` | 32 | How many triplets per training step. Directly affects in-batch negatives (31) |
| `epochs` | 3 | How many times we go through all 586K triplets |
| `warmup_ratio` | 0.1 | For the first 10% of training, gradually increase the learning rate from 0. Prevents early instability |
| `fp16` | true | Use half-precision floating point — halves memory usage with negligible accuracy loss |

**Why A100 GPU?** MNRL does 3 forward passes per step (query, positive, negative). With batch_size=32 and sequence length 512, this requires about 30GB of GPU memory. The A100 has 40GB. The T4 (a cheaper GPU) has only 15GB — we'd need to reduce batch_size to 4, which reduces in-batch negatives from 31 to 3, severely hurting training quality. We trained on Colab Pro ($10/month) with A100 for 288 minutes.

### Results

| Metric | Before Fine-Tuning | After Fine-Tuning |
|--------|-------------------|-------------------|
| Cosine range in top 5 | 0.047 (across ALL 1000 results) | 0.10 (in top 5 alone) |
| Hub trials | 3 trials in 33% of slots | 0 |
| BM25∩Semantic top-3 overlap | 0% | 7% |
| Semantic unique trials (20 queries, top 3) | 38 | 60 |
| Val NDCG@10 | n/a | 0.492 |
| Val MRR@10 | n/a | 0.426 |
| Val Recall@10 | n/a | 0.700 |

The key changes:
- **Anisotropy fixed**: Cosine scores in the top 5 now span 0.51-0.61 (was 0.85-0.90 for everything). The model can genuinely differentiate relevant from irrelevant.
- **Hub trials eliminated**: Every query now returns distinct, topically relevant results.
- **Qualitative improvement**: "bladder cancer BCG unresponsive" returns BCG-related trials. "EGFR mutated lung cancer" returns EGFR-targeted therapy trials. Before fine-tuning, both would have returned the same generic trials.

---

## 10. Evaluation: LLM-as-Judge, Bias Discovery, and the Pooling Fix

### How We Label Data

To compute NDCG and MRR, we need relevance labels: for each (query, trial) pair, a 0-3 score. We used Claude Haiku (a fast, inexpensive AI model) as a judge. Here's the actual labeling function:

```python
def label_pair(client, query, trial, eligibility):
    prompt = f"""Rate the relevance of this clinical trial to the patient query.

Patient query: {query}

Trial title: {trial.get("title", "")}
Conditions: {trial.get("conditions", "")}
Phase: {trial.get("phase", "Unknown")}
Status: {trial.get("status", "Unknown")}
Eligibility (first 500 chars): {eligibility[:500]}

Rate on a 0-3 scale:
0 = Wrong cancer type or completely irrelevant
1 = Same general area but wrong specifics (wrong drug, wrong stage)
2 = Relevant — patient could potentially be eligible
3 = Highly relevant — strong match for the patient's situation

Return JSON: {{"score": <0-3>, "reason": "<brief explanation>"}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    result = json.loads(response.content[0].text)
    return result["score"], result["reason"]
```

The function includes error handling: if Haiku returns invalid JSON (or wraps it in markdown code fences), we strip the fences and retry parsing. If parsing still fails, we return score=-1 (filtered out later). There's also a 0.1-second sleep between API calls for rate limiting.

Labeling 990 pairs cost about $2 and took about 10 minutes.

### Query ID Management

As we expanded the evaluation dataset across multiple rounds, we assigned non-overlapping ID ranges to keep things organized:

| ID Range | Source | Count | Current Use |
|----------|--------|-------|------------|
| 0-19 | `build_eval_dataset.py` | 20 queries | LightGBM training |
| 100-149 | `build_fair_eval.py` (v1) | 50 queries | LightGBM training (moved from test) |
| 200-274 | `expand_eval_data.py` | 75 queries | LightGBM training |
| 300-349 | `build_fair_eval.py` (v2) | 50 queries | **Held-out test** (never trained on) |

Total: 145 training queries + 50 held-out test queries. The ID gaps (20-99, 150-199, 275-299) are reserved for future expansion.

### The Evaluation Bias Discovery

This is one of the most important lessons in the entire project.

**What we did first (wrong):**
1. Searched using only the **fine-tuned** model, got the top 30 results per query
2. Labeled those 600 (query, trial) pairs with Claude Haiku
3. Compared both models (fine-tuned vs off-the-shelf) using those labels
4. **Reported: fine-tuned model is +123% better (NDCG@10 0.797 vs 0.357)**

**What was wrong:** When the off-the-shelf model retrieved a trial that wasn't in the fine-tuned model's top 30, that trial had no label. Our code defaulted it to **relevance 0** (irrelevant). But those trials might be very relevant! We just never labeled them.

Think of it this way: imagine you have two students taking a test, but you only grade the answers that Student A got right. Student B's unique correct answers get marked as wrong by default. Of course Student A "wins" — the test was rigged in their favor.

**The numbers:**
- 210 trials (21%) were retrieved by BOTH models
- 390 trials (39%) were retrieved ONLY by the fine-tuned model — labeled
- 390 trials (39%) were retrieved ONLY by the off-the-shelf model — **unlabeled, defaulted to 0**

Those 390 unlabeled trials tanked the off-the-shelf model's score from 0.534 (real) to 0.357 (artificially low).

**What we did to fix it (pooled evaluation):**
1. Searched using BOTH models, got top 30 from each
2. Merged and deduplicated via `pool_results()`:
   ```python
   def pool_results(results_by_model):
       seen = {}
       for model_name, results in results_by_model.items():
           for trial in results:
               nct_id = trial["nct_id"]
               if nct_id in seen:
                   seen[nct_id]["retrieved_by"] = "both"    # found by both models
               else:
                   seen[nct_id] = {**trial, "retrieved_by": model_name}
       return list(seen.values())
   ```
   This produces 990 unique (query, trial) pairs, each tagged with which model(s) found it.
3. Labeled ALL 990 pairs with Claude Haiku
4. Now both models are evaluated fairly against the same complete label set

**Corrected results:**

| Metric | Biased Eval | Pooled Eval (correct) |
|--------|------------|----------------------|
| Off-the-shelf NDCG@10 | 0.357 | **0.534** (+50% correction!) |
| Fine-tuned NDCG@10 | 0.797 | 0.796 (unchanged) |
| Reported improvement | +123% | **+49%** |

The real improvement is +49%, not +123%. This is a textbook case of **evaluation contamination**: when your evaluation methodology systematically favors one approach.

### Key Functions

**`src/TrialMine/evaluation/metrics.py`** — Our evaluation metrics. Let's look at the actual code.

**`ndcg_at_k(result_ids, relevance_scores, k)`** — Computes NDCG@k exactly as described in Section 3. Here's the actual code:

```python
def ndcg_at_k(result_ids, relevance_scores, k):
    if k <= 0 or not relevance_scores:
        return 0.0

    # DCG for the actual ranking
    dcg = 0.0
    for i, doc_id in enumerate(result_ids[:k], start=1):  # i goes 1, 2, 3, ...
        rel = relevance_scores.get(doc_id, 0.0)            # look up label; 0 if unlabeled
        dcg += (2**rel - 1) / math.log2(i + 1)             # gain / discount

    # IDCG: the perfect ranking — sort ALL labels descending, take top k
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg += (2**rel - 1) / math.log2(i + 1)

    if idcg == 0.0:       # no relevant docs at all → NDCG is 0
        return 0.0
    return dcg / idcg     # normalize: 0.0 = worst, 1.0 = perfect
```

**Walking through it with an example.** Suppose we search for "breast cancer immunotherapy" and get back `["NCT001", "NCT002", "NCT003"]`, and our labels say `{"NCT001": 0, "NCT002": 3, "NCT003": 1}`:

- Position 1 (NCT001, rel=0): gain = `(2^0 - 1) / log2(2)` = `0 / 1.0` = **0**
- Position 2 (NCT002, rel=3): gain = `(2^3 - 1) / log2(3)` = `7 / 1.585` = **4.416**
- Position 3 (NCT003, rel=1): gain = `(2^1 - 1) / log2(4)` = `1 / 2.0` = **0.5**
- DCG = 0 + 4.416 + 0.5 = **4.916**

The ideal order would be [3, 1, 0]:
- Position 1 (rel=3): `7 / 1.0` = **7.0**
- Position 2 (rel=1): `1 / 1.585` = **0.631**
- Position 3 (rel=0): `0 / 2.0` = **0**
- IDCG = 7.0 + 0.631 + 0 = **7.631**

NDCG@3 = 4.916 / 7.631 = **0.644**. Not great — the best result (NCT002) was at position 2 instead of 1.

**`mrr(result_ids, relevant_ids)`** — Scans through results until it finds the first relevant one, returns 1/position:

```python
def mrr(result_ids, relevant_ids):
    for i, doc_id in enumerate(result_ids, start=1):  # i = 1, 2, 3, ...
        if doc_id in relevant_ids:
            return 1.0 / i   # found it! Return reciprocal of position
    return 0.0                # no relevant result found at all
```

If the first relevant trial is at position 3, MRR = 1/3 = 0.333. We average MRR across all queries to get the final MRR score.

**`precision_at_k(result_ids, relevant_ids, k)`** — What fraction of the top-k results are relevant? If 3 of the top 5 results are relevant, precision@5 = 0.6. The code simply counts how many of `result_ids[:k]` appear in `relevant_ids` and divides by k.

**`recall_at_k(result_ids, relevant_ids, k)`** — What fraction of ALL relevant trials appear in the top k? If there are 10 relevant trials total and 4 appear in the top 5, recall@5 = 0.4. Divides by `len(relevant_ids)` instead of k.

**`scripts/evaluate.py` — `bootstrap_ci()` — The bootstrap confidence interval function:**

```python
def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    rng = np.random.RandomState(42)              # reproducible
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    alpha = (1 - ci) / 2                          # 0.025 for 95% CI
    ci_low = np.percentile(means, alpha * 100)    # 2.5th percentile
    ci_high = np.percentile(means, (1 - alpha) * 100)  # 97.5th percentile
    return np.mean(values), ci_low, ci_high
```

This takes a list of per-query scores (e.g., 50 NDCG values), resamples them 1000 times with replacement, and returns the mean plus the 95% confidence interval boundaries. The "with replacement" part means some queries get picked twice and some not at all in each sample — this simulates the randomness of which queries we happened to test on.

**`scripts/build_eval_dataset.py`** — The pooled evaluation dataset builder.

This script:
1. Runs hybrid search with BOTH embedding models (fine-tuned and off-the-shelf) for each of 20 queries
2. Takes the top 30 from each model, merges and deduplicates
3. For each unique (query, trial) pair, calls Claude Haiku to get a relevance label
4. Saves everything to `data/evaluation/labeled_queries.jsonl`
5. Supports **resume** via checkpoint (if it crashes, it doesn't re-label already-labeled pairs)

**`scripts/compare_embeddings.py`** — Compares the two models using the pooled labels. Runs hybrid search with each model on all 20 queries, computes NDCG@5, NDCG@10, and MRR per query, and reports which model wins on each query.

### Score Distribution

| Score | Count | Percentage | Meaning |
|-------|-------|-----------|---------|
| 0 | 225 | 22.7% | Wrong cancer type |
| 1 | 172 | 17.4% | Marginal |
| 2 | 138 | 13.9% | Relevant |
| 3 | 455 | 46.0% | Highly relevant |

46% of labels being score 3 is suspicious — it may reflect Claude Haiku being generous. Without human calibration, the absolute NDCG values shouldn't be taken as gospel. But relative comparisons (Method A vs Method B using the same labels) remain valid.

---

## 11. Stage 4: Cross-Encoder Re-Ranking

### Bi-Encoder vs Cross-Encoder: An Analogy

Think of hiring:
- **Bi-encoder** = Reading each resume separately, scoring each one, then comparing scores. Fast (you can pre-score all resumes), but you miss connections between the job description and specific resume details.
- **Cross-encoder** = Reading the job description AND resume side-by-side, looking for specific matches. Slow (you must re-read for every applicant), but much more accurate because you see the interaction.

In our pipeline, the bi-encoder (Stage 2) and BM25 (Stage 1) are the "fast scan" that narrows 140,000 trials to 50 candidates. The cross-encoder is the "careful review" that re-scores those 50 candidates.

### How the Cross-Encoder Works

The cross-encoder takes a (query, trial_text) pair and feeds BOTH into a single BioLinkBERT model simultaneously. The model outputs a single number (a **logit** — a raw, unbounded number) that represents how relevant the trial is to the query. We convert this to a 0-1 probability using the **sigmoid** function:

```
score = 1 / (1 + e^(-logit))
```

Sigmoid squishes any number into the range [0, 1]:
- Logit = -10 → score ≈ 0.00005 (very irrelevant)
- Logit = 0 → score = 0.5 (neutral)
- Logit = 10 → score ≈ 0.99995 (very relevant)

### Training the Cross-Encoder

We fine-tuned BioLinkBERT as a cross-encoder using our training triplets. Since cross-encoders take pairs (not triplets), we converted each triplet into two binary pairs:

- `(query, positive_trial)` → label 1.0 (relevant)
- `(query, negative_trial)` → label 0.0 (irrelevant)

This gave us 200,000 binary training pairs (from 100,000 subsampled triplets — we couldn't use all 586K because the full dataset would take 39 hours on a T4 GPU).

The loss function was **BinaryCrossEntropyLoss** — the standard loss for binary classification ("relevant" vs "irrelevant").

### The Critical Discovery: Pure CE Replacement HURTS

In a standard cross-encoder re-ranking pipeline, you replace the retrieval scores entirely with the cross-encoder scores. We tried this and the results were devastating:

| Approach | NDCG@5 | vs Hybrid Baseline |
|----------|--------|--------------------|
| Hybrid only (no re-ranking) | 0.816 | — |
| + Off-the-shelf ms-marco-MiniLM (pure CE) | 0.719 | **-11.9%** |
| + Fine-tuned BioLinkBERT CE (pure CE) | 0.657 | **-19.5%** |
| + Fine-tuned BioLinkBERT CE (blended 0.7 RRF + 0.3 CE) | 0.829 | **+1.6%** |

Both off-the-shelf and fine-tuned cross-encoders made results WORSE when used as the sole ranker. Why?

**Root cause: binary training labels.** Our cross-encoder was trained on binary labels (relevant=1, irrelevant=0), where "relevant" meant "correct cancer type" and "irrelevant" meant "wrong cancer type." The model learned to distinguish right disease from wrong disease, but that's ALL it learned. It can't distinguish between:

- A recruiting Phase 3 EGFR-targeted trial (highly relevant for an EGFR lung cancer patient)
- A completed Phase 1 supportive care trial for lung cancer (marginally relevant)

Both are about the right cancer → both get cross-encoder score ≈ 1.0. The nuanced ranking that RRF provided (based on multiple signals: keyword match quality, semantic similarity, field boosting) gets replaced by a blunt "right disease or not" signal.

### The Fix: Blended Scoring

Instead of replacing the RRF score with the CE score, we **blend** them:

```
blended_score = 0.7 × RRF_normalized + 0.3 × CE_sigmoid
```

- **0.7 weight on RRF**: preserves the multi-signal ranking from BM25 + semantic fusion
- **0.3 weight on CE**: allows the cross-encoder to break ties and make small corrections

The 0.7/0.3 split was chosen conservatively — the CE gets enough weight to help on hard queries but not enough to override correct RRF rankings.

Results: +1.6% NDCG@5, +3.6% MRR. The biggest gains were on the hardest queries (sarcoma +31%, glioblastoma +14%).

### Key Functions

**`src/TrialMine/models/cross_encoder.py` — `CrossEncoderReranker` class**

**`__init__(model_name)`** — Loads the cross-encoder model using `sentence_transformers.CrossEncoder`.

**`score(query, trial_texts)`** — The core scoring function. Here's the actual code:

```python
def score(self, query, trial_texts):
    if not trial_texts:
        return []
    pairs = [(query, t) for t in trial_texts]   # create all (query, trial) pairs
    scores = self.model.predict(pairs, convert_to_numpy=True)
    return scores.tolist()                        # raw logits (unbounded numbers)
```

For 50 candidates, this creates 50 pairs and feeds them all to the cross-encoder in one batch. The model returns 50 raw **logits** — numbers like [-2.1, 3.5, 0.8, ...]. A logit is an unbounded number (can be anything from -infinity to +infinity). Higher means more relevant. We later convert these to 0-1 probabilities using sigmoid.

**`rerank(query, candidates, top_k=20)`** — The complete re-ranking pipeline. Here's the actual code:

```python
def rerank(self, query, candidates, top_k=20, text_key="trial_text"):
    texts = [c[text_key] for c in candidates]
    ce_scores = self.score(query, texts)          # raw logits from cross-encoder

    # Step 1: Apply sigmoid to convert logits to [0, 1] probabilities
    import math
    for candidate, ce in zip(candidates, ce_scores):
        candidate["cross_encoder_score"] = 1 / (1 + math.exp(-ce))

    # Step 2: Normalize RRF scores to [0, 1] via min-max scaling
    rrf_scores = [c.get("score", 0.0) for c in candidates]
    rrf_max = max(rrf_scores) if rrf_scores else 1.0
    rrf_min = min(rrf_scores) if rrf_scores else 0.0
    rrf_range = rrf_max - rrf_min if rrf_max > rrf_min else 1.0

    # Step 3: Compute blended score
    for candidate in candidates:
        rrf_norm = (candidate.get("score", 0.0) - rrf_min) / rrf_range
        ce_norm = candidate["cross_encoder_score"]
        candidate["blended_score"] = 0.7 * rrf_norm + 0.3 * ce_norm

    ranked = sorted(candidates, key=lambda x: x["blended_score"], reverse=True)
    return ranked[:top_k]
```

**Walking through it with a concrete example.** Suppose we have 3 candidates:

| Candidate | RRF Score | CE Logit | CE Sigmoid | RRF Normalized | Blended |
|-----------|-----------|----------|------------|----------------|---------|
| Trial A | 0.032 | 2.5 | 0.924 | 1.0 | 0.7*1.0 + 0.3*0.924 = **0.977** |
| Trial B | 0.028 | 3.1 | 0.957 | 0.5 | 0.7*0.5 + 0.3*0.957 = **0.637** |
| Trial C | 0.024 | -1.0 | 0.269 | 0.0 | 0.7*0.0 + 0.3*0.269 = **0.081** |

Trial A stays at #1 (strong RRF + strong CE). Trial B has a higher CE score than Trial A (0.957 vs 0.924), but the 0.7 weight on RRF prevents the CE from overriding the RRF ranking. This is the key design decision: the CE can adjust rankings but not overthrow them.

**`rerank_with_timing()`** — Same as `rerank()` but wraps it in a timer. Returns `(ranked_results, elapsed_ms)`. This is how we know re-ranking takes about 4 seconds on CPU.

**`scripts/finetune_cross_encoder.py`** — The training script. Here's how the key conversion works:

```python
def load_and_convert(filepath):
    """Convert triplets to binary-labeled pairs for cross-encoder training."""
    rows = []
    for triplet in load_jsonl(filepath):
        # Each triplet becomes TWO rows:
        rows.append({"sentence1": triplet["query"], "sentence2": triplet["positive"], "label": 1.0})
        rows.append({"sentence1": triplet["query"], "sentence2": triplet["negative"], "label": 0.0})
    return Dataset.from_list(rows)
```

From 100K subsampled triplets, this produces 200K binary pairs (100K positive + 100K negative). The script uses a `CERerankingEvaluator` instead of the bi-encoder's `InformationRetrievalEvaluator` — this evaluator measures re-ranking quality (how well the model reorders a mixed list of positives and negatives) rather than retrieval quality.

The model early-stopped at NDCG@10 = 0.992 on the validation set — it's excellent at binary classification but limited by what binary labels can teach.

**`scripts/evaluate_cross_encoder.py`** — Evaluates the cross-encoder by comparing hybrid-only vs hybrid+CE on the 20 labeled queries. Reports per-query win/loss/tie.

---

## 12. Stage 5: LightGBM Metadata Blender

### Why Do We Need This?

The cross-encoder captures text relevance, but there's information about clinical trials that no text model can learn:

- A **recruiting** Phase 3 trial with 500 enrolled patients is much more useful to a patient than a **completed** Phase 1 trial with 20 patients, even if both have the same text relevance score.
- A trial that lists eligibility criteria is more actionable than one that doesn't.
- A trial whose title directly overlaps with the query words might be more relevant.

These are **metadata features** — structured information that complements the text-based scores. We need a model that can combine text scores with metadata to make a final ranking decision.

### What Is LightGBM?

**LightGBM** is a gradient boosted decision tree library. It builds many small decision trees, where each new tree tries to fix the mistakes of the previous trees. Decision trees make predictions by asking a series of yes/no questions:

```
Is cross_encoder_score > 0.8?
  Yes → Is phase_numeric >= 3?
    Yes → High relevance
    No → Is is_recruiting = 1?
      Yes → Medium relevance
      No → Low relevance
  No → Low relevance
```

LightGBM builds hundreds of these trees, and the final prediction is the sum of all trees' predictions. Here's a simplified example of how gradient boosting works:

```
Tree 1 predicts scores: Trial A = 0.5, Trial B = 0.3, Trial C = 0.8
  Actual ranking should be: C > A > B
  Error: Tree 1 got it right (C=0.8 > A=0.5 > B=0.3) ✓

Tree 2 focuses on cases Tree 1 got wrong. Suppose for a different query:
  Tree 1 predicted: Trial X = 0.6, Trial Y = 0.7
  But Y should rank below X. The "error" is that X needs +0.2, Y needs -0.2.
  Tree 2 learns these corrections: X = +0.15, Y = -0.18

Combined: X = 0.6 + 0.15 = 0.75, Y = 0.7 - 0.18 = 0.52
  Now X > Y ✓ — the second tree fixed the first tree's mistake.
```

Each new tree is trained on the **residual errors** of all previous trees combined. After 300 trees, the model has made 300 rounds of corrections, each one targeting the remaining mistakes.

### What Is LambdaRank?

**LambdaRank** is a training objective specifically designed for ranking problems. Standard machine learning objectives like "predict the relevance score accurately" (regression) don't account for **position** — getting the top-2 results in the wrong order is worse than getting results 8 and 9 in the wrong order, because users look at the top first.

LambdaRank directly optimizes NDCG by asking: "If I swapped these two documents in the ranking, how much would NDCG change?" Swaps that cause big NDCG changes get bigger gradient updates (the model learns more from important swaps).

### The 11 Features

For each (query, candidate_trial) pair, we compute 11 features:

| # | Feature | What It Measures | How It's Computed |
|---|---------|-----------------|-------------------|
| 1 | `bm25_score` | Keyword relevance | Raw BM25 score from Elasticsearch |
| 2 | `semantic_score` | Semantic relevance | Cosine similarity from FAISS |
| 3 | `cross_encoder_score` | Deep text relevance | Sigmoid of cross-encoder logit |
| 4 | `rrf_score` | Combined retrieval score | RRF fusion of BM25 + semantic |
| 5 | `phase_numeric` | Trial phase (1-4) | "Phase 3" → 3.0, "Phase 1/Phase 2" → 1.5, unknown → 0.0 |
| 6 | `is_recruiting` | Currently enrolling patients? | 1.0 if status = "RECRUITING", else 0.0 |
| 7 | `is_active` | Trial is active? | 1.0 if ACTIVE_NOT_RECRUITING or ENROLLING_BY_INVITATION |
| 8 | `enrollment_log` | How many patients enrolled | `log(1 + enrollment)` — log scale because enrollment ranges from 0 to 100,000+ |
| 9 | `condition_exact_match` | Do any query words appear in the conditions? | 1.0 if any overlap, else 0.0 |
| 10 | `title_query_overlap` | How much does the title overlap with the query? | Jaccard-like: `|query_words ∩ title_words| / |query_words|` |
| 11 | `has_eligibility` | Does the trial have eligibility criteria text? | 1.0 if eligibility text > 10 characters |

### What Is Leave-One-Query-Out Cross-Validation?

With 145 training queries, how do we know if the model generalizes? We use **leave-one-query-out cross-validation (LOOCV)**:

1. Remove query #1 from training. Train on the other 144 queries.
2. Evaluate the model on query #1.
3. Repeat for all 145 queries — each query gets a turn as the test query.
4. Average all 145 NDCG scores.

This is critical because ranking models must generalize to **unseen queries**, not unseen candidates. If we randomly split candidates (some from query #1 in training, others in test), the model could memorize query-specific patterns. LOOCV ensures each test fold sees a completely new query.

### Key Functions

**`src/TrialMine/models/ranker.py`**

**`phase_to_numeric(phase)`** — Converts phase strings to numbers. Uses a lookup table (`PHASE_MAP`): "Phase 1" -> 1.0, "Phase 3" -> 3.0, "Phase 1/Phase 2" -> 1.5 (average of the two), None -> 0.0.

**`compute_features(query, candidate, trial_doc)`** — The feature engineering function. Here's the actual code:

```python
def compute_features(query, candidate, trial_doc=None):
    query_words = set(query.lower().split())    # {"breast", "cancer", "immunotherapy"}

    # Retrieval scores — pulled directly from the candidate dict
    bm25_score = candidate.get("bm25_score", 0.0)
    semantic_score = candidate.get("semantic_score", 0.0)
    ce_score = candidate.get("cross_encoder_score", 0.0)
    rrf_score = candidate.get("score", candidate.get("rrf_score", 0.0))

    # Metadata from the trial document
    doc = trial_doc or candidate
    phase = doc.get("phase")
    status = doc.get("status", "")
    enrollment = doc.get("enrollment") or 0
    conditions = doc.get("conditions", "")
    title = doc.get("title", "")
    eligibility = doc.get("eligibility_criteria", "")

    # Derived features
    cond_words = set(conditions.lower().split()) if conditions else set()
    title_words = set(title.lower().split()) if title else set()
    condition_match = 1.0 if query_words & cond_words else 0.0   # any overlap?
    title_overlap = (
        len(query_words & title_words) / len(query_words)        # fraction of query in title
        if query_words else 0.0
    )

    return {
        "bm25_score": bm25_score,
        "semantic_score": semantic_score,
        "cross_encoder_score": ce_score,
        "rrf_score": rrf_score,
        "phase_numeric": phase_to_numeric(phase),
        "is_recruiting": 1.0 if status in RECRUITING_STATUSES else 0.0,
        "is_active": 1.0 if status in ACTIVE_STATUSES else 0.0,
        "enrollment_log": math.log1p(enrollment),      # log(1 + enrollment)
        "condition_exact_match": condition_match,
        "title_query_overlap": title_overlap,
        "has_eligibility": 1.0 if eligibility and len(eligibility) > 10 else 0.0,
    }
```

**Walking through a concrete example.** Query: `"breast cancer immunotherapy"`. Trial: a recruiting Phase 3 pembrolizumab trial with 500 patients.

```
query_words = {"breast", "cancer", "immunotherapy"}

Feature 1: bm25_score = 42.3      (raw Elasticsearch score)
Feature 2: semantic_score = 0.58   (cosine similarity)
Feature 3: cross_encoder_score = 0.91  (sigmoid of CE logit)
Feature 4: rrf_score = 0.031      (from RRF fusion)
Feature 5: phase_numeric = 3.0    ("Phase 3" -> 3.0 via PHASE_MAP)
Feature 6: is_recruiting = 1.0    (status = "RECRUITING")
Feature 7: is_active = 0.0        (RECRUITING is not in ACTIVE_STATUSES)
Feature 8: enrollment_log = 6.21  (log(1 + 500) = 6.21)
Feature 9: condition_exact_match = 1.0  ("breast" and "cancer" overlap with conditions)
Feature 10: title_query_overlap = 0.33  (1 of 3 query words in title: "breast")
Feature 11: has_eligibility = 1.0  (eligibility text > 10 chars)
```

These 11 numbers become one row in the feature matrix that LightGBM scores. The `math.log1p(enrollment)` (Feature 8) uses log scale because enrollment ranges wildly: some trials have 10 patients, others have 100,000. Without log, a 100K-patient trial would dominate the feature. Log compresses: `log(11) = 2.4`, `log(501) = 6.2`, `log(100001) = 11.5` — a much more balanced range for the tree model.

**`RankingBlender` class**

**`load(model_path)`** — Loads a trained LightGBM model from a `.lgb` file.

**`predict(features)`** — Takes a 2D numpy array of shape (n_candidates, 11) and returns predicted scores for each candidate.

**`rerank(query, candidates, top_k=20)`** — The full re-ranking pipeline. Here's the actual code:

```python
def rerank(self, query, candidates, top_k=20):
    if not candidates:
        return []

    # Step 1: Compute 11 features for every candidate
    feature_rows = []
    for c in candidates:
        feats = compute_features(query, c)
        feature_rows.append([feats[name] for name in FEATURE_NAMES])

    # Step 2: Stack into a matrix (50 candidates x 11 features)
    features = np.array(feature_rows, dtype=np.float32)

    # Step 3: LightGBM predicts a score for each row
    scores = self.predict(features)

    # Step 4: Attach scores and sort
    for candidate, score in zip(candidates, scores):
        candidate["blender_score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x["blender_score"], reverse=True)
    return ranked[:top_k]
```

**What `predict()` actually does:** LightGBM runs each of its hundreds of decision trees on the 11 features. Each tree outputs a small number (positive or negative). The final score is the **sum** of all tree outputs. A higher sum means "this candidate should rank higher." The scores have no inherent scale — only the relative ordering matters.

**`scripts/train_ranker.py`** — The training script. Here's the conceptual flow:

```python
# Step 1: Load and merge 3 label files
labels = load("labeled_queries.jsonl")       # 20 queries (IDs 0-19)
labels += load("test_labels.jsonl")          # 50 queries (IDs 100-149)
labels += load("train_labels_extra.jsonl")   # 75 queries (IDs 200-274)
# Total: 145 queries, ~6,018 labeled (query, trial) pairs

# Step 2: Feature engineering — compute 11 features per pair
for each (query, trial, label) in labels:
    features = compute_features(query, trial_candidate, trial_doc)
    # Produces one row of 11 numbers + the relevance label (0-3)

# Step 3: Save features to CSV for reproducibility
save("data/evaluation/ranking_features_v2.csv")  # 6,018 rows x 11 features

# Step 4: LOOCV — leave-one-QUERY-out, not leave-one-ROW-out
for held_out_query_id in all_145_query_ids:
    train_data = features[query_id != held_out_query_id]  # 144 queries
    test_data = features[query_id == held_out_query_id]    # 1 query
    model = lgb.train(train_data, objective="lambdarank", eval_at=[5, 10])
    ndcg_scores.append(evaluate(model, test_data))

print(f"LOOCV NDCG@5: {mean(ndcg_scores)}")  # 0.843

# Step 5: Final model trained on ALL 145 queries
final_model = lgb.train(all_features, objective="lambdarank")
final_model.save("models/ranker/v3-regularized/model.lgb")  # production path post-Phase-12
```

**The critical detail in Step 4:** We leave out an entire **query** (all 30-50 trials for that query), not a single row. If we randomly split rows, the model might see some trials from query #5 during training and other trials from query #5 during testing — it could memorize query-specific patterns. LOOCV ensures it has never seen the test query at all.

LightGBM's `lambdarank` objective requires grouping: trials belonging to the same query must be in a contiguous block, and LightGBM needs to know the group sizes. This is passed via the `group` parameter during training.

**`build_features()` — The feature engineering orchestration function** in `train_ranker.py` is worth understanding because it bridges retrieval and ML:

```python
def build_features(labels, es_index, faiss_index, embedder, reranker):
    rows = []
    for query_id, query_labels in group_by_query(labels):
        # Fetch BM25 + semantic scores for this query (top 500 each)
        bm25_scores, semantic_scores = build_score_lookups(
            query_labels[0]["query"], es_index, faiss_index, embedder, top_k=500
        )
        # Compute BM25 and semantic ranks from scores
        bm25_ranked = sorted(bm25_scores.items(), key=lambda x: -x[1])
        bm25_rank_map = {nct: r+1 for r, (nct, _) in enumerate(bm25_ranked)}

        # Score all labeled trials with cross-encoder (batch per query)
        trial_texts = [build_trial_text(get_trial(nct_id)) for nct_id in query_ncts]
        ce_raw = reranker.score(query, trial_texts)
        ce_scores = {nct: 1/(1+exp(-raw)) for nct, raw in zip(query_ncts, ce_raw)}

        # Compute RRF score from ranks
        for label in query_labels:
            nct_id = label["nct_id"]
            candidate = {
                "bm25_score": bm25_scores.get(nct_id, 0.0),
                "semantic_score": semantic_scores.get(nct_id, 0.0),
                "cross_encoder_score": ce_scores.get(nct_id, 0.0),
                "score": compute_rrf(bm25_rank_map, semantic_rank_map, nct_id),
                # ... metadata from ES
            }
            features = compute_features(query, candidate, trial_doc)
            rows.append({**features, "query_id": query_id, "relevance": label["relevance"]})
    return pd.DataFrame(rows)
```

This function runs the retrieval pipeline for each query, then computes all 11 features for each labeled (query, trial) pair. The output is saved to `data/evaluation/ranking_features_v2.csv` (6,018 rows x 11 features) for reproducibility — you can retrain LightGBM without re-running the entire retrieval pipeline.

**Feature importance is also saved as a PNG bar chart** via `save_feature_importance()`, which extracts LightGBM's gain-based importance and creates a horizontal bar plot. This is logged as an MLflow artifact for easy comparison across training runs.

### What Is Feature Importance "Gain"?

When LightGBM builds its decision trees, it needs to pick which feature to split on at each node. It picks the feature that **reduces the ranking error the most** at that split. The "gain" is the total amount of error reduction a feature contributed across ALL splits in ALL trees.

A gain of 1,673 for `cross_encoder_score` means: across all the hundreds of trees, every time LightGBM asked "which feature should I split on?", the cross-encoder score collectively provided 1,673 units of NDCG improvement. A higher gain means the model relies on that feature more heavily for ranking decisions.

### The OMP_NUM_THREADS Workaround

FAISS and LightGBM both use OpenMP for parallelism, but they conflict on macOS — both libraries try to initialize OpenMP simultaneously, causing a crash. The fix is to set `OMP_NUM_THREADS=1` before running any code that uses both libraries. This forces single-threaded execution in the OpenMP layer. The performance impact is minimal because our FAISS index fits in memory (brute-force over 140K vectors is fast even single-threaded) and LightGBM inference on 50 candidates with 11 features is instant.

### Feature Importance: The Data Scaling Story

One of the most interesting findings was how feature importance changed with more training data:

| Feature | v1 (20 queries) Gain | v2 (145 queries) Gain |
|---------|---------------------|----------------------|
| cross_encoder_score | 243 (#2) | **1,673 (#1)** |
| rrf_score | **393 (#1)** | 1,012 (#2) |
| phase_numeric | 57 | 593 (#3) |
| title_query_overlap | 45 | 557 (#4) |
| semantic_score | 93 | 520 (#5) |
| bm25_score | 132 | 516 (#6) |
| enrollment_log | 97 | 487 (#7) |

With only 20 queries, LightGBM didn't have enough data to learn that the cross-encoder score was the most informative signal. It defaulted to trusting the RRF score (which is always useful). With 145 queries (7x more data), it learned that the cross-encoder — despite being trained on binary labels — provides the single most valuable ranking signal when used as a feature rather than a standalone ranker.

**Key lesson:** The model architecture wasn't the problem — data was. With enough examples, LightGBM figured out the right way to use each signal.

---

## 13. Fair Evaluation: Testing on Queries We've Never Seen

### Why Fair Evaluation Matters

After expanding to 145 training queries, we moved the original 50 test queries into the training set (more data helps the model). But this means we need new test queries that the model has NEVER seen.

### The Test Set

We created 50 new queries that are deliberately harder than the training queries:

- **Rare cancers**: "blastic plasmacytoid dendritic cell neoplasm treatment," "systemic mastocytosis clinical trials"
- **Complex patients**: "clinical trials for pregnant women with breast cancer," "pediatric brain tumor trials for children under 5"
- **Comorbidities**: "lung cancer trials for patients with diabetes and kidney disease"
- **Health equity**: "cervical cancer trials for indigenous women in rural areas"

These queries test whether the model generalizes beyond typical "breast cancer phase 3" queries.

### Key Functions

**`scripts/build_fair_eval.py`** — Generates the 50 test queries, labels them with Claude Haiku (pooling from BM25 + semantic + hybrid to avoid bias), and runs the full ablation. Produces `docs/fair-evaluation-report.md`.

**`scripts/evaluate.py`** — The general ablation evaluation script. Runs all 5 pipeline stages on labeled queries, computes metrics with bootstrap confidence intervals, and generates a report.

**`scripts/expand_eval_data.py`** — Generates additional training and test queries. Created the 75 extra training queries (IDs 200-274) and the 50 fair test queries (IDs 300-349).

### Results

| Method | NDCG@5 | NDCG@10 | MRR | Latency |
|--------|--------|---------|-----|---------|
| BM25 only | 0.617 ± 0.09 | 0.614 ± 0.08 | 0.768 ± 0.10 | 21ms |
| Semantic only | 0.606 ± 0.07 | 0.603 ± 0.06 | 0.815 ± 0.08 | 36ms |
| Hybrid (BM25 + Semantic) | 0.636 ± 0.08 | 0.644 ± 0.06 | 0.807 ± 0.09 | 71ms |
| + Cross-Encoder | 0.651 ± 0.08 | 0.657 ± 0.06 | 0.825 ± 0.09 | 6166ms |
| + LightGBM Blender | **0.670 ± 0.08** | 0.657 ± 0.07 | 0.806 ± 0.09 | 6134ms |

**Key observations:**

1. **Monotonic improvement**: NDCG@5 increases at every stage: 0.617 → 0.636 → 0.651 → 0.670. Every stage adds value.

2. **Total improvement**: +8.6% NDCG@5 from BM25-only to full pipeline.

3. **The MRR trade-off**: MRR drops from 0.825 (CE) to 0.806 (LightGBM). This is because LambdaRank optimizes the entire list's NDCG, which sometimes means pushing the single best result from position 1 to position 2 in order to promote a better overall ranking. This is a design choice, not a bug — we optimize list quality, not first-result quality.

4. **Lower numbers than training set**: The training evaluation showed NDCG@5 = 0.843 (LOOCV). The fair test shows 0.670. This isn't model regression — the test queries are intentionally harder (rare cancers, comorbidities). The BM25 baseline is the same (0.617), confirming the queries themselves are harder.

5. **Wide confidence intervals**: ± 0.08 means adjacent stages have overlapping confidence intervals. The trend is convincing, but individual stage improvements are not statistically significant in isolation.

---

## 14. The Full Pipeline: How Everything Connects

### The Multi-Stage Retrieval Pattern

Before walking through the code, let's frame our pipeline in the standard terminology used by recommendation systems and search teams at companies like Google, Netflix, and Spotify. This vocabulary is expected in MLE interviews for ranking and retrieval roles.

**The fundamental problem:** We have 140,000 trials. Our best model (the cross-encoder) takes ~80ms per (query, trial) pair. Scoring all 140K trials would take 140,000 × 80ms = **3.1 hours**. Obviously impossible for real-time search. The solution is a **multi-stage funnel** where each stage narrows the candidate set, allowing subsequent stages to use more expensive models.

**Stage 1: Candidate Generation** (BM25 + Semantic → ~350 unique candidates)

Goal: **high recall** — find as many potentially relevant trials as possible. Tolerate low precision (false positives are fine; false negatives are not). The candidate generation stage is the only one that touches the full corpus, so it must be fast.

Our approach: run two parallel candidate generators:
- **BM25** (Elasticsearch): lexical matching, returns top 200 candidates in ~22ms
- **Semantic** (FAISS): embedding similarity, returns top 200 candidates in ~37ms
- **RRF fusion**: merge both lists into ~350 unique candidates in ~2ms

Why two generators? They find different trials. BM25 catches exact drug names and gene mutations that the embedding model might miss. Semantic search catches meaning matches that BM25 misses ("chemo stopped working" → "failed prior chemotherapy"). Together, recall is higher than either alone.

Total latency: ~60ms. Candidates narrowed from 140,000 → ~350.

**Stage 2: Ranking** (Cross-encoder scores top 50 candidates)

Goal: **higher precision** — re-score the candidates with a model that deeply understands query-trial relevance. This stage can be slower because it only processes 50 candidates, not 140K.

Our approach: the cross-encoder reads each (query, trial) pair jointly through all 12 BERT layers. It sees the full interaction between query words and trial text — something the bi-encoder's separate encoding can't capture.

Why only 50, not all 350? The cross-encoder takes ~80ms per pair. 50 × 80ms = 4 seconds (already our bottleneck). 350 × 80ms = 28 seconds — way too slow. We take the top 50 from RRF because RRF scores are a reasonable pre-filter: if a trial didn't rank in the top 50 by combined BM25 + semantic evidence, the cross-encoder is unlikely to rescue it.

Latency: ~4,000ms. Candidates narrowed from ~350 → 50 scored.

**Stage 3: Re-ranking / Business Logic** (LightGBM blends metadata to select top 20)

Goal: incorporate **non-text signals** that pure text models miss. A trial can be textually perfect but practically useless (completed, Phase 1 with 10 patients, not recruiting). This stage adds the "business logic" layer.

Our approach: LightGBM LambdaRank takes 11 features per candidate — the four retrieval scores (BM25, semantic, RRF, cross-encoder) plus seven metadata features (phase, recruiting status, enrollment size, condition match, title overlap, eligibility presence) — and produces a final ranking score optimized for NDCG.

In production recsys, this is where additional business rules typically enter: diversity constraints ("don't show 5 breast cancer trials in a row"), freshness boosts ("prefer recently posted trials"), user personalization ("this patient is in Boston — boost trials at MGH"). Our LightGBM captures some of this (recruiting status, enrollment size) but doesn't yet have personalization or diversity.

Latency: ~10ms. Final output: top 20 ranked results.

**The full funnel:**

```
                     140,000 trials in corpus
                              │
                    ┌─────────┴─────────┐
                    │   Candidate Gen   │  ← Stage 1: fast, high recall
                    │   BM25 + Semantic │     ~60ms, scans full corpus
                    │   + RRF fusion    │
                    └─────────┬─────────┘
                          ~350 candidates
                              │
                    ┌─────────┴─────────┐
                    │     Ranking       │  ← Stage 2: slow, high precision
                    │   Cross-encoder   │     ~4,000ms, 50 forward passes
                    │   (top 50 only)   │
                    └─────────┬─────────┘
                         50 scored
                              │
                    ┌─────────┴─────────┐
                    │    Re-ranking     │  ← Stage 3: fast, adds metadata
                    │     LightGBM     │     ~10ms, feature-based
                    │   (all features) │
                    └─────────┬─────────┘
                         top 20 results
                              │
                              v
                        User sees results
```

**Why multi-stage? The cost-quality tradeoff.**

| Stage | Candidates | Model cost per candidate | Total time | Accuracy |
|---|---|---|---|---|
| Candidate gen | 140,000 | ~0.0004ms (index lookup) | ~60ms | Low (no cross-attention) |
| Ranking | 50 | ~80ms (full BERT pass) | ~4,000ms | High (full query-trial interaction) |
| Re-ranking | 50 | ~0.2ms (LightGBM) | ~10ms | Highest (text + metadata combined) |

Each stage makes a tradeoff: cheaper models for more candidates, expensive models for fewer candidates. The funnel ensures that the expensive stages are always tractable.

**Interview question: "How would you improve this pipeline?"** Several options: (1) Speed up Stage 2 with model distillation (MiniLM) to handle 200 candidates instead of 50 — better recall. (2) Add a lightweight pre-ranker between Stages 1 and 2 (a small neural model that's faster than the cross-encoder but better than RRF). (3) Add personalization to Stage 3 (user location, prior searches, saved trials). (4) GPU-serve the cross-encoder to reduce Stage 2 from 4s to ~200ms.

### Batch vs Real-time vs Nearline Processing

Every component in an ML system runs in one of three processing modes. Knowing which mode each component uses — and why — is essential for system design interviews and for understanding where bottlenecks live.

**Batch processing** (minutes to hours, run offline on a schedule):
- Processes large amounts of data at once, not triggered by user requests
- Latency doesn't matter — correctness and throughput do
- Typically runs on a schedule (daily, weekly) or when data changes

**Real-time processing** (milliseconds, per user request):
- Triggered by a user action (a search query)
- Latency is critical — the user is waiting
- Must complete within a latency budget (typically <1 second for search)

**Nearline processing** (seconds, per user request but too slow for real-time):
- Triggered by a user request, like real-time
- But too slow to meet the real-time latency budget
- A sign that optimization is needed (distillation, GPU serving, caching)

**How each pipeline component maps:**

| Component | Mode | Latency | When it runs | Why this mode |
|---|---|---|---|---|
| **Download trials** (ClinicalTrials.gov API) | Batch | ~2 hours | When corpus is updated | 140K API calls, rate-limited to 2/sec |
| **Elasticsearch indexing** (`build_index.py`) | Batch | ~10 min | After download | Bulk-insert 140K documents |
| **FAISS index building** (`build_index.py`) | Batch | 2-3 hours | After model retrain | Embed 140K trials through BioLinkBERT |
| **LightGBM training** (`train_ranker.py`) | Batch | ~30 sec | After new labels added | Train on 6,018 feature vectors |
| **Bi-encoder training** (`finetune_embeddings.py`) | Batch | ~5 hours (A100) | After new training data | 586K triplets × 3 epochs |
| **BM25 search** | Real-time | **22ms** | Per query | Inverted index lookup — O(terms × postings) |
| **Query embedding** | Real-time | **15ms** | Per query | Single BERT forward pass |
| **FAISS search** | Real-time | **22ms** | Per query | 140K dot products (brute force) |
| **RRF fusion** | Real-time | **2ms** | Per query | Dict merge + sort |
| **LightGBM inference** | Real-time | **10ms** | Per query | 50 candidates × 11 features |
| **Cross-encoder scoring** | **Nearline** | **~4,000ms** | Per query | 50 BERT forward passes on CPU |

**The nearline bottleneck.** Our cross-encoder is the only nearline component — it's triggered per-query but takes 4 seconds, far beyond real-time latency budgets. This is our #1 production concern. The total pipeline time is ~4,070ms, and the cross-encoder accounts for ~98% of it.

```
Per-query latency breakdown:

BM25 search:          ████ 22ms
Query embedding:      ███ 15ms
FAISS search:         ████ 22ms
RRF fusion:           █ 2ms
Cross-encoder:        ████████████████████████████████████████████████████ 4,000ms
LightGBM inference:   ██ 10ms
                      ─────────────────────────────────────────────────────
Total:                ~4,070ms (cross-encoder = 98% of latency)
```

**Moving nearline to real-time — the optimization roadmap:**

| Approach | Expected latency | Tradeoff |
|---|---|---|
| Current (CPU, BioLinkBERT) | ~4,000ms | Accurate but too slow |
| GPU serving (same model) | ~200ms | Needs GPU infrastructure ($) |
| Distillation (MiniLM, CPU) | ~800ms | Small accuracy loss, no GPU needed |
| Distillation + int8 quantization | ~400ms | Slightly more accuracy loss |
| Distillation + GPU | ~50ms | Best of both, most expensive |
| Caching frequent queries | 0ms (cache hit) | Stale results for cached queries |

**How updates propagate through the system:**

When new trials are added to ClinicalTrials.gov:
1. **Batch**: Download new trials → update SQLite → bulk-insert into Elasticsearch → rebuild FAISS index
2. All real-time components automatically use the updated indices on the next query — no model retraining needed

When the bi-encoder model is retrained:
1. **Batch**: Retrain model → rebuild FAISS index (must re-embed all 140K trials with the new model) → deploy new model for query encoding
2. **Critical**: The FAISS index and the query encoder MUST use the same model version. Mixing old index with new encoder is a training-serving skew bug (see Section 2).

When new labeled data is generated:
1. **Batch**: Merge new labels → retrain LightGBM → deploy new model file
2. LightGBM training is fast (~30 seconds), so this can happen frequently

**Interview answer for "How would you architect the serving system?":** "Separate batch and real-time paths. Batch jobs (indexing, embedding, training) run on scheduled workers. The serving path is: load balancer → FastAPI → parallel BM25 + semantic retrieval → RRF merge → cross-encoder (the bottleneck — GPU-serve or distill for production) → LightGBM → return results. I'd add a Redis cache for the top 1000 most frequent queries to serve cache hits in <10ms. For the cross-encoder bottleneck, I'd start with MiniLM distillation (CPU-friendly) and evaluate GPU serving if latency requirements tighten."

### The `full_pipeline()` Method

The single most important function in the codebase is `HybridRetriever.full_pipeline()` in `src/TrialMine/retrieval/hybrid.py`. Let's walk through exactly what it does:

```
Query: "triple negative breast cancer immunotherapy"
                      |
                      v
    +=========================================+
    |  STAGE 1: BM25 Search (22ms)            |
    |  self.bm25.search(query, top_k=200)     |
    |  -> 200 keyword-matched candidates      |
    |  Also builds bm25_score_map:            |
    |    {nct_id -> BM25 relevance score}     |
    +====================+====================+
                         |
                         v
    +=========================================+
    |  STAGE 2: Semantic Search (37ms)        |
    |  self.embedder.embed_text(query)        |
    |  self.semantic.search(embedding)        |
    |  -> 200 meaning-matched candidates      |
    |  Also builds semantic_score_map:        |
    |    {nct_id -> cosine similarity}        |
    +====================+====================+
                         |
                         v
    +=========================================+
    |  STAGE 3: RRF Merge (~2ms)              |
    |  reciprocal_rank_fusion(bm25, sem)      |
    |  -> ~350 unique trials, merged          |
    |  Take top 50 (rerank_top_k)             |
    |  Enrich with metadata from ES           |
    |  Attach bm25_score + semantic_score     |
    +====================+====================+
                         |
                         v
    +=========================================+
    |  STAGE 4: Cross-Encoder (4s)            |
    |  For each of 50 candidates:             |
    |    Build trial_text (title + conditions |
    |      + summary, joined by [SEP])        |
    |  reranker.score(query, texts)           |
    |  Apply sigmoid -> CE scores [0,1]       |
    |  (Blending happens in Stage 5 or        |
    |   as fallback if no LightGBM)           |
    +====================+====================+
                         |
                         v
    +=========================================+
    |  STAGE 5: LightGBM Blender (10ms)       |
    |  blender.rerank(query, candidates)      |
    |    -> compute_features() for each       |
    |       (11 features per candidate)       |
    |    -> predict() with LightGBM           |
    |    -> sort by blender_score             |
    |    -> return top 20                     |
    +====================+====================+
                         |
                         v
    Return (ranked_results, timings_dict)
```

**What happens in Stage 4 in the code** is worth understanding in detail. The `full_pipeline()` method (lines 193-315 of `hybrid.py`) builds `trial_text` for each candidate by joining title, conditions, and summary with `[SEP]` tokens and truncating to 2048 characters. Then it calls `reranker.score(query, texts)` which returns raw logits, and converts each to a probability with sigmoid: `c["cross_encoder_score"] = 1 / (1 + math.exp(-raw))`.

**The fallback logic** in Stage 5 is important: if a LightGBM blender is provided, it uses `blender.rerank()` (the full 11-feature pipeline). If no blender is loaded, it falls back to the simpler 0.7/0.3 blended scoring from the cross-encoder section. This means the pipeline works with or without LightGBM — it gracefully degrades.

The function also returns a `timings` dictionary with per-stage milliseconds, so we can see exactly where time is spent:

| Stage | Typical Time | % of Total |
|-------|-------------|------------|
| BM25 | 22ms | 0.4% |
| Semantic | 37ms | 0.6% |
| RRF Merge | 2ms | 0.03% |
| Cross-Encoder | 4,000ms | **95%** |
| LightGBM | 10ms | 0.2% |
| **Total** | **~4,070ms** | 100% |

The cross-encoder dominates latency. This is the biggest production bottleneck.

### The API Layer

**`src/TrialMine/api/app.py`** — The FastAPI application.

`create_app()` creates the web application. The `lifespan()` context manager runs on startup and shutdown:
- **Startup**: Connects to Elasticsearch, loads the FAISS index from disk, loads the embedding model, creates a `HybridRetriever` with all three components. All are stored in `app.state` so the route handlers can access them.
- **Shutdown**: Closes the Elasticsearch connection.

The app also adds CORS middleware (Cross-Origin Resource Sharing) — this allows the Streamlit frontend (running on port 8501) to make requests to the API (running on port 8000). Without CORS, browsers block cross-origin requests for security.

**`src/TrialMine/api/routes.py`** — The endpoint handlers.

- `POST /api/v1/search` — The main search endpoint. Takes a JSON body with `query`, `top_k`, `filters`, and `method` (bm25, semantic, or hybrid). Routes to the appropriate search function.
- `GET /api/v1/trial/{nct_id}` �� Fetches a single trial's details.
- `GET /health` — Returns `{"status": "ok"}` for health checks.

**`src/TrialMine/api/schemas.py`** — Pydantic models for the API.

- `SearchRequest`: validates that the query is non-empty, top_k is between 1-100, method is one of the three allowed values
- `SearchResponse`: the response structure with results, total count, timing info
- `TrialResult`: one result with all metadata and scores
- `ErrorResponse`: structured error response (the API never returns raw 500 errors)

**IMPORTANT NOTE:** The current API uses `HybridRetriever.search()` (stages 1-3 only), NOT `full_pipeline()`. The cross-encoder and LightGBM stages are only used in evaluation scripts. This is a known gap — see Section 15.

### MLflow Experiment Tracking

**MLflow** is a tool for tracking machine learning experiments. Every time we run training or evaluation, MLflow records:
- **Parameters:** hyperparameters like learning rate, batch size, number of epochs
- **Metrics:** NDCG@5, NDCG@10, MRR, training loss at each step
- **Artifacts:** saved model files, evaluation reports, feature importance plots

The tracking data is stored in a local SQLite database (`mlflow.db`), and you can browse it through a web UI (`make mlflow` → http://localhost:5001). We use three experiments:
- `trialmind-retrieval`: bi-encoder training and embedding comparisons
- `trialmind-cross-encoder`: cross-encoder training and evaluation
- `trialmind-ranker` and `trialmind-ablation`: LightGBM training and pipeline ablation

This is invaluable for comparing runs — if we retrain the cross-encoder with graded labels, we can directly compare the new NDCG to the old one in the MLflow UI.

### All Scripts at a Glance

Here's every script in the `scripts/` directory and what it does:

| Script | What It Does |
|--------|-------------|
| `download_data.py` | Downloads 140K oncology trials from ClinicalTrials.gov API |
| `build_index.py` | Indexes trials into Elasticsearch AND/OR builds FAISS index |
| `build_faiss.py` | Builds FAISS index only (older script, `build_index.py` supersedes) |
| `generate_training_data.py` | Creates 586K training triplets from 3 sources |
| `finetune_embeddings.py` | Fine-tunes BioLinkBERT bi-encoder with MNRL |
| `finetune_cross_encoder.py` | Fine-tunes BioLinkBERT cross-encoder with binary loss |
| `compare_methods.py` | Runs 20 queries across BM25/semantic/hybrid, logs to MLflow |
| `compare_embeddings.py` | Compares fine-tuned vs off-the-shelf embeddings |
| `build_eval_dataset.py` | Pools results from both models, labels with Claude Haiku |
| `evaluate_cross_encoder.py` | Evaluates CE re-ranking vs hybrid baseline |
| `demo_reranker.py` | Before/after re-ranking demo on 3 example queries |
| `expand_eval_data.py` | Generates additional training + test queries |
| `train_ranker.py` | Trains LightGBM LambdaRank on labeled features |
| `build_fair_eval.py` | Creates 50 held-out test queries, runs fair ablation |
| `evaluate.py` | Full ablation evaluation with bootstrap CIs |

### How the FAISS Index Is Built

`scripts/build_index.py` has a **memory-efficient streaming approach** for building the FAISS index. Instead of loading all 140K trials into RAM at once, it streams them from SQLite in chunks of 2,000:

```python
def build_semantic_index(db_path, faiss_path, model_name, batch_size):
    embedder = TrialEmbedder(model_name)
    index = faiss.IndexFlatIP(768)
    trial_ids = []

    # Stream trials from SQLite in chunks of 2000
    for chunk in read_trials_in_chunks(db_path, chunk_size=2000):
        texts = [build_text(t.title, t.conditions, t.summary) for t in chunk]
        embeddings = embedder.embed_batch(texts, batch_size=batch_size)
        faiss.normalize_L2(embeddings)      # normalize for cosine similarity
        index.add(embeddings)               # add to index
        trial_ids.extend([t.nct_id for t in chunk])
        del embeddings                      # free memory immediately

    faiss.write_index(index, faiss_path)    # save 412 MB index file
    json.dump(trial_ids, open(mapping_path, "w"))  # save NCT ID mapping
```

The script also supports **model aliases**: `--model fine-tuned` resolves to `models/embeddings/fine-tuned`, and `--model off-the-shelf` resolves to `michiyasunaga/BioLinkBERT-base`. The FAISS output path is automatically set based on the alias (`data/faiss_finetuned.index` or `data/faiss_offshelf.index`).

After building, the script runs a test query ("immunotherapy for melanoma that has spread") and prints the top-5 results as a sanity check.

### Docker and the Development Environment

**Elasticsearch runs in Docker.** The `docker-compose.yml` configures it:

```yaml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    environment:
      - discovery.type=single-node       # no clustering needed
      - xpack.security.enabled=false     # disable auth for local dev
    ports:
      - "9200:9200"                      # REST API port
    volumes:
      - es_data:/usr/share/elasticsearch/data  # persist index data
```

Start with `docker compose up -d elasticsearch` (or `docker start es` if the container already exists).

### Makefile Targets

The `Makefile` provides convenient shortcuts for every step:

| Command | What It Does |
|---------|-------------|
| `make setup` | Install package in editable mode with dev dependencies |
| `make download` | Download 140K trials from ClinicalTrials.gov |
| `make index` | Build Elasticsearch + FAISS indexes |
| `make serve` | Start FastAPI backend (port 8000) |
| `make ui` | Start Streamlit frontend (port 8501) |
| `make mlflow` | Start MLflow dashboard (port 5001) |
| `make training-data` | Generate 586K training triplets |
| `make finetune` | Fine-tune BioLinkBERT bi-encoder |
| `make finetune-cross-encoder` | Fine-tune cross-encoder |
| `make train-ranker` | Train LightGBM LambdaRank (`OMP_NUM_THREADS=1`) |
| `make evaluate` | Full ablation evaluation (`OMP_NUM_THREADS=1`) |
| `make test` | Run pytest test suite |
| `make lint` | Run ruff linter |

Note that `train-ranker` and `evaluate` prepend `OMP_NUM_THREADS=1` to avoid the FAISS+LightGBM OpenMP conflict on macOS.

**Example API request and response:**

```
POST http://localhost:8000/api/v1/search
{
  "query": "breast cancer immunotherapy",
  "top_k": 5,
  "method": "hybrid",
  "filters": {"status": "RECRUITING"}
}

Response:
{
  "results": [
    {
      "nct_id": "NCT04191135",
      "title": "A Study of Pembrolizumab in Triple-Negative Breast Cancer",
      "conditions": "Breast Cancer",
      "phase": "Phase 3",
      "status": "RECRUITING",
      "score": 0.0317,
      "source": "both"    // found by BM25 AND semantic
    },
    ...
  ],
  "total": 5,
  "search_method": "hybrid"
}
```

### The Streamlit Frontend

**`src/TrialMine/ui/app.py`** — The patient-facing web interface.

Built with Streamlit (a Python library for building web apps with minimal code). It communicates with the FastAPI backend via HTTP requests using the `httpx` library.

Features:
- Search bar with 3 example query buttons
- Sidebar: search method selector, status filter, phase filter, max results slider (5-100)
- Result cards showing: title, conditions, phase badge, status colored label, relevance score, ClinicalTrials.gov link, source tag (keyword/semantic/both)
- Color coding: green = RECRUITING, blue = ACTIVE_NOT_RECRUITING, gray = COMPLETED

---

## 15. What's Planned But Not Yet Built

### LangGraph Agent System (Stubs)

The `src/TrialMine/agents/` directory contains **stub implementations** for a LangGraph-based agent system. All functions currently raise `NotImplementedError`. Here's what's planned:

**`query_parser.py` — QueryParser Node**
- Takes a raw patient description like "I'm a 45-year-old woman with EGFR-positive lung cancer"
- Extracts structured fields: cancer_type, stage, biomarkers, age, sex, location
- Produces a reformulated search query optimized for the retrieval pipeline
- Will use Claude with structured output (Pydantic schema)

**`tools.py` — Four Agent Tools**
1. **`search_trials(query, top_k)`** — Calls the full pipeline (BM25 + semantic + CE + LightGBM)
2. **`get_trial_details(nct_id)`** — Fetches complete trial record from SQLite
3. **`check_eligibility(nct_id, patient_profile)`** — Matches patient against trial criteria
4. **`explain_trial(nct_id, patient_profile)`** — Generates a plain-English explanation of why a trial matches

**`orchestrator.py` — SearchOrchestrator Node**
- ReAct-style agent loop that decides which tools to call and when
- Receives structured query from QueryParser, calls search, checks eligibility, generates explanations
- Decides when enough results are found (stopping condition)

**`pipeline.py` — End-to-End Pipeline**
- Builds a LangGraph `StateGraph` wiring QueryParser → SearchOrchestrator
- Provides `search(patient_description)` as the public entry point
- Supports streaming via LangGraph checkpointer

**Why this matters for the interview:** This shows the progression from a pure ML pipeline (retrieve + rank) to an **agentic system** where an LLM orchestrates the search, decides what additional information to fetch, and explains results to patients in plain language.

### Feature Extraction (Stubs)

The `src/TrialMine/features/` directory contains stubs for two feature extraction modules:

**`eligibility.py` — `compute_eligibility_features(trial, patient_profile)`**
Planned to extract structured eligibility features:
- `age_match`: Is the patient's age within the trial's [min_age, max_age] range?
- `sex_match`: Does the patient's sex match the trial's requirement?
- `biomarker_match`: What fraction of trial-required biomarkers does the patient have?
- `prior_therapy_mention`: Do the patient's prior treatments overlap with trial criteria?

These features would be added to the LightGBM feature set (expanding from 11 to ~15 features), enabling the ranker to incorporate patient-specific eligibility signals.

**`concepts.py` — Medical Concept Extraction**
Planned to use **SciSpacy** (already a dependency) with UMLS entity linking:
- `extract_concepts(text)` — NER on clinical text to find cancer types, biomarkers, drugs
- `normalise_concept(text)` — Map surface forms to UMLS concept IDs (e.g., "non-small cell lung cancer" → C0007131)

This would enable concept-level matching: a query about "NSCLC" would match a trial about "non-small cell lung cancer" even without semantic embeddings.

### Dependencies and Technology Stack

The `pyproject.toml` defines 28 dependencies. Here are the key ones grouped by purpose:

| Category | Packages | Purpose |
|----------|----------|---------|
| **API/Serving** | FastAPI, uvicorn, pydantic, httpx | Web API + request validation |
| **Search** | elasticsearch[async], faiss-cpu | BM25 + vector search |
| **ML/Embeddings** | torch, transformers, sentence-transformers | Neural models |
| **Ranking** | lightgbm, scikit-learn | LambdaRank + utilities |
| **NLP** | scispacy | Biomedical NER (planned) |
| **Tracking** | mlflow, optuna | Experiment logging, hyperparameter tuning (planned) |
| **Agents** | langgraph, langchain-anthropic | LLM agents (planned) |
| **Frontend** | streamlit | Patient-facing UI |
| **Infrastructure** | prometheus-client, python-dotenv | Monitoring (planned), env config |

Packages marked "planned" are installed but not yet integrated into the working pipeline.

Three CLI entry points are defined:
- `trialmine-serve` → starts the FastAPI backend
- `trialmine-ui` → starts the Streamlit frontend
- `trialmine-ingest` → runs the data download pipeline

---

## 16. Current Problems: What's Wrong and Why

### Problem 1: LLM Labels Are Not Calibrated

All our relevance labels come from Claude Haiku. We don't know if Haiku's "score 2" matches what a human oncologist would say. 46% of labels are score 3 (highly relevant), which seems generous. Without a human gold standard (labeling the same pairs with domain experts and measuring **Cohen's kappa** — a statistical measure of inter-rater agreement), we can't trust the absolute NDCG values. Only relative comparisons between methods are reliable.

### Problem 2: Cross-Encoder Was Trained on Binary Labels

The cross-encoder can only answer "right cancer or wrong cancer?" It can't distinguish between a perfect-match Phase 3 recruiting trial (relevance 3) and a marginal Phase 1 completed trial (relevance 1) for the same cancer type. This limits its usefulness as a feature. We have 6,018 pairs with graded (0-3) labels now — retraining the CE on these graded labels (using regression loss instead of binary classification) should significantly improve its feature quality.

### Problem 3: Only 145 Training Queries

While 145 is much better than the original 20, LightGBM is a powerful model that benefits from more data. More queries means more diverse cancer types, more edge cases, and more reliable feature importance estimates. The LOOCV confidence intervals are still wide.

### Problem 4: 4-Second Cross-Encoder Latency

Cross-encoder inference on CPU takes ~4 seconds for 50 candidates. This is 95% of pipeline latency. For a production system where users expect results in under 1 second, this is unacceptable. Solutions include GPU serving, model distillation (training a smaller, faster model to mimic the large model), or ONNX export (converting the model to a format optimized for fast inference).

### Problem 5: The API Doesn't Use the Full Pipeline

The FastAPI endpoint currently calls `HybridRetriever.search()`, which only runs stages 1-3 (BM25 + semantic + RRF). The cross-encoder and LightGBM stages exist in the code but aren't integrated into the API. Users get the hybrid results but miss the +5% improvement from re-ranking. Fixing this requires loading the CrossEncoderReranker and RankingBlender models on startup and routing search requests through `full_pipeline()`.

### Problem 6: No Query Understanding

The pipeline treats every query as a raw text string. A query like "I'm a 45-year-old woman with stage 3 EGFR-positive lung cancer looking for immunotherapy trials near San Francisco" contains structured information (age, sex, cancer type, stage, biomarker, treatment type, location) that could dramatically improve search. The LangGraph agents (`src/TrialMine/agents/`) are planned to parse queries into structured intent, but they're currently stubs.

### Problem 7: No Eligibility Matching

Finding relevant trials is only half the problem. A patient also needs to know if they **qualify** for the trial. Eligibility criteria are stored as free text ("Participants must be 18 years or older, have ECOG performance status 0-1, and have measurable disease per RECIST v1.1"). Automatically parsing these into structured rules and matching them against patient data would be hugely valuable. The `src/TrialMine/features/eligibility.py` stub exists but isn't implemented.

### Problem 8: Stale Configuration Files

The `configs/development.yaml` references outdated models: `allenai/scibert_scivocab_uncased` instead of BioLinkBERT, the v1 ranker instead of v2, and a local MLflow directory instead of SQLite. The `api/app.py` hardcodes FAISS paths (`data/trial_embeddings.faiss`) that don't match the actual paths (`data/faiss_finetuned.index`). These inconsistencies don't break anything currently (the correct paths are used in scripts), but they would cause confusion if someone tried to use the config file or deploy the API with the full pipeline.

### Problem 9: Limited Statistical Power

With 50 test queries and confidence interval widths of ±0.08, adjacent stages in the pipeline have overlapping confidence intervals. We can't say with 95% confidence that any individual stage improvement is statistically significant. The monotonic trend across all 5 stages is convincing, but we'd need 200+ test queries to achieve narrow enough intervals for rigorous statistical claims.

---

## 17. What We Can Do Next

### 1. Retrain the Cross-Encoder on Graded Labels

We now have 6,018 (query, trial) pairs with graded relevance scores (0-3). Instead of binary classification (relevant/irrelevant), we can train the cross-encoder with regression loss (predict the relevance score directly). This would teach the model to distinguish "highly relevant" (3) from "marginally relevant" (1), which should dramatically improve its value as a LightGBM feature.

### 2. Optuna Hyperparameter Tuning for LightGBM

**Optuna** is a library for automated hyperparameter optimization. It searches over settings like `num_leaves` (tree complexity), `learning_rate`, `min_data_in_leaf`, etc. to find the combination that maximizes NDCG on cross-validation. With 145 queries, this is now viable — at 20 queries, it would have overfit the hyperparameters to those specific queries.

### 3. Complete LangGraph Agents

Build the query parser (extract cancer type, stage, biomarkers, age, sex, location from the query) and search orchestrator (decide which search method to use, when to refine the query, when to stop). The agents would use Claude to understand the query and generate explanations for why each trial matches.

### 4. Human Gold Standard

Have domain experts (oncologists, clinical trial coordinators) label 100-200 (query, trial) pairs. Compute Cohen's kappa against Claude Haiku's labels to measure agreement. This tells us how much to trust the LLM labels and calibrates the absolute NDCG values.

### 5. Model Distillation for CE Latency

**Distillation** means training a small, fast model to mimic the predictions of a large, slow model. We could train a MiniLM-sized model (22M parameters instead of 110M) to reproduce the BioLinkBERT cross-encoder's scores. Expected to reduce inference from 4 seconds to under 500 milliseconds.

### 6. Embedding-Based Hard Negative Mining

Currently, our hard negatives are mined using keyword overlap on condition strings. Better approach: encode all trials with the current fine-tuned model, and for each positive trial, find the nearest neighbors that aren't relevant. These are the hardest possible negatives — the cases where the model is currently most confused.

### 7. Eligibility Criteria Parsing

Use medical NER (Named Entity Recognition) models like SciSpacy to extract structured eligibility requirements (age range, required biomarkers, excluded conditions) from free text. Match these against patient data to show which trials a patient likely qualifies for.

### 8. Production Serving

Export the cross-encoder to ONNX format (optimized for inference) or serve on a GPU. Add API key authentication, rate limiting, response caching for frequent queries, and multi-worker Uvicorn for handling concurrent users.

---

## 18. Lessons Learned

### 1. Evaluation Design Matters More Than Model Design

The pooling fix changed the reported improvement from +123% to +49% — a 2.5x difference just from fixing the evaluation methodology. If we hadn't caught this, we would have published inflated numbers and made incorrect decisions about the model. **Get evaluation right first, then improve the model.**

### 2. BM25 Is a Stronger Baseline Than You Think

Throughout all experiments, BM25 drives first-result quality. MRR is 0.768-0.912 across all evaluations — a relevant trial almost always appears in position 1 or 2, courtesy of keyword matching. The 5-stage pipeline improves NDCG@5 by only +8.6% over BM25 alone. Don't dismiss simple methods.

### 3. The Same Model Can Help or Hurt Depending on Its Role

The cross-encoder **destroyed** results as a standalone re-ranker (NDCG@5 dropped 19.5%). But as a **feature** in LightGBM, it became the single most important signal (gain = 1,673). The model didn't change — its role did. A binary disease-matching signal is useless as a sole ranker but extremely valuable as one input among many in a learned combination.

### 4. More Data Changes What the Model Can Learn

With 20 training queries, LightGBM thought RRF was the most important feature (gain = 393, CE was #2 at 243). With 145 queries, the same model architecture learned that CE is actually more important (gain = 1,673, RRF dropped to 1,012). The model wasn't architecturally flawed at 20 queries — it was **data-starved**. Always consider whether poor performance is a data problem before redesigning the architecture.

### 5. Measure Before Assuming

The anisotropy problem (cosine range of 0.047) was invisible without explicit measurement. If we hadn't measured the cosine score distribution, we might have concluded "semantic search doesn't work for clinical trials" and abandoned it entirely. Instead, we diagnosed it as an embedding geometry problem with a clear fix (contrastive fine-tuning). **Always measure the intermediate signals, not just the final output.**

### 6. Binary Labels Are a Ceiling

The cross-encoder achieved 0.992 NDCG@10 on its validation set — nearly perfect at binary classification. But it couldn't distinguish graded relevance (score 2 vs score 3) because it was never trained to. The training signal defines what the model can learn. If you train on binary labels, you get a binary model, no matter how powerful the architecture.

---

## 19. MLE Interview Guide: Questions This Project Prepares You For

This section maps common MLE interview questions to the specific concepts and sections in this document. For each question, we give the key points to hit and where the deeper explanation lives.

### System Design: "Design a clinical trial search engine"

This is a system design question you can answer end-to-end from this project. Hit these points in order:

**1. Requirements clarification (30 seconds):**
- How many trials? (~140K now, could be 1M+)
- Latency target? (<1s for users, <5s if acceptable)
- Query type? (patient language, not clinical — vocabulary mismatch)
- Ranking quality metric? (NDCG@5, MRR)

**2. Architecture (walk through the 5 stages from Section 1):**
```
Query → BM25 (22ms, keywords) → Semantic (37ms, meaning)
      → RRF Merge → Cross-Encoder (4s, deep relevance)
      → LightGBM (10ms, metadata features) → Top 20 results
```

**3. Key design decisions to mention:**
- RRF over score-level fusion (Section 7) — rank-based avoids incomparable score scales
- Blended CE scoring, not replacement (Section 11) — binary CE destroys nuanced RRF rankings
- LambdaRank over regression (Section 12) — directly optimizes NDCG
- Pooled evaluation (Section 10) — avoids evaluation bias

**4. Scaling discussion:**
- 140K → 10M: replace IndexFlatIP with HNSW (Section 2)
- 4s CE latency: model distillation to MiniLM, ONNX export, GPU serving (Section 17)
- Multi-user: async Uvicorn, response caching for common queries

### System Design: "How would you evaluate this system in production?"

**Offline evaluation (what we have):**
- Held-out test queries with LLM-labeled relevance (Section 13)
- Bootstrap confidence intervals for statistical reliability (Section 3)
- Ablation study to isolate each stage's contribution (Section 13)

**Online evaluation (what we'd add):**
- **A/B testing:** Route 5% of traffic to pipeline variants, measure click-through rate
- **Interleaving:** For each query, interleave results from two pipelines, measure which results users click
- **User feedback signals:** Click-through rate, time-on-trial-page, "save trial" actions, query refinement rate
- **Degradation detection:** Monitor NDCG on a fixed set of canary queries daily, alert on regression

**Mention the evaluation bias story (Section 10):** This shows you understand how evaluation can go wrong — a critical MLE skill.

### ML Fundamentals: "Explain how a Transformer works"

**Key points from Section 2:**
1. Self-attention: each word computes attention scores with every other word via Q/K/V vectors
2. Multi-head: 12 parallel attention heads for different linguistic aspects
3. Scaling: `Q·K / sqrt(d)` prevents softmax saturation
4. 12 stacked layers progressively build deeper representations
5. Pre-training: MLM (predict masked words) + NSP (predict sentence order)
6. Contextual embeddings: same word gets different vectors in different contexts

**BioLinkBERT-specific detail to mention:** Pre-trained with citation links — if Paper A cites Paper B, their sentences are related. Teaches cross-document biomedical relationships.

### ML Fundamentals: "What is the bias-variance tradeoff?"

**Key points from Section 2:**
- Bias = underfitting (model too simple to capture patterns)
- Variance = overfitting (model memorizes noise in training data)
- Our concrete example: LightGBM with 20 queries overfit (variance too high, NDCG 0.980 train vs 0.844 LOOCV). With 145 queries, the gap shrunk — more data reduces variance.
- Our other example: base BioLinkBERT underfit the ranking task (never trained to rank = high bias). Fine-tuning fixed the bias.

### ML Fundamentals: "What loss function would you use for X?"

**Map tasks to losses (from Section 2):**
- Binary classification → BCE (our cross-encoder)
- Regression → MSE (would use for graded CE retraining)
- Retrieval → Contrastive/MNRL (our bi-encoder)
- Ranking → LambdaRank (our LightGBM)
- Multi-class → Categorical cross-entropy

**Critical follow-up: "Your CE has 99.2% accuracy but real-world performance is poor. What happened?"**
Answer: Binary labels created a ceiling. The model perfectly distinguishes right-disease from wrong-disease, but can't distinguish highly-relevant (score 3) from marginally-relevant (score 1). The loss function (BCE) defines what the model can learn. Fix: retrain on graded labels with MSE or ordinal regression. (Section 11)

### IR-Specific: "Explain NDCG"

Walk through the worked example from Section 3:
1. DCG = sum of `(2^relevance - 1) / log2(position + 1)` for each result
2. IDCG = DCG of the perfect ranking
3. NDCG = DCG / IDCG (0 to 1, higher is better)
4. Why `2^rel`: exponentially rewards high-relevance items
5. Why `log2(pos+1)`: discounts items further down the list

**Be ready for:** "When would you use NDCG vs MRR vs MAP?"
- NDCG: when relevance is graded (0-3) and you care about the full ranking quality
- MRR: when you only care about the first relevant result (e.g., "I need any relevant trial quickly")
- MAP: when relevance is binary and you care about both precision and recall at multiple cutoffs

### IR-Specific: "Compare BM25 and semantic search"

| Dimension | BM25 | Semantic |
|-----------|------|----------|
| Matching | Exact keywords | Meaning/concepts |
| "pembrolizumab" | Perfect match (rare term = high IDF) | Also matches (if pre-trained on biomedical text) |
| "chemo stopped working" | No match for "failed prior chemotherapy" | Matches (synonyms map to similar vectors) |
| Latency | 22ms | 37ms |
| Failure mode | Vocabulary mismatch | Anisotropy, hub trials (Section 6) |
| Pre-computation | Inverted index (built once) | Embeddings (built once) |

**Key point:** They find completely different relevant trials (0% top-3 overlap before fine-tuning). This complementarity is why hybrid search (RRF) works — you get the best of both.

### IR-Specific: "What is Learning to Rank?"

**Three paradigms from Section 2:**
1. **Pointwise:** predict relevance per document independently (our CE)
2. **Pairwise:** predict which of two documents is more relevant (our bi-encoder/MNRL)
3. **Listwise:** optimize the entire ranked list (our LightGBM/LambdaRank)

**Key insight:** Each paradigm has a different inductive bias. Pointwise doesn't know documents compete for positions. Pairwise doesn't know position 1 matters more than position 10. Listwise handles both. Our pipeline uses progressively more sophisticated paradigms at each stage.

### Applied ML: "How do you handle limited labeled data?"

**What we did (from Sections 8-13):**
1. **Transfer learning:** Start from pre-trained BioLinkBERT, not random initialization (Section 2)
2. **Synthetic data augmentation:** 242K metadata pairs + 1,500 Claude Haiku queries + 730K hard negatives (Section 8)
3. **In-batch negatives:** MNRL gives 33 contrasts per step from a batch of 32 (Section 9)
4. **Leave-one-query-out CV:** Maximum use of limited labeled queries for evaluation (Section 12)
5. **Progressive data collection:** Started with 20 queries, expanded to 145 as we learned what the model needed (Sections 10, 12, 13)
6. **LLM-as-judge:** Generate thousands of labels at $2 instead of $800 of human annotation time (Section 10)

### Applied ML: "Your model improved metrics but hurt the product. What happened?"

**Our exact story from Section 11:**
- Cross-encoder achieved 0.992 NDCG on validation
- When used to replace RRF rankings, NDCG@5 dropped from 0.816 to 0.657 (-19.5%)
- Root cause: binary training labels → model only learned "right disease or wrong disease"
- It replaced nuanced multi-signal RRF rankings with blunt disease matching
- Fix: use CE as a **feature** (0.3 weight) not a **replacement** — blending preserves RRF quality while adding CE signal
- Lesson: a model that scores 99.2% on its validation metric can still hurt the overall system if its training objective doesn't match the system's needs

### Applied ML: "How would you debug a model that isn't working?"

**Our debugging methodology (from Sections 6, 10, 11):**

1. **Measure intermediate signals, not just final output:**
   - Cosine score distribution revealed anisotropy (Section 6): range of 0.047 → model can't differentiate
   - Hub trial analysis: 3 trials in 33% of slots → embedding space collapse

2. **Compare against baselines:**
   - BM25-only baseline was already strong (NDCG 0.789)
   - Off-the-shelf CE actually hurt (-11.9%) → problem isn't "no model," it's "wrong model"

3. **Check the evaluation methodology before blaming the model:**
   - Evaluation bias inflated improvement from +49% (real) to +123% (artifact) — Section 10
   - The evaluation was wrong, not the model

4. **Look at per-query breakdowns, not just averages:**
   - CE helped on hard queries (sarcoma +31%) but hurt on easy ones
   - Feature importance shifted when we added data (CE went from #2 to #1)

### Applied ML: "Walk me through a feature engineering decision"

**Our 11 LightGBM features (Section 12) as a case study:**

"We needed to combine retrieval scores with trial metadata for final ranking. The key engineering decisions were:

1. **Log-transform enrollment:** Raw enrollment ranges from 0 to 100,000. `log(1 + enrollment)` compresses this to [0, 11.5], preventing one feature from dominating tree splits.

2. **Separate is_recruiting from is_active:** Initially seemed redundant, but recruiting trials are joinable NOW while active-not-recruiting trials might open slots. Different signals for patients.

3. **Condition exact match as binary, title overlap as continuous:** Exact condition match is a strong binary signal (right cancer type or not). Title overlap is more nuanced (0.0 to 1.0 fraction of query words in title). Different feature types for the tree model.

4. **Using raw scores, not ranks:** We included raw BM25/semantic/CE scores rather than converting to ranks. The tree model can learn non-linear relationships with raw scores (e.g., 'BM25 > 30 AND CE > 0.8') that would be lost with ranks."

### Rapid-Fire Concept Definitions

For quick recall during interviews:

| Concept | One-sentence definition | Our pipeline example |
|---------|------------------------|---------------------|
| **Attention** | Mechanism for each word to selectively weight information from other words | BERT's 12-layer, 12-head self-attention |
| **Transfer learning** | Re-using knowledge from one task for another | Pre-trained BioLinkBERT → fine-tuned for trial search |
| **Embedding** | Dense vector representation of text in continuous space | 768-dim vectors for queries and trials |
| **Cosine similarity** | Measures angle between two vectors (direction, not magnitude) | How we compare query and trial embeddings |
| **Contrastive learning** | Training by showing positive and negative examples | MNRL: 1 positive + 32 negatives per query |
| **Hard negatives** | Negatives that are similar to positives (hard to distinguish) | Same cancer, different drug |
| **Fine-tuning** | Adjusting pre-trained model weights for a specific task | BioLinkBERT → clinical trial bi-encoder |
| **Feature engineering** | Creating informative input signals from raw data | 11 LightGBM features from scores + metadata |
| **LambdaRank** | Ranking loss that weights gradients by NDCG impact | LightGBM's training objective |
| **RRF** | Combining ranked lists using reciprocal ranks | Merging BM25 + semantic (k=60) |
| **Cross-validation** | Evaluating on held-out subsets of training data | Leave-one-query-out for LightGBM |
| **Evaluation bias** | When evaluation methodology systematically favors one approach | Labeling only one model's results |
| **Anisotropy** | Embeddings clustered in same direction, can't differentiate | Base BioLinkBERT cosine range 0.047 |
| **Sigmoid** | Squishes any number to [0, 1] range | Converting CE logits to probabilities |
| **Softmax** | Converts scores to probabilities summing to 1.0 | MNRL: probability of correct positive among 33 options |

---

*Last updated: 2026-03-28. This document is for personal learning and is not committed to version control.*
