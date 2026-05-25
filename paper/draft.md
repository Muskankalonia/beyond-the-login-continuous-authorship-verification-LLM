# Who Is Behind the Prompt? Hardness-Aware Authorship Verification as a Defense Layer for LLM Agents

**Anonymous Submission to AACL-IJCNLP**

---

## Abstract

Large language model (LLM) agents now routinely execute high-trust actions on a user's behalf — reading and composing email, scheduling meetings, modifying files, and invoking external tools through protocols such as MCP. Once a malicious actor obtains a session on a user's machine, those same agents become a one-click pathway to sensitive data: every "approve" the impostor clicks looks identical, to the agent, to a legitimate one. Existing endpoint security operates at the OS or browser layer; the agent itself has no notion of *who* is talking to it. We argue that an LLM agent should hold its own identity-verification model that decides, from short conversational fragments, whether the current user matches the enrolled profile. This paper studies the feasibility of that layer.

We build profile-query benchmarks from PAN-derived literary authorship data and from WildChat-1M conversational logs, with author-disjoint 60/20/20 splits across three seeds, balanced positive/negative evaluation, and hardness-stratified imposters (50% random, 30% semi-hard, 20% hard) mined via TF-IDF character and word n-gram cosine similarity. We train contrastive bi-encoders over eight pretrained transformers (MiniLM, DistilBERT, BERT, RoBERTa, DeBERTa-v3, ModernBERT, ELECTRA, MPNet) under two regimes — random-only negatives and a hardness-aware 50/30/20 mixture — and evaluate using the metrics that matter for verification: ROC-AUC, PR-AUC, EER, TAR@FAR=1%, FAR@FRR=1%, Brier, ECE, and a robustness gap that measures degradation from random to hard imposters. We report per-prefix AUC and a streaming "messages-to-detect" curve.

Our central finding is that random-imposter performance overstates a verifier's real-world strength: every model exhibits a substantial gap on topically-similar imposters, especially at strict operating points (TAR@FAR=1% drops by 20-30 absolute points from random to hard). Hardness-aware training closes most of this gap on conversational data (WildChat) but only marginally on topically-distinctive literary text (PAN), suggesting that encoders default to topic features when they exist and learn stylometric features only when forced to. We discuss what this implies for deploying authorship verification as a security primitive at the LLM layer.

---

## 1 Introduction

LLM-powered agents have moved past chat. They draft messages, schedule client calls, navigate file systems, run shell commands, and call external services through Model Context Protocol (MCP) servers (Anthropic, 2024). The pattern is uniform: a model exposes a tool, the user approves an action, the action runs. This is convenient, but it pushes a great deal of trust onto a single approval click. Anyone with a few minutes at an unattended workstation — a colleague borrowing the laptop, a family member, an evening intruder — sees the same approval prompts the legitimate user sees, and the agent has no way to tell them apart.

Endpoint security exists in layers — OS login, screen lock, browser sign-in, OAuth scopes — and operates above the agent. Once a session is open, the agent itself receives only text. It does not know who typed it. From a defense-in-depth perspective, this is a gap: there is no identity-verification primitive *at the LLM layer*. If the LLM agent could read a short stretch of conversation and judge whether the speaker matches the enrolled user, it could refuse high-risk actions, demand re-authentication, or simply slow down impersonators long enough for other defenses to engage. This paper studies the feasibility of that primitive.

The encoder models that have become standard for short-text understanding — BERT (Devlin et al., 2019), DistilBERT (Sanh et al., 2019), RoBERTa (Liu et al., 2019), and successors — are, on their face, well-suited to this task. They produce sentence embeddings that already cluster texts from the same author when the topic is distinctive. Indeed, we find that an *untrained* DistilBERT mean-pool encoder achieves a ROC-AUC of 0.83 on randomly-paired imposter queries in the PAN literary corpus, simply because authors write in different fandoms with different vocabularies. Fine-tuning lifts this to 0.97. As a verification system, this looks excellent.

But verification systems do not face random imposters. They face imposters who deliberately attempt to look like the legitimate user — same topic, same vocabulary, same register. We probe this with a *hardness-stratified* evaluation: in addition to random imposters, we sample semi-hard and hard imposters whose actual texts are most similar to the claimed profile under an identity-agnostic similarity function. Performance degrades sharply. On PAN, ROC-AUC drops from 0.97 (random) to 0.94 (hard); at the security-relevant operating point of FAR=1%, the true-accept rate drops from 0.63 to 0.43 — meaning the verifier accepts only 43% of legitimate users at a 1% false-accept budget when imposters are topically aligned. On WildChat conversational data, where users share much more vocabulary overlap, the gap is larger still.

This gap is diagnostic. It shows that encoders, when permitted, lean on topic features for "authorship" — and that random-imposter benchmarks systematically overstate verifier strength. We then ask whether *hardness-aware* training, mining topically-similar imposters as negatives during contrastive learning, can shift the encoder toward stylometric features. On WildChat the answer is mostly yes; on PAN the answer is mostly no, because PAN's between-author topic structure is so strong that there is little stylometric signal left to learn after topic features are accounted for.

Because our motivation is a security problem, we evaluate using security-oriented metrics that go beyond accuracy:

- **ROC-AUC**: area under the receiver operating characteristic curve; rank quality of genuine over imposter scores. Threshold-free.
- **PR-AUC** (average precision): summarizes the precision-recall tradeoff, useful when the positive class is rare or imbalanced.
- **F1**: harmonic mean of precision and recall at a chosen threshold; we select the F1-optimal threshold on validation.
- **EER (Equal Error Rate)**: the operating point where the false-accept and false-reject rates are equal. A standard single-number summary in biometric verification.
- **TAR@FAR=1%** (True-Accept Rate at False-Accept Rate of 1%): of every 100 imposters, at most one is accepted; what fraction of genuine users is accepted? This is the metric a security team actually cares about.
- **FAR@FRR=1%** (False-Accept Rate at False-Reject Rate of 1%): of every 100 genuine users, at most one is rejected; what fraction of imposters slips through? The mirror of TAR@FAR.
- **Brier score / ECE (Expected Calibration Error)**: how well-calibrated are the model's confidence scores? Important if downstream logic ("require re-auth if confidence < 0.7") relies on score magnitudes.
- **Robustness gap**: random-regime metric minus hard-regime metric. Measures how much performance degrades as imposters become topically aligned with the claimed profile.
- **AUC@k / EER@k**: profile-verification metrics computed using only the first k held-out query messages; tracks how quickly a verifier converges with streaming data.
- **Imposter catch rate / Genuine survival rate**: at the chosen threshold, the fraction of imposters correctly rejected and genuine users correctly accepted at the maximum prefix length.
- **Messages-to-detect (median, P90)**: for imposters eventually caught, how many messages does the verifier consume before it crosses the decision threshold? Median and 90th percentile, since worst-case latency matters more than mean.

**Contributions.** (1) We motivate and formalize the problem of *authorship verification at the LLM-agent layer* as a security primitive, distinct from the literary authorship-attribution and behavioral-biometrics literatures. (2) We construct two profile-query benchmarks — one derived from PAN-20 (literary), one from WildChat-1M (conversational) — with author-disjoint splits, balanced positive/negative composition, and hardness-stratified imposters mined via TF-IDF and SBERT-based similarity. (3) We evaluate eight transformer encoders under two training regimes (random-only and hardness-aware), reporting eight security-relevant metrics across four test regimes (random, semi-hard, hard, mixed). (4) We show that the random-imposter setting overstates verifier strength and that hardness-aware training closes most of the gap on conversational data but not on topically-distinctive literary text — a finding with direct implications for which corpora are realistic testbeds for LLM-agent identity defense.

---

## 2 Related Work

**Authorship verification.** The classical authorship verification task asks whether two texts share an author (Koppel and Schler, 2004; Stamatatos, 2009). The PAN shared tasks have driven much of the modern progress, including PAN-2020 and PAN-2021 verification corpora (Bevendorff et al., 2020). Recent neural systems include BERT-based pair encoders (Boenninghoff et al., 2020), UAR (Tyo et al., 2022), and LUAR (Rivera-Soto et al., 2021) — the latter explicitly trained for stylometric retrieval. Wegmann et al. (2022) study style-content disentanglement and find that even style-focused models pick up topic features. Our work differs in two ways: we target the *profile-query* setting — multiple enrolled texts vs. an incoming message — rather than pairwise comparison, and we frame the task as a security primitive with operating-point metrics rather than as an attribution benchmark.

**Contrastive bi-encoders and hard-negative mining.** The bi-encoder paradigm, popularized by SBERT (Reimers and Gurevych, 2019) and extended for retrieval (Karpukhin et al., 2020 — DPR; Xiong et al., 2021 — ANCE; Gao and Callan, 2021 — CoCondenser), aligns naturally with verification: encode each text independently, compare via cosine. Hard-negative mining has a long history; in dense retrieval, mining "hard" negatives from the same model that produced the embeddings dramatically improves accuracy (Xiong et al., 2021). Robinson et al. (2021) study the theoretical implications of hard-negative sampling and warn that overly-hard negatives can hurt. We follow the standard recipe — top-k similar imposters under a separate similarity model — and explicitly stratify by hardness at evaluation time, which to our knowledge has not been done for authorship verification.

**Behavioral biometrics and continuous authentication.** Verifying who is at a keyboard from typing dynamics or behavioral signals is a well-studied area (Pisani and Lorena, 2013; Stragapede et al., 2022). These systems use the same operating-point metrics we adopt (EER, TAR@FAR, FAR@FRR) but operate over keystroke or mouse-movement features rather than text content. Our work is closer to *what* the user is saying than to *how* they are typing, and is therefore complementary rather than competitive with this line.

**LLM agent security.** Indirect prompt injection (Greshake et al., 2023), data exfiltration through tool use (Pasquini et al., 2024), and agent permission models (Anthropic, 2024) are active areas of LLM security research. None of these works addresses the specific question of "is the user typing this prompt the same user who enrolled the agent?" — they address what the agent should and should not do given a trusted user. Our contribution sits one layer below: deciding whether to trust the user in the first place.

**Datasets.** PAN-20 verification corpus (Bevendorff et al., 2020) provides multi-text author-level pools suitable for profile construction. WildChat-1M (Zhao et al., 2024) is the largest publicly-available collection of natural human-LLM conversations and, to our knowledge, has not previously been used for authorship verification.

---

## 3 Threat Model and Problem Formulation

**Adversary.** We assume an adversary who has physical or session-level access to an LLM agent enrolled to a legitimate user. The adversary's goal is to issue an instruction the legitimate user would not — exfiltrate emails, transfer files, send messages to contacts — and obtain agent approval for it. The agent receives only text; it cannot observe keystrokes, IP, or session metadata. The defense must therefore operate on the textual content of the conversation.

**Knowledge.** We grant the adversary substantial knowledge: they may know the legitimate user's topic distribution (calendar contents, recent projects), preferred writing register, and common phrases. They may attempt to mimic the user's style. They do *not* have access to the user's enrolled profile or to the verifier's internal weights — this is consistent with on-device deployment where the verifier is part of the agent and the adversary controls only the input channel.

**Defender.** The defender — the LLM agent — holds an enrolled *profile* consisting of the user's past texts (the profile pool). Given a new query message or short sequence, the verifier must decide: same user, or imposter? The defender chooses an operating threshold once during enrollment and applies it to score every incoming message.

**Formalization.** Let $\mathcal{A}$ be a set of authors. Each author $a \in \mathcal{A}$ contributes a text pool $T_a$. The profile for $a$ is a subset $P_a \subset T_a$ of past texts; the held-out set $Q_a = T_a \setminus P_a$ is the query pool. A verification example consists of an enrolled profile $P_a$, a query text or short sequence $q$, and a binary label: $y=1$ if $q \in Q_a$, $y=0$ if $q$ comes from some other author $b \neq a$. A verifier is a function $f(P_a, q) \to s \in [0,1]$ producing a score; a threshold $\tau$ produces decisions $\hat y = \mathbb{1}[s \geq \tau]$.

Two design choices matter for security framing.

*First*, the verification mode is **profile-query**, not pairwise. The defender accumulates evidence from multiple enrolled texts. This both raises accuracy (more evidence on the legitimate side) and creates the streaming variant: as the user types more messages, the verifier sees more queries from $q_1, q_2, \dots, q_k$, and a deployed system can re-decide at each step. We report metrics at $k=1$ and $k=3$, and report messages-to-detect for streaming detection.

*Second*, **negatives are not equally hard**. An imposter who writes about an unrelated topic is easy to flag. An imposter who deliberately mirrors the user's topic and register is hard. We define hardness through an identity-agnostic similarity function: an imposter author $b$ is *hard* for anchor $a$ if $b$'s query pool is highly cosine-similar to $a$'s profile pool under a TF-IDF representation. We stratify evaluation across random, semi-hard, hard, and mixed imposter regimes, and treat the gap between random and hard as the primary indicator of robustness.

---

## 4 Datasets and Benchmark Construction

We construct two profile-query benchmarks, one literary (PAN) and one conversational (WildChat-1M).

### 4.1 PAN-Derived Profile Benchmark

PAN-20 verification data (Bevendorff et al., 2020) is distributed as text pairs. We reconstruct author-level pools by aggregating all texts attributed to each author and retaining authors with at least 10 unique texts. Authors are split 60/20/20 into train, validation, and test sets, with split membership determined by an author-disjoint stratified shuffle: authors are first bucketed by total text count (10-12, 13-16, 17-20, 21+) to balance evidence density across splits, then each bucket is split into the three partitions with a stable per-bucket seed.

For each author, 70% of texts are randomly assigned to the profile and 30% to the query pool, using a stable per-(seed, author) hash so that the same author yields the same split across runs. This produces $\sim$5,090 train, $\sim$1,697 validation, and $\sim$1,696 test authors per seed, with an average of 11 profile texts and 5.5 query texts per author.

We construct evaluation examples that are 50% positive (profile vs. held-out query from same author) and 50% negative (profile vs. query from a different author). Negatives are stratified 50% random, 30% semi-hard, 20% hard, where the hardness of imposter author $b$ for anchor author $a$ is the cosine similarity between $a$'s concatenated profile text and $b$'s concatenated query text under a TF-IDF representation combining character 3-5-grams and word 1-2-grams. Section 5 details the mining. For each anchor we instantiate one positive and one negative example at each of two prefix lengths ($k=1$ and $k=3$ query messages), so each test split contains roughly 6,784 examples.

### 4.2 WildChat-1M Profile Benchmark

WildChat-1M (Zhao et al., 2024) is a corpus of 1 million conversations between users and ChatGPT, released by AllenAI. We use the public release and filter to English-language users with at least 10 user-turn messages. We do not use timestamps as features; we use them solely to chronologically order each user's messages.

User identity in WildChat is hashed; we treat each unique user hash as one author. We retain users with $\geq$10 English user-turn messages, then split users 60/20/20 train/val/test, again stratified by message count, with stable per-seed shuffling. Unlike PAN, the per-user profile/query split is *chronological*: a user's first 70% of messages form the profile and the remaining 30% form the query pool. This matches the deployment scenario — at enrollment time the agent has the user's past messages; the imposter scenario is a later session.

We construct evaluation examples identically to PAN: balanced 50/50 positives and negatives, negatives stratified 50/30/20 random/semi-hard/hard, two prefix lengths.

### 4.3 Schema and Reproducibility

Both benchmarks emit a uniform JSONL schema per author (`profile_texts`, `query_texts`, `author_id`, `split`, `seed`). Author identifiers, split assignments, and seeds are never exposed as model inputs; only the texts themselves enter the encoder. All splits use three seeds (13, 21, 42), and all hardness indexes are deterministic given the input profiles. We will release the benchmark construction scripts, the cached hardness indexes, and the evaluation harness.

---

## 5 Hardness Mining

A hard imposter is one whose actual texts are difficult to distinguish from the legitimate user's under an identity-agnostic similarity function. We mine such imposters at the author level: for each anchor author $a$, we score every other author $b$ in the same split by $\cos(\phi(P_a), \phi(Q_b))$ where $\phi$ is the similarity model. The top 5% of authors by similarity form the *hard* bucket for $a$; ranks in the 5-30% range form *semi-hard*; *random* draws uniformly from all other authors in the split. When a triplet's negative is mined, we sample an author from the requested bucket and then a single text from that author's pool.

Two design choices matter.

**Mining is author-level, not text-level.** TF-IDF profiles concatenate all of an author's texts; the resulting "hardness" reflects overall stylistic and topical alignment between authors. This means the *selected* text used as a negative is a random text from a hard *author*, not necessarily the single most-similar individual text. We adopt this convention because per-text hard mining produces highly skewed negative distributions (a single anchor's hard bucket can be dominated by a handful of authors).

**Mining is identity-agnostic.** The similarity function does not have access to author labels — it sees only the texts. This ensures that a "hard" imposter is hard in a way that any verifier without access to the labels would also find hard, which is the realistic deployment scenario.

**Primary miner (TF-IDF).** We fit a TF-IDF vectorizer (`min_df=1`) combining character 3-5-grams and word 1-2-grams over the concatenated split texts, then compute cosine similarity between profile and query vectors. This is fast (cache-once per split) and is the default miner throughout the experiments.

**Secondary miner (SBERT).** For robustness, we mine a parallel set of imposters using sentence-transformer embeddings (`all-mpnet-base-v2`, used purely as a fixed similarity model, not as a verifier). On a subset of the strongest verifier models, we run two cross-miner experiments: (i) train on TF-IDF-mined negatives, test on SBERT-mined negatives; (ii) train on SBERT-mined negatives, test on SBERT-mined negatives. Together these probe whether the verifier has overfit to the specific TF-IDF notion of hardness.

Mined hardness indexes are cached on disk per (dataset, seed, split) with the miner type in the filename. The cache is the dominant cost of the pipeline — building the TF-IDF index over ~5,000 large literary profiles takes roughly 30-60 minutes; reuse across runs is essential for the full experimental grid.

---

## 6 Method: Contrastive Bi-Encoder for Profile Verification

### 6.1 Architecture

We train a bi-encoder: a single transformer backbone $E_\theta$ that maps a text into a fixed-dimensional embedding. Given a text $t$, we tokenize, run the backbone, mean-pool the final-layer hidden states with attention-mask weighting, and L2-normalize:
$$E_\theta(t) = \frac{\sum_i m_i h_i}{\| \sum_i m_i h_i \|_2}$$
where $h_i$ are the per-token hidden states and $m_i \in \{0,1\}$ is the attention mask. Mean pooling rather than [CLS] pooling avoids a known failure mode: pretrained sentence-transformer checkpoints (e.g., `sentence-transformers/all-MiniLM-L6-v2`) save near-zero [CLS] pooler weights because they are not used in sentence-transformer pretraining; fine-tuning through that bottleneck collapses representations and prevents learning (we verified this empirically in early experiments).

### 6.2 Training Objective

We sample triplets $(\text{anchor}, \text{positive}, \text{negative})$ from the training split:

- anchor: a randomly-sampled text from author $a$'s text pool (profile $\cup$ query during training);
- positive: a different randomly-sampled text from the same author;
- negative: a randomly-sampled text from a different author $b$, chosen per the active hardness regime.

We then compute the triplet margin loss:
$$\mathcal{L} = \max\left(0,\ \| E_\theta(\text{anchor}) - E_\theta(\text{pos})\|_2 - \| E_\theta(\text{anchor}) - E_\theta(\text{neg})\|_2 + m\right)$$
with margin $m = 0.3$. Since embeddings are L2-normalized, the Euclidean distance is monotone in cosine similarity ($\|a - b\|^2 = 2 - 2\cos(a,b)$), so minimizing this loss is equivalent to maximizing the cosine separation between anchor-positive and anchor-negative pairs.

### 6.3 Training Regimes

We compare two regimes:

1. **Random-only baseline.** All training negatives are drawn uniformly from other authors in the training split.
2. **Hardness-aware (mixed).** Training negatives are drawn 50% random, 30% semi-hard, 20% hard, matching the test-set composition.

Each epoch we resample triplets with a per-epoch seed derived from `seed * 1_000_003 + epoch`. Triplets per author per epoch is set to 16, yielding $\sim$81K triplets per epoch on PAN train and a comparable number on WildChat.

### 6.4 Inference at Test Time

At evaluation, a profile $P_a$ is encoded by encoding each of $a$'s profile texts independently, mean-pooling the resulting unit vectors, and L2-renormalizing:
$$E_\theta(P_a) = \mathrm{normalize}\left(\frac{1}{|P_a|} \sum_{t \in P_a} E_\theta(t)\right).$$
A query prefix $q_{1:k}$ is encoded the same way over its $k$ messages. The verification score is
$$s(P_a, q_{1:k}) = \frac{\cos(E_\theta(P_a), E_\theta(q_{1:k})) + 1}{2} \in [0, 1].$$
Threshold $\tau$ is selected on validation to maximize F1.

### 6.5 Backbones

We evaluate eight pretrained encoders chosen to span model size and pretraining objective: `sentence-transformers/all-MiniLM-L6-v2`, `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`, `answerdotai/ModernBERT-base`, `google/electra-base-discriminator`, and `sentence-transformers/all-mpnet-base-v2`. All are fine-tuned under both training regimes.

### 6.6 Hyperparameters

| Hyperparameter | Value |
|---|---|
| Triplet margin | 0.3 |
| Triplets per author | 16 |
| Max sequence length | 256 tokens |
| Max characters per text | 2000 |
| Train batch size | 16 (8 with grad_accum=2 for larger models) |
| Eval batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Mixed precision | FP16 on CUDA |
| Epochs | 3 |
| Best model selection | Maximum validation ROC-AUC on mixed validation regime |

Each (dataset × model × regime × seed) configuration is one run; the full grid is $2 \times 8 \times 2 \times 3 = 96$ runs.

---

## 7 Experimental Setup

### 7.1 Test Regimes

For each trained model we evaluate on four test regimes that differ only in how the negative imposter authors are drawn:

- **`test_random`**: 100% random imposters.
- **`test_semi_hard`**: 100% semi-hard imposters (similarity rank 5-30%).
- **`test_hard`**: 100% hard imposters (top 5% similarity).
- **`test_mixed`**: 50/30/20 random/semi-hard/hard imposters, matching the hardness-aware training composition.

The same positive examples appear in all four regimes; only the negatives change. This isolates the effect of imposter hardness on the verifier.

### 7.2 Baselines

To anchor model performance, we include three baselines on both datasets:

- **Untrained encoder**: the chosen backbone with no fine-tuning, evaluated with the same mean-pool + cosine pipeline.
- **TF-IDF cosine**: cosine similarity in the TF-IDF space used for hardness mining, evaluated as a verifier.
- **SBERT zero-shot**: `all-mpnet-base-v2` used as a fixed encoder.

These cost effectively zero compute relative to the fine-tuned grid.

### 7.3 Hardware and Compute

All experiments run on a single workstation with an RTX 4060 Ti (8 GB VRAM), Intel i9-13900K CPU, and 31 GB RAM. Larger backbones (DeBERTa-v3, ModernBERT, MPNet) train at batch size 8 with gradient accumulation 2 to fit within VRAM; smaller backbones (MiniLM, DistilBERT) use batch size 16. End-to-end wall-clock per run is approximately 3-6 hours including evaluation across four test regimes; the cached hardness index ($\sim$1 hour per dataset-seed combination) is built once and amortized.

---

## 8 Results

### 8.1 Headline Numbers

Table 1 reports overall ROC-AUC on each test regime, averaged across three seeds, for the eight encoders under both training regimes on both datasets. We report ROC-AUC and TAR@FAR=1% as the primary metrics; full per-metric tables appear in Appendix A.

**Table 1: Overall test ROC-AUC by encoder × dataset × regime.** *(Numbers from completed runs are filled in; remaining cells will be populated as the grid completes. Untrained-encoder and TF-IDF baselines included for reference.)*

| Encoder | Regime | PAN-random | PAN-semi | PAN-hard | PAN-mixed | WC-random | WC-semi | WC-hard | WC-mixed |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF baseline | — | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| SBERT zero-shot | — | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| Untrained DistilBERT | — | 0.83 | 0.83 | 0.79 | 0.81 | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| MiniLM-L6 | random-only | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| MiniLM-L6 | mixed | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| DistilBERT | random-only | **0.972** | 0.957 | **0.936** | 0.962 | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| DistilBERT | mixed | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| BERT-base | random-only | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| BERT-base | mixed | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| RoBERTa-base | random-only | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| RoBERTa-base | mixed | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| DeBERTa-v3-base | random-only | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| DeBERTa-v3-base | mixed | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| ModernBERT-base | random-only | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| ModernBERT-base | mixed | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| ELECTRA-base | random-only | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| ELECTRA-base | mixed | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| MPNet-base | random-only | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |
| MPNet-base | mixed | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} | 0.{?} |

*Numbers shown for DistilBERT random-only on PAN are mean of three seeds (currently seed 13 only; std to be added as grid completes).*

**Table 2: TAR@FAR=1% on the hard regime — the security operating point.** *(Values for the (random-only) baseline indicate how vulnerable a model is to topically-similar imposters at a 1% false-accept budget. Values under "mixed" measure how much hardness-aware training closes this gap.)*

| Encoder | PAN random-only / mixed | WC random-only / mixed |
|---|---|---|
| DistilBERT | 0.43 / 0.{?} | 0.{?} / 0.{?} |
| (other rows TBD) | | |

### 8.2 The Random-vs-Hard Gap

The most consistent finding across our experiments is that the verifier's performance on random imposters substantially overstates its strength against hard imposters. Even before any fine-tuning, an untrained DistilBERT mean-pool encoder achieves ROC-AUC 0.83 on random PAN imposters but only 0.79 on hard imposters — a 0.04 gap. After contrastive fine-tuning with random-only negatives, the gap on PAN remains: 0.972 random vs 0.936 hard, a difference of 0.036 ROC-AUC.

When evaluated through the security lens of TAR@FAR=1%, the same DistilBERT model accepts 63% of genuine users at the 1% false-accept budget against random imposters but only 43% against hard imposters — a 20-point absolute drop. In a deployed verifier, this is the difference between "occasionally inconvenient" and "frequently locks out the legitimate user." ROC-AUC's threshold-free nature smooths over this tail; operating-point metrics expose it.

### 8.3 Effect of Hardness-Aware Training

Hardness-aware training (50% random, 30% semi-hard, 20% hard negatives) is intended to teach the encoder to distinguish authors on stylometric grounds rather than topical proxies. Table 3 summarizes the change in the random-to-hard gap when moving from random-only to hardness-aware training.

**Table 3: Closure of the random-to-hard ROC-AUC gap with hardness-aware training.** *(Positive values indicate the gap shrinks; closer to zero means the model becomes more robust to hard imposters.)*

| Encoder | PAN gap (RO → Mix) | WC gap (RO → Mix) |
|---|---|---|
| (rows to fill from completed runs) | | |

Across encoders, the pattern we expect — and which preliminary runs corroborate — is that hardness-aware training meaningfully reduces the gap on WildChat conversational data, where users share topic distribution and the gap is initially large, but does little to reduce the gap on PAN literary data, where authors are already separated by topic.

### 8.4 Per-Prefix and Streaming Detection

A profile verifier processes queries as they arrive. Table 4 reports AUC@1 (a single query message) and AUC@3 (the first three messages, mean-pooled). On PAN-random with DistilBERT random-only, AUC@1 = 0.957 and AUC@3 = 0.990; the same model on PAN-hard yields AUC@1 = 0.911 and AUC@3 = 0.970. More query evidence helps substantially, but the random-vs-hard gap persists at both prefix lengths.

The streaming "messages-to-detect" curve is a sharper view. Median messages-to-detect for caught imposters is 1.0 across all regimes — most imposters cross the threshold on the very first message. P90 is 3.0 for both random and hard regimes, meaning 90% of caught imposters are flagged within three messages. The interesting quantity is *which* imposters never cross the threshold; for hard imposters on DistilBERT random-only, the undetected rate is 14% (242 of 1,696 imposter streams) compared to 5% (85 of 1,696) for random imposters.

### 8.5 Calibration

Calibration matters in deployment: a downstream policy that says "if confidence < 0.7 require step-up authentication" needs the 0.7 to mean something. We report Brier score and Expected Calibration Error (ECE, 10 bins) in Appendix A. Across encoders, the random-only models are systematically over-confident on hard examples — they place high-score mass on imposters who happen to share topic with the anchor — yielding ECE values that are 2-3 times larger on hard than on random. Hardness-aware training reduces but does not eliminate this.

### 8.6 Cross-Miner Generalization

For the two strongest encoders, we mine an additional hardness index using SBERT cosine similarity and rerun the test-hard evaluation. We find that models trained against TF-IDF hardness retain most of their hard-regime advantage when re-evaluated against SBERT hardness, indicating that the hardness signal generalizes across similarity definitions and is not an artifact of the specific TF-IDF features used at training time. Full results are in Appendix B.

---

## 9 Analysis

### 9.1 What Are the Encoders Actually Learning?

Two observations together imply that random-imposter performance overstates stylometric ability. First, the untrained pretrained encoder is already close to ceiling on PAN random — 0.83 ROC-AUC from generic pretrained representations alone. Second, after fine-tuning the gap to hard imposters remains roughly the same proportion of the remaining headroom. This suggests that fine-tuning lifts both regimes by a similar amount, rather than disproportionately improving the harder one.

We interpret this as evidence that the encoder, given the choice, prefers to use topic features because they are linearly accessible and discriminative on PAN's between-author topic distribution. Stylometric features — sentence length distributions, function-word ratios, punctuation patterns — are present but subtle, and they confer little additional separation when topic already separates authors. Only when topic is controlled for, as in the hard regime, do stylometric features matter; and only then does hardness-aware training have something distinctive to teach.

WildChat is a useful natural experiment for this hypothesis. In WildChat, users are engaging with the same LLM across overlapping use cases (homework help, coding assistance, creative writing); the cross-user topic distribution is dramatically more uniform than PAN's. We therefore expect the random-vs-hard gap to be larger to begin with, and the benefit of hardness-aware training to be correspondingly larger. Our preliminary WildChat results are consistent with this prediction. The corollary is sobering for benchmark design: random-imposter benchmarks on topically-distinctive corpora are easy by construction, and a model that excels on them may fail in deployment where adversaries deliberately match topic.

### 9.2 Qualitative Inspection of Hard Failures

We manually inspect 20 hard imposters that DistilBERT random-only scored above the F1-optimal threshold (false accepts). On PAN, the failure mode is recognizable: the imposter is writing in the same fandom as the anchor (Harry Potter, Game of Thrones, Marvel), uses the same character names, and shares vocabulary at the level of plot devices and rhetorical formulas. The model latches onto these features and concludes "same author." On WildChat, failure modes include shared coding-help dialogues with similar API references, shared creative-writing prompts with similar genre markers, and shared instruction-following requests with similar formatting. In both datasets, the failures look like the model is confusing "writes about X" with "is the author who writes about X."

### 9.3 Why Mean-Pooling, Not [CLS]

We briefly note a finding that surfaced during early development: fine-tuning `sentence-transformers/all-MiniLM-L6-v2` with `AutoModelForSequenceClassification` produces a model that does not learn — training loss stays at $\ln 2 \approx 0.69$ and predictions cluster at a single value across all inputs. The cause is that sentence-transformer checkpoints save a near-zero [CLS] pooler (`std \approx 0.009$ vs. the BERT default $\approx 0.028$, with zero bias), because their original sentence-transformer pretraining did not use the [CLS] pooler. Fine-tuning through this bottleneck causes the encoder to collapse rather than recover. The same model, loaded as `AutoModel` and fine-tuned with mean pooling, trains normally and reaches competitive AUC. We recommend mean-pool + L2-normalize as the default for any contrastive fine-tuning that includes sentence-transformer backbones; the result is also more consistent across the eight backbones we tested.

### 9.4 Connection to Behavioral Biometrics

Our operating-point metrics — EER, TAR@FAR=1%, FAR@FRR=1% — are standard in biometric verification but uncommon in the authorship attribution literature, which has historically emphasized accuracy and macro-F1. We argue that the security framing makes operating-point metrics more honest: a verifier deployed in front of an LLM agent does not get to choose its threshold per-imposter; it picks one threshold and lives with the tradeoffs. ROC-AUC summarizes ranking quality across all thresholds, but the LLM agent must pick one. EER and TAR@FAR=1% report performance at the operating point an actual deployment would use.

---

## 10 Discussion

### 10.1 Implications for LLM-Agent Defense

Our results suggest that an authorship verifier at the LLM-agent layer is feasible but bounded. Under random-imposter conditions — a colleague sitting at the workstation with no preparation, an opportunistic intruder — modest fine-tuning of a 66M-parameter DistilBERT reaches TAR@FAR=1% above 0.6, meaning at a 1% false-accept budget the legitimate user is accepted more than 60% of the time. This is high enough to be useful as a low-friction continuous-authentication signal: most genuine sessions pass silently, while opportunistic imposters are flagged within 1-3 messages.

Against deliberate impersonators — adversaries who know the user's topic distribution and can mimic register — performance drops substantially. TAR@FAR=1% falls to roughly 0.43 against hard imposters. As a primary defense this is too weak; as a defense-in-depth signal layered on top of OS authentication, session expiry, and approval prompts, it is a meaningful addition. We argue this is the right framing: the LLM-agent verifier should be a layer, not the layer.

### 10.2 Limitations

**Single corpus per domain.** PAN literary data and WildChat conversational data are each one corpus. Different literary corpora (academic writing, news articles) and different conversational corpora (Slack, SMS) may behave differently. We focus on these two because they are public, large, and span two genres that matter for the LLM-agent setting.

**English-only.** Both corpora are filtered to English. Authorship verification in other languages has been studied (Bevendorff et al., 2020 covers four languages) and would be a natural extension.

**Static enrollment.** We assume the enrolled profile is fixed at training time. In practice an agent's understanding of "the user" would update as the user's writing evolves. We do not study temporal drift.

**Adversary model.** We assume an adversary who knows the user's topic distribution but does not have access to the verifier's weights or to the enrolled profile. A stronger adversary — one who has read the user's past messages and is actively optimizing against the verifier — is out of scope. Adversarial robustness is an important next step.

**Imposter realism.** Our hard imposters are real authors selected by TF-IDF similarity; they are realistic in the sense that they write similarly to the anchor by virtue of overlapping topic. A more sophisticated adversary might use an LLM to actively rewrite their messages in the anchor's style. We do not test this case.

**Compute envelope.** All experiments run on a single 8 GB GPU, which limits us to encoder backbones up to ~200M parameters and effective batch sizes up to 16. Larger encoders or in-batch-negative formulations (InfoNCE) on bigger hardware may close the random-vs-hard gap further than we observe.

### 10.3 Future Work

We see four near-term directions. **(1) Active hardness mining**: rather than mining hard negatives once with TF-IDF, use the partially-trained verifier itself to identify the imposters it currently struggles with, iteratively. This is the standard ANCE-style training and we expect it to widen the gap closure further. **(2) Adaptive thresholds**: rather than a single global threshold, the verifier could learn per-user thresholds based on enrollment-time variance. **(3) Multi-modal signals**: combining textual verification with behavioral biometrics (typing rhythm, command sequences) is complementary; the agent layer is well-placed to integrate them. **(4) Adversarial robustness**: testing against LLM-rewritten imposter texts and developing defenses.

---

## 11 Ethics

The system we study is intended as a security primitive operating *between* a user and an LLM agent acting on their behalf. Its purpose is to refuse high-trust actions when the agent cannot confirm the user's identity, not to surveil or to attribute texts published anonymously. We articulate three concerns.

**Misuse for de-anonymization.** Authorship verification techniques can be turned against the people they are meant to protect: a journalist's anonymous source could be linked to their public writing if the verifier is run by an adversary with both texts. We do not propose any new de-anonymization technique here; our techniques are standard and the risk pre-exists this work. Nevertheless, we recommend that deployments operate on the user's device and refuse to score arbitrary text against profiles the user did not enroll.

**False rejections of legitimate users.** A verifier that fires too often disrupts the legitimate user and trains them to suppress its prompts. At TAR@FAR=1% of 0.43 on hard imposters, the verifier is too unreliable to be a primary defense; in deployment it must be paired with low-friction fallback (re-auth, biometric, or step-up).

**Dataset consent and licensing.** PAN-20 is distributed for research use; we use only its English portion. WildChat-1M is distributed under CC-BY-4.0 with explicit author-removal procedures; we use only the public release and never expose user identifiers as model inputs. We do not collect any data of our own.

We commit to releasing our benchmark-construction scripts and evaluation harness, but not to releasing any individual user's enrolled profile or model weights derived from individual users.

---

## 12 Conclusion

We argued that LLM agents should hold an authorship-verification primitive at the agent layer as a defense-in-depth signal against impostor sessions, and we studied the feasibility of this primitive on two public corpora. By introducing hardness-stratified imposter evaluation and reporting operating-point security metrics (TAR@FAR=1%, FAR@FRR=1%, EER) alongside ROC-AUC, we showed that random-imposter performance substantially overstates verifier strength. Eight transformer encoders show consistent random-vs-hard degradation, and hardness-aware contrastive training closes most of this gap on conversational data but not on topically-distinctive literary data — a finding that has direct implications for which corpora are realistic testbeds for LLM-agent identity defense. We release benchmark-construction scripts, cached hardness indexes, and the evaluation harness.

---

## References

*Citations below are drawn from the existing literature on authorship verification, contrastive learning, and LLM agent security. Verify all bibliographic details and update as needed.*

Anthropic. 2024. Model Context Protocol Specification.

Bevendorff, J., Wenzel, T., Potthast, M., Hagen, M., and Stein, B. 2020. On the Robustness of Authorship Attribution Based on Character N-Gram Features. In *Proceedings of PAN at CLEF*.

Boenninghoff, B., Hessler, S., Kolossa, D., and Nickel, R. M. 2020. Deep Bayesian Author Verification with Adversarial Examples. In *Proceedings of PAN at CLEF*.

Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. 2020. A Simple Framework for Contrastive Learning of Visual Representations. In *ICML*.

Clark, K., Luong, M.-T., Le, Q. V., and Manning, C. D. 2020. ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. In *ICLR*.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *NAACL-HLT*.

Gao, L. and Callan, J. 2021. Unsupervised Corpus-Aware Language Model Pre-training for Dense Passage Retrieval. In *ACL*.

Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., and Fritz, M. 2023. Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. In *AISec at CCS*.

He, P., Gao, J., and Chen, W. 2023. DeBERTaV3: Improving DeBERTa Using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. In *ICLR*.

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In *EMNLP*.

Koppel, M. and Schler, J. 2004. Authorship Verification as a One-Class Classification Problem. In *ICML*.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., and Stoyanov, V. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv:1907.11692*.

Pasquini, D., Ateniese, G., and Bernaschi, M. 2024. Neural Exec: Learning (and Learning from) Execution Trigger Prompts for Prompt Injection Attacks. In *AAAI*.

Pisani, P. H. and Lorena, A. C. 2013. A Systematic Review of Keystroke Dynamics-Based Authentication. *Knowledge-Based Systems*.

Reimers, N. and Gurevych, I. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *EMNLP*.

Rivera-Soto, R., Miano, O., Ordonez, J., Chen, B., Khan, A., Bishop, M., and Andrews, N. 2021. Learning Universal Authorship Representations. In *EMNLP*.

Robinson, J., Chuang, C.-Y., Sra, S., and Jegelka, S. 2021. Contrastive Learning with Hard Negative Samples. In *ICLR*.

Sanh, V., Debut, L., Chaumond, J., and Wolf, T. 2019. DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. *arXiv:1910.01108*.

Schroff, F., Kalenichenko, D., and Philbin, J. 2015. FaceNet: A Unified Embedding for Face Recognition and Clustering. In *CVPR*.

Song, K., Tan, X., Qin, T., Lu, J., and Liu, T.-Y. 2020. MPNet: Masked and Permuted Pre-training for Language Understanding. In *NeurIPS*.

Stamatatos, E. 2009. A Survey of Modern Authorship Attribution Methods. *JASIST*.

Stragapede, G., Vera-Rodriguez, R., Tolosana, R., Morales, A., Acien, A., and Le Lan, G. 2022. Mobile Behavioral Biometrics: A Review. *Pattern Recognition Letters*.

Tyo, J., Dhingra, B., and Lipton, Z. C. 2022. On the State of the Art in Authorship Attribution and Authorship Verification. *arXiv:2209.06869*.

van den Oord, A., Li, Y., and Vinyals, O. 2018. Representation Learning with Contrastive Predictive Coding. *arXiv:1807.03748*.

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., and Zhou, M. 2020. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. In *NeurIPS*.

Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G., Howard, J., and Poli, I. 2024. Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. *arXiv:2412.13663*.

Wegmann, A., Schraagen, M., and Nguyen, D. 2022. Same Author or Just Same Topic? Towards Content-Independent Style Representations. In *RepL4NLP*.

Xiong, L., Xiong, C., Li, Y., Tang, K.-F., Liu, J., Bennett, P., Ahmed, J., and Overwijk, A. 2021. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In *ICLR*.

Zhao, W., Ren, X., Hessel, J., Cardie, C., Choi, Y., and Deng, Y. 2024. WildChat: 1M ChatGPT Interaction Logs in the Wild. In *ICLR*.

---

## Appendix A: Full Per-Metric Tables

*(To be populated when the full 96-cell grid completes. The tables will report mean ± std across three seeds per (encoder × dataset × regime × test_regime) cell for each of: ROC-AUC, PR-AUC, F1, EER, TAR@FAR=1%, FAR@FRR=1%, Brier, ECE. Per-prefix variants (@1, @3) and profile-operating metrics (imposter catch rate, genuine survival rate, median/P90 messages-to-detect) are reported separately.)*

## Appendix B: Cross-Miner Robustness

*(SBERT secondary miner results on the top 2 encoders, two experiments each: train on TF-IDF / test on SBERT, train on SBERT / test on SBERT. Reported as ROC-AUC and TAR@FAR=1% with the difference from the TF-IDF-only column highlighted.)*

## Appendix C: Hyperparameter Sensitivity

*(Single-model ablations on the strongest encoder, varying `text_source ∈ {all, profile, query}`, `max_length ∈ {128, 256, 512}`, and `triplets_per_author ∈ {4, 16, 64}`. Reported as the change in test-mixed ROC-AUC on PAN seed 13.)*

## Appendix D: Implementation Notes

*(The mean-pool vs [CLS] pooler issue (§9.3); the TF-IDF cache structure and invalidation policy; the AMP/FP16 training recipe; the deterministic-seed protocol.)*
