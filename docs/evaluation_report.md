# IWT Detection System — Model Evaluation Report
**Project:** Illegal Wildlife Trade Detection (SUSTAINOVA 2026, SDG 15)
**Date:** 2026-11-19 13:27:26 UTC
**Evaluator:** Arijeet Singh (ML Backend Lane)
**Architecture:** TF-IDF (20,000 max features) + Balanced Logistic Regression
**Dataset:** WLT-OSN (Wildlife Trade Open Social Network, 8,647 deduplicated posts)

---

## 1. Executive Summary

This report establishes the verified empirical benchmark for the text classification
engine of the Illegal Wildlife Trade (IWT) detection system. Evaluation was executed
on a rigorously partitioned, held-out test split of social media listings.

> **Integrity Statement:**
> *These results were obtained on a held-out test set.
> No test data was used in training or model selection.*

The system achieves high discriminative power under extreme class imbalance
(~2.9% positive prevalence), obtaining a **PR-AUC of 0.9112** and an
**F1-score of 0.8382** (86.3% recall at 81.5% precision).

---

## 2. Methodology & Partitioning Protocol

### 2.1 Group-Aware Splitting
In online wildlife trade networks, listings often originate from connected social
accounts or collection clusters. Standard random splitting causes massive data leakage
where seller-specific idiosyncrasies inflate performance.

- **Split Protocol:** `GroupShuffleSplit` on `source_group`
- **Test Ratio:** 20% of groups held out
- **Random Seed:** 0 (fully reproducible)
- **Leakage Verification:** 0% overlap of source groups between partitions

### 2.2 Test Partition Composition
- **Total Test Samples:** 1,314
- **Positive Trade Instances:** 117 (8.90%)
- **Negative / Benign Instances:** 1197 (91.10%)

---

## 3. Test Set Performance Metrics

| Metric | Score | Benchmark Target | Status |
|:---|:---:|:---:|:---:|
| **PR-AUC** | **0.9112** | ≥ 0.8500 | Passed |
| **Precision** | **0.8145** (81.5%) | ≥ 0.7500 | Passed |
| **Recall (Sensitivity)** | **0.8632** (86.3%) | ≥ 0.8000 | Passed |
| **F1 Score** | **0.8382** | ≥ 0.8000 | Passed |

*Classification threshold = 0.50*

### 3.1 Confusion Matrix

| | Pred Negative (0) | Pred Positive (1) | Total Actual |
|---|:---:|:---:|:---:|
| **Actual Negative** | 1174 (TN) | 23 (FP) | 1197 |
| **Actual Positive** | 16 (FN) | 101 (TP) | 117 |
| **Total Predicted** | 1190 | 124 | 1314 |

- **False Positive Rate:** 1.92%
- **False Negative Rate:** 13.68%

---

## 4. Precision-Recall Curve

The Precision-Recall curve illustrates trade-offs across decision thresholds:

![Precision-Recall Curve](file:///Users/arijeetsingh/Biodiversity-/data/processed/pr_curve.png)

*(Generated at `data/processed/pr_curve.png`)*

---

## 5. Performance by Source Group Breakdown

Performance across top source groups with ≥5 samples in the held-out test split:

| Source Group | Samples | Positives | Precision | Recall | F1 Score |
|---|:---:|:---:|:---:|:---:|:---:|
| `seed_0` | 130 | 106 | 1.0000 | 0.8774 | 0.9347 |
| `follower_69` | 96 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `following_41` | 83 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `following_39` | 78 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `follower_58` | 75 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `following_37` | 73 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `follower_37` | 67 | 2 | 1.0000 | 1.0000 | 1.0000 |
| `follower_14` | 65 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `follower_24` | 57 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `follower_79` | 56 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `follower_31` | 53 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `following_24` | 52 | 1 | 1.0000 | 1.0000 | 1.0000 |
| `follower_41` | 37 | 1 | 0.0000 | 0.0000 | 0.0000 |
| `follower_20` | 36 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `follower_102` | 31 | 0 | 1.0000 | 1.0000 | 1.0000 |

---

## 6. Qualitative Failure Mode Analysis

Understanding failure modes is key for deploying human-in-the-loop review queues.

### 6.1 Top False Positives (Benign Classified as Suspicious)
Benign listings scored high by the classifier typically contain antique descriptions,
non-wildlife ivory descriptors (e.g., piano keys), or art terminology:

**FP 1 (Score: 0.8287)**
> "Women's Valentino Miss V Yellow and Ivory Blazer via {{MENTION}} {{URL}}..."
*Analysis:* Lexical tokens shared with trade listings without biological provenance.

**FP 2 (Score: 0.7012)**
> "Vintage Ivory Shadowline Camisole Lace by AlizarinsAntiques {{URL}} via {{MENTION}}..."
*Analysis:* Lexical tokens shared with trade listings without biological provenance.

**FP 3 (Score: 0.6880)**
> "Vintage Ivory Shadowline Camisole Lace Brocade - Size Small Gorgeous by AlizarinsAntiques: 20.00 USDThis camis..."
*Analysis:* Lexical tokens shared with trade listings without biological provenance.

**FP 4 (Score: 0.6793)**
> "Vintage Ivory Wedding Negligee by Cinema by AlizarinsAntiques via {{MENTION}}..."
*Analysis:* Lexical tokens shared with trade listings without biological provenance.

**FP 5 (Score: 0.6691)**
> "Vintage Ivory Wedding Negligee by Cinema by AlizarinsAntiques via {{MENTION}} {{URL}}..."
*Analysis:* Lexical tokens shared with trade listings without biological provenance.

**FP 6 (Score: 0.6669)**
> "Lovely Vintage Cutwork and Embroidered Tablecloth with Matching Napkins - Ivory Floral Full Cutwork with Embro..."
*Analysis:* Lexical tokens shared with trade listings without biological provenance.

### 6.2 Top False Negatives (Suspicious Classified as Benign)
Trade listings missed by the lexical baseline typically rely on novel slang,
obfuscated keywords, or sparse captions where signal is primarily in an image:

**FN 1 (Score: 0.0287)**
> "Vintage Valentine Cat Mug, Otagiri Cat Mugs, Kittens, Tabby Cat, I Heart Cats, Cat Lovers, I Love Cats T-Shirt..."
*Analysis:* Implicit trade intent with minimal overt indicator words; requires multimodal CLIP or deep semantic contextualization.

**FN 2 (Score: 0.0392)**
> "Did you know 2017 is the year of the fire #rooster in the #Chinese calendar? Going at a #giveaway price at {{U..."
*Analysis:* Implicit trade intent with minimal overt indicator words; requires multimodal CLIP or deep semantic contextualization.

**FN 3 (Score: 0.0453)**
> "{{URL}}..."
*Analysis:* Implicit trade intent with minimal overt indicator words; requires multimodal CLIP or deep semantic contextualization.

**FN 4 (Score: 0.0564)**
> "Who can guess what this piece of art is? I bet you'll not get it right. Click on the link below to f {{URL}}..."
*Analysis:* Implicit trade intent with minimal overt indicator words; requires multimodal CLIP or deep semantic contextualization.

**FN 5 (Score: 0.0667)**
> "Offer Link: {{URL}} #necklace #jewelry #earrings #bracelet #pendant #jewellery #accessories #ring #choker #sil..."
*Analysis:* Implicit trade intent with minimal overt indicator words; requires multimodal CLIP or deep semantic contextualization.

**FN 6 (Score: 0.0678)**
> "Impressive 19th Ce. Meissen #Porcelain of a #family having a good time under the tree. Available at {{URL}} #a..."
*Analysis:* Implicit trade intent with minimal overt indicator words; requires multimodal CLIP or deep semantic contextualization.

---

## 7. Operational Recommendations for Human-in-the-Loop Review

1. **Dual-Threshold Flagging Strategy:**
   - **Score ≥ 0.75:** High-confidence flag (`CRITICAL` severity) → immediate queue.
   - **0.30 ≤ Score < 0.75:** Elevated inquiry (`MANUAL REVIEW`) → batch triaging.
   - **Score < 0.30:** Cleared (`NO CLEAR EVIDENCE DETECTED`).

2. **Multimodal Reinforcement:**
   - Listings scoring in 0.30–0.50 range rely on companion CLIP image classifier
     (0.6 text + 0.4 image fusion) to resolve ambiguity.

3. **Domain Knowledge Hard Rules:**
   - CITES Appendix I species and India Wildlife Protection Act Schedule I terms
     floor the confidence to trigger manual human review.
