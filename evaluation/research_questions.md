# Research Questions (RQs 1-5)

This document formally defines the five research questions (RQs) that are addressed in the evaluation phase of the Hallucination-Aware RAG (HaRAG) pipeline.

---

### RQ1: Factuality Detection Accuracy
> **How accurately does the HaRAG detector identify factuality and hallucination in RAG generations compared to standard baseline classifiers?**
- **Evaluation Subset**: RAGTruth test set, HaluEval balanced sets (QA, Dialogue, Summarization).
- **Key Metrics**: Accuracy, Precision, Recall, F1-Score, AUROC.
- **Goal**: Demonstrate that fusing retrieval quality, evidence span support, and contradiction verification scores yields superior factuality alignment.

---

### RQ2: Entity Resolution Impact
> **What is the impact of entity-level verification / entity resolution on the contradiction verifier's false positive rate?**
- **Evaluation Subset**: Curated fact verification pairs (e.g. SQuAD/FEVER sanity suite).
- **Key Metrics**: Exact Match (EM), token-level overlap recall, contradiction detection specificity.
- **Goal**: Show that verification at the normalized token level prevents false contradictions arising from minor syntactic differences (e.g. "Delhi" vs "New Delhi").

---

### RQ3: Retrieval Quality Improvement
> **Do document chunking and Reciprocal Rank Fusion (RRF) statistically improve retrieval quality?**
- **Evaluation Subset**: SQuAD, FEVER, and RAGTruth validation datasets (separately).
- **Key Metrics**: Recall@1, Recall@5, Recall@10, Mean Reciprocal Rank (MRR), nDCG@10.
- **Statistical Significance**: Paired t-tests on reciprocal ranks and recall indicators.
- **Goal**: Quantify and statistically prove the retrieval performance improvements of chunked RRF retrieval over a baseline passage retriever.

---

### RQ4: Confidence Score Calibration
> **How does temperature scaling calibration impact the reliability of confidence scores (VCS)?**
- **Evaluation Subset**: RAGTruth and TruthfulQA datasets.
- **Key Metrics**: Expected Calibration Error (ECE), Brier Score, reliability diagrams.
- **Goal**: Prove that post-hoc temperature scaling adjusts predicted probabilities to align with actual correctness rates, reducing calibration error.

---

### RQ5: Component Contributions (Ablation)
> **Which components (dense retrieval, evidence highlighter, contradiction verifier, confidence scorer) contribute most to the final hallucination detection performance?**
- **Evaluation Subset**: Ablation study on HaluEval using pipeline variants (Systems A to E).
- **Key Metrics**: Ablation delta in Accuracy, F1-Score, AUROC, ECE, Brier Score.
- **Goal**: Quantify the individual contributions of each step in the retrieval-augmented fact-checking chain.
