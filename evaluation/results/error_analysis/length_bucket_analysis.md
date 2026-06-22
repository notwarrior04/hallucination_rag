# Length-based Detection Performance Analysis

This report analyzes how hallucination detection performance scales with generated response length (estimated in whitespace-split tokens).

## HaluEval Length Buckets

| Response Length | Sample Count | Accuracy | F1-Score | AUROC |
| --- | --- | --- | --- | --- |
| 0-50 tokens | 8466 | 0.6601 | 0.7288 | 0.7655 |
| 50-100 tokens | 1405 | 0.6940 | 0.8063 | 0.5899 |
| 100-200 tokens | 130 | 0.8077 | 0.8908 | 0.8171 |
| 200+ tokens | 1 | 1.0000 | 1.0000 | 0.5000 |

## RAGTruth Length Buckets

| Response Length | Sample Count | Accuracy | F1-Score | AUROC |
| --- | --- | --- | --- | --- |
| 0-50 tokens | 55 | 0.4182 | 0.3846 | 0.5669 |
| 50-100 tokens | 408 | 0.4877 | 0.3724 | 0.5622 |
| 100-200 tokens | 387 | 0.4806 | 0.3232 | 0.5137 |
| 200+ tokens | 50 | 0.5800 | 0.3226 | 0.5190 |

