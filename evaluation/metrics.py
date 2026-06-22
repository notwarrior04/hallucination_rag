from collections import Counter
from typing import List, Dict
import numpy as np
from evaluation.evaluation_utils import normalize_answer, tokenize_answer, fuzzy_exact_match

def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Computes exact match score (0.0 or 1.0) on normalized texts."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def fuzzy_exact_match_score(prediction: str, ground_truth: str) -> float:
    """Computes fuzzy exact match score (0.0 or 1.0) handling containment and token overlap."""
    return fuzzy_exact_match(prediction, ground_truth)

def token_f1_score(prediction: str, ground_truth: str) -> float:
    """Computes token-level F1 score between prediction and ground truth."""
    prediction_tokens = tokenize_answer(normalize_answer(prediction))
    ground_truth_tokens = tokenize_answer(normalize_answer(ground_truth))
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 1.0 if prediction_tokens == ground_truth_tokens else 0.0
        
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def aggregate_metrics(results: List[Dict]) -> Dict:
    """Aggregates EM (strict + fuzzy), F1, and retrieval recall from detailed results."""
    ems = [r["em"] for r in results]
    fuzzy_ems = [r.get("fuzzy_em", r["em"]) for r in results]
    f1s = [r["f1"] for r in results]
    recalls = [r["retrieval_recall"] for r in results]
    
    return {
        "mean_em": round(float(np.mean(ems)), 4) if ems else 0.0,
        "mean_fuzzy_em": round(float(np.mean(fuzzy_ems)), 4) if fuzzy_ems else 0.0,
        "mean_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "mean_recall": round(float(np.mean(recalls)), 4) if recalls else 0.0
    }
