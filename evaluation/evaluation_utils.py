import re
import string

def normalize_answer(s: str) -> str:
    """Standard normalization: lowercasing, removing punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def tokenize_answer(s: str) -> list:
    """Split normalized text into token list."""
    return s.split()

def contains_answer(expected: str, doc_text: str) -> bool:
    """
    Checks if all normalized tokens of expected (ignoring common articles/stopwords) 
    are present in the normalized doc_text. Handles multi-word answers by falling back 
    to the last token (usually surname or core noun).
    """
    norm_expected = normalize_answer(expected)
    norm_doc = normalize_answer(doc_text)
    
    expected_tokens = tokenize_answer(norm_expected)
    doc_tokens = set(tokenize_answer(norm_doc))
    
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
        "for", "with", "by", "of", "about", "is", "was", "were", "are"
    }
    
    # Filter expected tokens to check only meaningful non-stopword tokens
    meaningful_expected = [t for t in expected_tokens if t not in stopwords]
    
    if not meaningful_expected:
        # Fallback if expected contains only stopwords
        meaningful_expected = expected_tokens
        
    if not meaningful_expected:
        return False
        
    # 1. Full token subset match
    if all(token in doc_tokens for token in meaningful_expected):
        return True
        
    # 2. Fallback for multi-word entities (check if last token like surname is present)
    if len(meaningful_expected) > 1:
        if meaningful_expected[-1] in doc_tokens:
            return True
            
    return False

def fuzzy_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Fuzzy exact match that handles containment and high token overlap.
    Returns 1.0 if match found, 0.0 otherwise.
    
    Handles cases like:
      - "Vistula River" vs "Vistula" (containment)
      - "Japanese attack on Pearl Harbor" vs "Attack on Pearl Harbor" (token overlap)
      - "Roman Republic" vs "Roman Empire" (low overlap → 0.0)
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    
    # 1. Standard exact match
    if pred_norm == gt_norm:
        return 1.0
    
    # 2. Containment (either direction)
    if pred_norm and gt_norm:
        if pred_norm in gt_norm or gt_norm in pred_norm:
            return 1.0
    
    # 3. High token overlap (>= 80% of shorter answer's tokens are in the longer)
    pred_tokens = set(pred_norm.split())
    gt_tokens = set(gt_norm.split())
    if pred_tokens and gt_tokens:
        overlap = pred_tokens & gt_tokens
        shorter_len = min(len(pred_tokens), len(gt_tokens))
        if shorter_len > 0 and len(overlap) / shorter_len >= 0.8:
            return 1.0
    
    return 0.0
