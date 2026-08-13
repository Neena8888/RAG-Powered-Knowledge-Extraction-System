"""
Sentiment Analysis and Evaluation Module.
Uses lexicon-based sentiment scoring and provides accuracy evaluation against ground-truth labels.
"""

import re
import pandas as pd


def compute_rule_sentiment(text_content):
    """
    Computes sentiment orientation (Positive, Neutral, Negative)
    using lightweight lexicon polarity rules.
    """
    if not isinstance(text_content, str) or not text_content.strip():
        return "Neutral"

    pos_lexicon = {
        "good", "great", "excellent", "efficient", "optimal", 
        "advanced", "robust", "secure", "innovative", "accurate",
        "performance", "success", "effective", "powerful", "reliable"
    }
    neg_lexicon = {
        "bad", "error", "vulnerability", "slow", "failure", 
        "deprecated", "flaw", "bug", "attack", "issue", 
        "insecure", "bottleneck", "latency", "risk", "defect"
    }

    tokens = re.findall(r'\b\w+\b', text_content.lower())
    
    pos_score = sum(1 for w in tokens if w in pos_lexicon)
    neg_score = sum(1 for w in tokens if w in neg_lexicon)

    if pos_score > neg_score:
        return "Positive"
    elif neg_score > pos_score:
        return "Negative"
    else:
        return "Neutral"


def evaluate_sentiment_accuracy():
    """
    Evaluates the sentiment classification accuracy on a small manually labeled validation set.
    """
    # Small manually labeled evaluation dataset
    validation_dataset = [
        {"text": "The framework provides optimal performance and highly robust security features.", "label": "Positive"},
        {"text": "Severe memory leak and latency bottlenecks cause system crashes under heavy load.", "label": "Negative"},
        {"text": "Relational databases organize data into rows, columns, and primary key structures.", "label": "Neutral"},
        {"text": "This advanced deep learning architecture achieved accurate classification results.", "label": "Positive"},
        {"text": "Unencrypted API keys expose critical vulnerabilities to external network attacks.", "label": "Negative"},
        {"text": "The computer network uses standardized TCP/IP communication protocols.", "label": "Neutral"}
    ]

    correct_predictions = 0
    print("\n--- SENTIMENT ACCURACY EVALUATION ---")
    
    for idx, item in enumerate(validation_dataset, 1):
        pred = compute_rule_sentiment(item["text"])
        is_correct = (pred == item["label"])
        if is_correct:
            correct_predictions += 1
        
        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"[{idx}] Text snippet: '{item['text'][:45]}...'")
        print(f"    Expected: {item['label']} | Predicted: {pred} -> {status}")

    accuracy_score = (correct_predictions / len(validation_dataset)) * 100
    print(f"\nFinal Validation Accuracy: {accuracy_score:.2f}%\n")
    return accuracy_score


if __name__ == "__main__":
    evaluate_sentiment_accuracy()