"""
BERT-based sentiment analyzer.

Uses a pre-trained DistilBERT model fine-tuned on the SST-2 sentiment dataset.
This is a state-of-the-art transformer model that understands context.

- Model: distilbert-base-uncased-finetuned-sst-2-english
- Size: ~260 MB
- Accuracy: ~91-92% on standard sentiment benchmarks
"""

from transformers import pipeline


class BertSentimentAnalyzer:
    """Wraps a HuggingFace sentiment pipeline for easy batch inference."""

    def __init__(self):
        # Downloads model on first use (~260 MB), cached afterward
        print("Loading BERT model (first run may take a minute)...")
        self.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
        print("BERT model loaded successfully.")

    def predict_batch(self, texts):
        """
        Classify a list of texts.

        Returns:
            list of dicts: [{'label': 'POSITIVE'|'NEGATIVE',
                             'confidence': float (0-100)}, ...]
        """
        results = []
        for text in texts:
            # BERT can only handle up to 512 tokens; truncate very long text
            truncated = text[:1000]
            try:
                output = self.classifier(truncated)[0]
                results.append({
                    "label": output["label"].capitalize(),  # "Positive" / "Negative"
                    "confidence": round(output["score"] * 100, 1),
                })
            except Exception as e:
                results.append({
                    "label": "Error",
                    "confidence": 0.0,
                    "error": str(e),
                })
        return results


# ---------- Self-test ----------
if __name__ == "__main__":
    analyzer = BertSentimentAnalyzer()

    test_sentences = [
        "I absolutely love this product! Best purchase ever.",
        "This is terrible. Worst experience of my life.",
        "Cornell students voice frustration at public hearing over rape case.",
        "Israel honours Captain Smit Machchhar's valour with billboards in Mumbai.",
        "Breaking the blister pack: Counterfeit medicines network busted",
        "PM Modi speaks to Flydubai pilot Captain Smit Machchhar",
    ]

    print("\n=== BERT Predictions ===\n")
    for text, result in zip(test_sentences, analyzer.predict_batch(test_sentences)):
        print(f"[{result['label']:9s} {result['confidence']:5.1f}%] {text}")