from transformers import pipeline

_classifier = None

def detect_emotion(text):
    global _classifier
    if _classifier is None:
        _classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    labels = ["joy", "sadness", "anger", "surprise", "fear", "love"]
    result = _classifier(text, labels)
    return dict(zip(result["labels"], result["scores"]))
