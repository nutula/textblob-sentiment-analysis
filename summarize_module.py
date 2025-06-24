from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def get_summary(text):
    return summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']