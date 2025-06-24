import streamlit as st
from sentiment_module import analyze_sentiment
from rag_module import get_similar_text
from summarize_module import get_summary
from emotion_module import detect_emotion

st.title("AI-Powered Text Analyzer")

text_input = st.text_area("Enter your text here")

if st.button("Analyze"):
    sentiment = analyze_sentiment(text_input)
    summary = get_summary(text_input)
    emotion = detect_emotion(text_input)
    similar = get_similar_text(text_input)

    st.subheader("Sentiment Analysis")
    st.write(sentiment)

    st.subheader("Summary")
    st.write(summary)

    st.subheader("Emotion Score")
    st.write(emotion)

    st.subheader("Similar Past Statement")
    st.write(similar)