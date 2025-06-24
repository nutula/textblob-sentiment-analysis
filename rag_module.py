from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv("data/past_quotes.csv")
df['embedding']= df['text'].apply(lambda x: model.encode(x))

def get_similar_text(user_input):
    input_emb = model.encode(user_input)
    similarities = [util.cos_sim(input_emb, emb)[0][0] for emb in df['embedding']]
    top_idx = similarities.index(max(similarities))
    return df['text'][top_idx]