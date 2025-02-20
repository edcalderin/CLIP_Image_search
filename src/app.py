from pathlib import Path

import clip
import joblib
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics.pairwise import cosine_similarity

torch.classes.__path__ = []


def load_embeddings():
    try:
        with open(Path(__file__).parent.parent / "embeddings.pkl", "rb") as embeddings:
            return joblib.load(embeddings)
    except FileNotFoundError:
        return {}


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

image_embeddings = load_embeddings()
st.header("Image Search App")
search_term = f"a picture of {st.text_input('Search: ')}"
tokenized_text = clip.tokenize(search_term).to(device)
search_embedding = model.encode_text(tokenized_text).cpu().detach().numpy()

st.sidebar.header("App Settings")
top_number = st.sidebar.slider("Number of Search Results", min_value=1, max_value=30)
picture_width = st.sidebar.slider("Picture Width", min_value=100, max_value=500)

rank_results: list = []
for path, embedding in image_embeddings.items():
    embedding_2d = embedding.reshape(1, -1)
    search_embedding_2d = search_embedding.reshape(1, -1)
    score = cosine_similarity(embedding_2d, search_embedding_2d).flatten().item()
    rank_results.append((path, score))

df_rank = pd.DataFrame(rank_results, columns=["image_path", "cosine_similarity"])

df_rank.sort_values(
    "cosine_similarity", ascending=False, ignore_index=True, inplace=True
)

col1, col2, col3 = st.columns(3)

df_result = df_rank.head(top_number)
for i in range(top_number):
    if i % 3 == 0:
        with col1:
            st.image(df_result.loc[i, "image_path"], width=picture_width)
    elif i % 3 == 1:
        with col2:
            st.image(df_result.loc[i, "image_path"], width=picture_width)
    elif i % 3 == 2:
        with col3:
            st.image(df_result.loc[i, "image_path"], width=picture_width)
