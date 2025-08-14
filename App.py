import os
import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Wine Expert Chatbot 🍷", page_icon="🍷", layout="wide")

# -------------------- LOAD ENV --------------------
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")  # replace with secrets
QDRANT_URL = os.getenv("QDRANT_URL")  # or your Qdrant cloud URL
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "None")  # if using Qdrant Cloud

# -------------------- INIT MODELS --------------------
@st.cache_resource
def load_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def init_qdrant():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


encoder = load_encoder()
qdrant = init_qdrant()

# -------------------- DATA INGEST --------------------
@st.cache_data
def load_and_index_data():
    data_frame = pd.read_csv("top_rated_wines.csv")  # make sure this file is in your repo
    data_frame = data_frame[data_frame['variety'].notna()]

    # Create collection if not exists
    try:
        qdrant.get_collection("top_wines")
    except:
        qdrant.create_collection(
            collection_name="top_wines",
            vectors_config=models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )

        data = data_frame.to_dict(orient="records")
        points = [
            models.PointStruct(
                id=idx,
                vector=encoder.encode(doc["notes"]).tolist(),
                payload=doc
            ) for idx, doc in enumerate(data)
        ]
        qdrant.upload_points(collection_name="top_wines", points=points)

    return data_frame


df = load_and_index_data()

# -------------------- FUNCTIONS --------------------
def search_wines(query: str, limit=3):
    hits = qdrant.search(
        collection_name="top_wines",
        query_vector=encoder.encode(query).tolist(),
        limit=limit
    )
    return [hit.payload for hit in hits]

def generate_response(user_prompt, context):
    client = OpenAI(api_key=API_KEY)
    messages = [
        {"role": "system", "content": "You are a wine specialist helping users select wines."},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": str(context)}
    ]
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return completion.choices[0].message.content


# -------------------- STREAMLIT UI --------------------
st.title("🍷 Wine Expert Chatbot")
st.caption("Ask me about wines, and I’ll find the perfect one for you!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask me for a wine recommendation...")
if user_input:
    # Store user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching the wine cellar... 🍇"):
            search_results = search_wines(user_input)
            reply = generate_response(user_input, search_results)
            st.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

# Sidebar info
with st.sidebar:
    st.header("About")
    st.write("This chatbot uses Qdrant + Sentence Transformers + OpenAI to recommend wines.")
    st.divider()
    st.write("**Data Source:** `top_rated_wines.csv`")
