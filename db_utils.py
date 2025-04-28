import os
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

print("Qdrant_url", st.secrets["QDRANT_URL"])
print("Qdrant_api_key",st.secrets["QDRANT_API_KEY"])
print("grok_api", st.secrets['GROQ_API_KEY'])

qdrant = QdrantClient(
    qdrant_url = st.secrets["QDRANT_URL"],
    qdrant_api_key = st.secrets["QDRANT_API_KEY"]
)

model = SentenceTransformer('all-MiniLM-L6-v2')

def setup_collection(collection_name="docs"):
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

def insert_documents(docs, collection_name="docs"):
    vectors = model.encode(docs).tolist()
    payload = [{"text": doc} for doc in docs]
    qdrant.upsert(
        collection_name=collection_name,
        points=[
            {"id": i, "vector": vector, "payload": payload[i]}
            for i, vector in enumerate(vectors)
        ]
    )


def search(query, collection_name="docs", top_k=5):
    vector = model.encode(query).tolist()
    search_result = qdrant.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k
    )
    return [hit.payload['text'] for hit in search_result]