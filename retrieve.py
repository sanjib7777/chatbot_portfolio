import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from openai import OpenAI
from qdrant_client.http.models import Distance, VectorParams
from embedding import embeddings
# ========= LOAD ENV =========
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

COLLECTION_NAME = "sanjib_portfolio"

# ========= EMBEDDING =========
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name="BAAI/bge-m3",
#     encode_kwargs={"normalize_embeddings": True}
# )

# ========= CONNECT QDRANT =========
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ========= CONNECT GROQ (OpenAI-Compatible) =========
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print("✅ Collection created.")
else:
        print("ℹ️ Collection already exists.")

portfolio_store = QdrantVectorStore( client=qdrant, collection_name=COLLECTION_NAME, embedding=embeddings )
def retrieve_context(query, top_k=5):
    docs = portfolio_store.similarity_search(query, k=top_k)
   
    contexts = []
    for d in docs:
        contexts.append({
            "text": d.page_content,
            "section": d.metadata.get("section", ""),
            "source": d.metadata.get("source", "")
        })

    return contexts



def answer_with_context(question):
    contexts = retrieve_context(question)
   

    context_text = "\n\n".join(
        [f"[{c['section']}] {c['text']}" for c in contexts]
    )

    prompt = f"""
You are an AI assistant answering questions about Sanjib Shah using ONLY the context below.

Context:
{context_text}

Question:
{question}

Rules:
- If the answer is not in the context, say "I don't have that information about it yet."
- Be concise and professional.
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a helpful portfolio assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content



