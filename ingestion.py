import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv

load_dotenv()

# ========= CONFIG =========
COLLECTION_NAME = "sanjib_portfolio"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ========= EMBEDDING =========
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True}
)

# ========= SECTION HEADERS =========
SECTION_HEADERS = [
    "ABOUT ME",
    "WORK EXPERIENCE",
    "EDUCATION AND TRAINING",
    "PROJECTS",
    "VOLUNTEERING",
    "PUBLICATIONS",
    "CONFERENCES AND SEMINARS",
    "HONOURS AND AWARDS",
    "SKILLS"
]


def split_by_sections(text: str):
    sections = {}
    current_section = None
    buffer = []

    for line in text.splitlines():
        line_clean = line.strip()

        # Name → BASIC INFO
        if line_clean.upper() == "SANJIB SHAH":
            if current_section and buffer:
                sections[current_section] = "\n".join(buffer).strip()
            current_section = "BASIC INFO"
            buffer = []
            continue

        # Known headers
        if line_clean.upper() in SECTION_HEADERS:
            if current_section and buffer:
                sections[current_section] = "\n".join(buffer).strip()
            current_section = line_clean.upper()
            buffer = []
            continue

        if current_section:
            buffer.append(line)

    if current_section and buffer:
        sections[current_section] = "\n".join(buffer).strip()

    return sections


def ingest_cv(cv_path: str):
    print(f"📄 Ingesting: {cv_path}")

    # ========= CONNECT QDRANT =========
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # ========= CREATE COLLECTION =========
    VECTOR_SIZE = 1024

    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print("✅ Collection created.")
    else:
        print("ℹ️ Collection already exists.")

    # ========= LOAD PDF =========
    loader = PyMuPDFLoader(cv_path)
    docs = loader.load()
    full_text = "\n".join(d.page_content for d in docs)

    # ========= SPLIT INTO SECTIONS =========
    sections = split_by_sections(full_text)

    # ========= CHUNK =========
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)

    all_chunks = []
    for section, content in sections.items():
        if not content.strip():
            continue

        chunk_docs = splitter.create_documents(
            texts=[content],
            metadatas=[{"section": section, "source": os.path.basename(cv_path)}]
        )
        all_chunks.extend(chunk_docs)

    print(f"🧩 Created {len(all_chunks)} chunks")

    # ========= EMBED =========
    texts = [c.page_content for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]
    vectors = embeddings.embed_documents(texts)

    # ========= UPSERT =========
    points = []
    for idx, (vector, text, meta) in enumerate(zip(vectors, texts, metadatas)):
        points.append({
            "id": idx,
            "vector": vector,
            "payload": {
                "page_content": text,
                "metadata": meta
            }
        })

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print("🚀 CV uploaded to Qdrant with page_content + metadata payloads!")


