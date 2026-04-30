import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

PDF_FOLDER = "rag_files"
FAISS_INDEX_PATH = "faiss_index"

print("Loading PDFs...")
documents = []

for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        filepath = os.path.join(PDF_FOLDER, filename)
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        documents.extend(docs)
        print(f"Loaded: {filename} ({len(docs)} pages)")

print(f"Total pages loaded: {len(documents)}")

print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

print("Creating embeddings and building FAISS index...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local(FAISS_INDEX_PATH)
print(f"Done. FAISS index saved to '{FAISS_INDEX_PATH}/'")