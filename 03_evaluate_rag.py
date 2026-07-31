import os
import pandas as pd
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# Bypass TF warnings
os.environ["USE_TF"] = "0"

# 1. Groq Setup
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    groq_key = input("Enter your Groq API Key: ")
    os.environ["GROQ_API_KEY"] = groq_key

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=groq_key)

# 2. Connect to Vector DB
print(" Loading RAG Query Engine...")
client = QdrantClient(path="./qdrant_db")
vector_store = QdrantVectorStore(
    client=client, 
    collection_name="sec_filings",
    enable_hybrid=True,
    fastembed_sparse_model="Qdrant/bm25"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

window_postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=3)

query_engine = index.as_query_engine(
    vector_store_query_mode="hybrid",
    similarity_top_k=10,
    sparse_top_k=10,
    node_postprocessors=[window_postproc, reranker]
)

# 3. Benchmark Questions
eval_questions = [
    "What are the primary risk factors mentioned regarding foreign manufacturing?",
    "How does Apple describe its research and development investments?",
    "What regulatory compliance risks does Apple highlight in this filing?"
]

print(" Querying RAG pipeline and collecting benchmarks...")
records = []

for q in eval_questions:
    response = query_engine.query(q)
    
    # Measure retrieved context size & top reranker relevance score
    top_score = response.source_nodes[0].score if response.source_nodes else 0.0
    num_sources = len(response.source_nodes)
    
    records.append({
        "Question": q,
        "Generated Answer": response.response,
        "Retrieved Sources": num_sources,
        "Top Reranker Score": round(top_score, 4)
    })

# 4. Generate Report
df = pd.DataFrame(records)
print("\n==========================================")
print(" EVALUATION & RETRIEVAL SUMMARY REPORT")
print("==========================================")
print(df[["Question", "Retrieved Sources", "Top Reranker Score"]])

df.to_csv("rag_evaluation_report.csv", index=False)
print("\n Report successfully generated and saved to 'rag_evaluation_report.csv'!")