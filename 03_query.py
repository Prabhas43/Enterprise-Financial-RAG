import os
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import LLMRerank

# Set your API key (or use local Ollama / Groq)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 1. Connect to the existing Qdrant vector store
client = QdrantClient(path="./qdrant_db")
vector_store = QdrantVectorStore(client=client, collection_name="sec_filings")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 2. Re-load index from vector store
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)

# 3. Add a Reranker to get high-precision context (top 3 results)
reranker = LLMRerank(top_n=3)

query_engine = index.as_query_engine(
    similarity_top_k=10,        # First fetch top 10 relevant chunks
    node_postprocessors=[reranker] # Refine down to top 3 using LLM Reranking
)

# 4. Ask complex financial questions
response = query_engine.query("What are the primary risk factors mentioned for Apple's supply chain?")

print("\n--- ANSWER ---")
print(response)

print("\n--- SOURCES USED ---")
for node in response.source_nodes:
    print(f"\nScore: {node.score}\nContent: {node.node.get_text()[:200]}...")