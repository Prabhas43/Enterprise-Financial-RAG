import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from qdrant_client import QdrantClient

# 1. Set up dense embeddings
print(" Loading Dense FastEmbed Model...")
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. Configure Sentence Window Parser (Parent-Child chunking)
# Searches pinpoint sentences, then expands window by 3 sentences before & after
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# 3. Initialize Qdrant Vector Store with Hybrid Search
client = QdrantClient(path="./qdrant_db")
vector_store = QdrantVectorStore(
    client=client, 
    collection_name="sec_filings",
    enable_hybrid=True,
    fastembed_sparse_model="Qdrant/bm25"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 4. Read documents
print(" Reading SEC documents...")
documents = SimpleDirectoryReader("./data").load_data()

# 5. Build Windowed Nodes
print(" Parsing documents into Sentence Window nodes...")
nodes = node_parser.get_nodes_from_documents(documents)

# 6. Index into Qdrant
print("⚡ Ingesting Parent-Child nodes into Qdrant DB...")
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True
)

print("✅ Upgrade 2 Ingestion Complete! Sentence Window nodes stored in ./qdrant_db")