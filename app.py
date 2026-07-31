import os
import streamlit as st
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RouterQueryEngine, CustomQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool

# Bypass TF warnings
os.environ["USE_TF"] = "0"

# Page Configuration
st.set_page_config(page_title="Enterprise Guardrailed RAG", page_icon="🛡️", layout="wide")
st.title("🛡️ Enterprise Financial RAG (Guardrails & Citations)")
st.caption("Agentic Routing + Anti-Hallucination Guardrails + Detailed Metadata Attribution")

# 1. Automatic Streamlit Secrets & Environment Configuration
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.header("Configuration")
    if not api_key:
        api_key = st.text_input("Groq API Key (Free)", type="password")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=api_key)
else:
    st.warning("⚠️ Please configure your GROQ_API_KEY in Streamlit Secrets or enter it in the sidebar to proceed.")
    st.stop()

# 2. Free Local Embeddings
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 3. Define General Chat fallback engine
class GeneralChatEngine(CustomQueryEngine):
    def custom_query(self, query_str: str):
        llm = Settings.llm
        response = llm.complete(
            f"You are a helpful assistant. Answer this request directly and concisely: {query_str}"
        )
        return str(response)

# 4. Custom System Prompt (Strict Guardrail Template)
GUARDRAIL_PROMPT_TMPL = (
    "Context information is provided below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information above and not prior knowledge, answer the query.\n"
    "Strict Rules:\n"
    "1. Rely ONLY on the clear facts directly mentioned in the context.\n"
    "2. Do NOT extrapolate or speculate on figures or financial trends.\n"
    "3. If the context does not contain the necessary information to answer, state clearly: "
    "'[GUARDRAIL NOTICE]: The provided financial document context does not contain sufficient data to answer this question.'\n\n"
    "Query: {query_str}\n"
    "Answer: "
)

# Cached Router Engine setup
@st.cache_resource
def load_rag_engine():
    # A. Connect to Qdrant Vector Store
    client = QdrantClient(path="./qdrant_db")
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name="sec_filings",
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    
    # B. Document Query Engine Pipeline
    window_postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=3)
    
    doc_query_engine = index.as_query_engine(
        vector_store_query_mode="hybrid",
        similarity_top_k=10,
        sparse_top_k=10,
        node_postprocessors=[window_postproc, reranker]
    )

    # Apply Guardrail Prompt to synthesizing step
    guardrail_prompt = PromptTemplate(GUARDRAIL_PROMPT_TMPL)
    doc_query_engine.update_prompts({"response_synthesizer:text_qa_template": guardrail_prompt})

    # C. Wrap Engines into Query Engine Tools
    doc_tool = QueryEngineTool.from_defaults(
        query_engine=doc_query_engine,
        description=(
            "Useful for answering specific questions about Apple's SEC 10-K financial filing, "
            "revenue figures, risk factors, R&D expenses, or corporate disclosures."
        )
    )

    general_tool = QueryEngineTool.from_defaults(
        query_engine=GeneralChatEngine(),
        description=(
            "Useful for general greetings (hello, hi), casual conversation, "
            "or questions completely unrelated to Apple's financial filings."
        )
    )

    # D. Router Engine Initialization
    router_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(),
        query_engine_tools=[doc_tool, general_tool]
    )

    return router_engine

try:
    query_engine = load_rag_engine()
    st.success("Guardrailed Agentic Engine Ready!", icon="🛡️")
except Exception as e:
    st.error(f"Error loading Vector Store: {e}")
    st.stop()

# Chat Interface Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Prompt Handling
if prompt := st.chat_input("Ask a financial question or greeting..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Applying Guardrails & Synthesizing Answer..."):
            response = query_engine.query(prompt)
            
            # Extract raw response string
            resp_text = str(response.response) if hasattr(response, "response") else str(response)
            st.markdown(resp_text)

            # Display Metadata & Citation Details (If retrieved from Document Engine)
            source_nodes = getattr(response, "source_nodes", None)
            if source_nodes:
                with st.expander("📌 Source Citations & Metadata Attribution"):
                    for idx, node in enumerate(source_nodes, 1):
                        meta = node.node.metadata if hasattr(node, "node") else {}
                        file_name = meta.get("file_name", "SEC_Filing.html")
                        score = getattr(node, "score", 0.0) or 0.0
                        text_snippet = node.node.get_text()[:400] if hasattr(node, "node") else ""
                        
                        st.markdown(f"**Citation {idx}** | **Source File:** `{file_name}` | **Relevance Score:** `{score:.4f}`")
                        st.caption(f"**Retrieved Passages:** {text_snippet}...")
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": resp_text})