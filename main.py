import streamlit as st
from pymongo import MongoClient
import gridfs
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv
import tempfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()


@st.cache_resource
def initialize_mongodb():
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI not found in environment variables")

    client = MongoClient(mongo_uri)
    try:
        client.admin.command("ismaster")
        print("[DEBUG] Successfully connected to MongoDB")
    except ConnectionFailure as e:
        print(f"[ERROR] Failed to connect to MongoDB: {e}")
        raise

    return client


client = initialize_mongodb()
db_name = "Lorenzo"
collection_name = "RAG"

db = client[db_name]
collection = db[collection_name]
fs = gridfs.GridFS(db)


@st.cache_resource
def initialize_embed_model():
    try:
        model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100)
        print("[DEBUG] OpenAIEmbedding model initialized successfully")
        return model
    except Exception as e:
        print(f"[ERROR] OpenAIEmbedding initialization failed: {e}")
        raise


embed_model = initialize_embed_model()

Settings.embed_model = embed_model
Settings.chunk_size = 100
Settings.chunk_overlap = 10


@st.cache_resource
def initialize_vector_store():
    try:
        vector_store = MongoDBAtlasVectorSearch(
            mongo_client=client,
            db_name=db_name,
            collection_name=collection_name,
            vector_index_name="vector_index",
            exact="True",
        )
        print("[DEBUG] MongoDBAtlasVectorSearch initialized successfully")
        return vector_store
    except Exception as e:
        print(f"[ERROR] Vector store initialization failed: {e}")
        raise


vector_store = initialize_vector_store()
storage_context = StorageContext.from_defaults(vector_store=vector_store)


def process_pdf(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name

        print(f"[DEBUG] Temporary file created at {temp_file_path}")

        documents = SimpleDirectoryReader(input_files=[temp_file_path]).load_data()
        print(f"[DEBUG] Documents loaded: {len(documents)}")

        vector_store_index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=True
        )
        vector_store_index.storage_context.persist()

        print("[DEBUG] VectorStoreIndex created and saved successfully")
        os.unlink(temp_file_path)
        print("[DEBUG] Temporary file deleted")

        return len(documents)
    except Exception as e:
        print(f"[ERROR] PDF processing failed: {e}")
        raise


def check_vector_store():
    try:
        count = collection.count_documents({})
        print(f"[DEBUG] Number of documents in vector store: {count}")
    except Exception as e:
        print(f"[ERROR] Document count failed: {e}")


def load_index():
    try:
        print("[DEBUG] Loading VectorStoreIndex from vector store")
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        return index
    except Exception as e:
        print(f"[ERROR] Index loading failed: {e}")
        return None


def search_and_answer_with_top_blocks(query, index):
    try:
        print(f"[DEBUG] Received query: {query}")
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            verbose=True,
        )
        response = query_engine.query(query)
        print(f"[DEBUG] Response received: {response.response}")

        # Extract text from the top 3 nodes (blocks) to avoid displaying metadata
        top_nodes = response.source_nodes
        top_blocks = []
        for i, node_with_score in enumerate(top_nodes[:3], start=1):
            # Each node contains the actual text in node_with_score.node.get_content()
            content = node_with_score.node.get_content()
            top_blocks.append(content)

        return response.response, top_blocks
    except Exception as e:
        print(f"[ERROR] Query processing failed: {e}")
        return "Error: Unable to process the query", []


st.title("RAG Application with Streamlit, MongoDB, and LlamaIndex")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        try:
            num_documents = process_pdf(uploaded_file)
            st.success(
                f"PDF processed and saved successfully. Documents processed: {num_documents}"
            )
            check_vector_store()
        except Exception as e:
            st.error(f"Error processing the PDF: {e}")

index = load_index()

query = st.text_input("Ask a question about the uploaded documents:")
if query and index:
    with st.spinner("Searching..."):
        answer, top_blocks = search_and_answer_with_top_blocks(query, index)
        if "Error" in answer:
            st.error(answer)
        else:
            st.write("**Answer:**", answer)
            if top_blocks:
                st.write("**Top 3 Blocks Used (Sources):**")
                for i, block in enumerate(top_blocks, 1):
                    # Display only the content as text, without metadata
                    st.markdown(f"**Block {i}:**\n```\n{block.strip()}\n```")
            else:
                st.write("No sources found.")
