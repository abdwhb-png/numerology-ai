import os
import shutil

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma

from functions.embedding_and_llm import get_embedding, get_llm

KNB_DIR = "knbs"
CHROMA_PATH = "chroma"


def split_docs():
    # Load pdf documents from knbs directory
    loader = PyPDFDirectoryLoader(KNB_DIR)
    docs = loader.load()

    # Initialize a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split the documents into chunks
    chunks = text_splitter.split_documents(docs)

    print(f"Splited {len(docs)} documents into {len(chunks)} chunks.")

    return chunks


# Create a persistent vector Chroma DB
def save_to_chroma():
    print("\nCreating Chroma DB ...")
    # Load the existing database.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding())

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(split_docs())

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"➕ Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("👉 No new documents to add")


def clear_database():
    if os.path.exists(CHROMA_PATH):
        print("Clearing database ...")
        shutil.rmtree(CHROMA_PATH)


def calculate_chunk_ids():
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in split_docs():
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return split_docs()


def get_index():
    try:
        llm = get_llm()["llm"]

        if os.path.exists(CHROMA_PATH):
            print(f"\nLoading Chroma vector DB from : {CHROMA_PATH} ...")
            vector_store = Chroma(
                persist_directory=CHROMA_PATH,
                collection_name="rag-chroma",
                embedding_function=get_embedding(),
            )
        else:
            print("\n❌ No CHROMA_DIR found !")
            save_to_chroma()
            return get_index()

    except Exception as e:
        print(f"\n❌ Error while loading vector DB from Chroma: {e}")

        print("\nCreating in-memory vector store ...")

        vector_store = InMemoryVectorStore.from_documents(
            documents=split_docs(), embedding=get_embedding()
        )

    index = {
        "llm": llm,
        "vector_store": vector_store,
        "retriever": vector_store.as_retriever(),
    }
    print("✅ Successfully loaded vector store !")

    return index
