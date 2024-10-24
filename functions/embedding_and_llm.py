import os

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)


def get_embedding(multilingual: bool = True):
    if multilingual is True:
        embd = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    elif os.getenv("APP_ENV") == "production":
        embd = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
    else:
        embd = OllamaEmbeddings(model="llama3.1")

    return embd


def get_llm():
    if os.getenv("APP_ENV") == "production":
        model_tested = "llama3-8b-8192"
        llm = ChatGroq(model=model_tested)
    else:
        model_tested = "llama3.1"
        llm = OllamaLLM(model=model_tested)

    return {"llm": llm, "model_tested": model_tested}
