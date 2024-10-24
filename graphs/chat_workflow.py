import os
import operator
from typing import List, Sequence
from typing_extensions import Annotated, TypedDict

from langchain.schema import Document
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from functions.index import get_index
from functions.chat import get_rag_chain, get_rag_chain_with_history
from graphs.retrieval_grader import retrieval_grader
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        input: user input
        generation: LLM generated answer
        search: whether to add search
        documents: list of documents
        steps: list of processed steps
    """

    input: str
    name: str
    birth_date: str
    generation: str
    context: str
    web_search: str
    steps: List[str] = []
    loop_step: Annotated[int, operator.add]
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    documents: List[str] = []


# Post-processing
index = get_index()
retriever = index["retriever"]
llm = index["llm"]
web_search_tool = TavilySearchResults(k=3)


def get_list(state: GraphState, key):
    if key in state:
        return state[key]
    return []


def retrieve(state: GraphState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    print("\n---RETRIVE---")
    input = state["input"]
    documents = retriever.invoke(input)

    steps = get_list(state, "steps")
    steps.append("retrieve_documents")

    return {"documents": documents, "steps": steps}


def chat(state: GraphState):
    """
    Runs rag chain with history and update the chat history with the user_input and ai response

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New Key added to state, chat_history, that contains the chat history
    """

    print("\n---CHAT WITH STORY---")
    input = state["input"]

    # RAG generation
    rag_chain = get_rag_chain_with_history(retriever=retriever, llm=llm)
    generation = rag_chain.invoke(state)

    steps = get_list(state, "steps")
    steps.append("chat_with_history")

    return {
        "input": input,
        "generation": generation,
        "context": generation["context"],
        "chat_history": [
            HumanMessage(input),
            AIMessage(generation["answer"]),
        ],
        "steps": steps,
    }


def generate(state: GraphState):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    print("\n---GENERATE---")
    input = state["input"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG generation
    rag_chain = get_rag_chain(retriever=retriever, llm=llm)
    generation = rag_chain.invoke(state)

    steps = get_list(state, "steps")
    steps.append("generate_answer")

    return {
        "documents": documents,
        "input": input,
        "generation": generation,
        "steps": steps,
        "loop_step": loop_step + 1,
        "context": generation["context"],
        "chat_history": [
            HumanMessage(input),
            AIMessage(generation["answer"]),
        ],
    }


def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("\n---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    input = state["input"]
    documents = state["documents"]

    steps = get_list(state, "steps")
    steps.append("grade_documents")

    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"input": input, "documents": d.page_content})
        grade = score["score"]
        if grade == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
            continue

    return {
        "documents": filtered_docs,
        "input": input,
        "web_search": web_search,
        "steps": steps,
    }


def web_search(state: GraphState):
    """
    Web search based on the re-phrased input.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("\n---WEB SEARCH---")
    input = state["input"]
    documents = state.get("documents", [])

    steps = get_list(state, "steps")
    steps.append("web_search")

    # Web search
    docs = web_search_tool.invoke({"query": input})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "input": input, "steps": steps}


def decide_to_generate(state: GraphState):
    """
    Determines whether to generate an answer, or re-generate a input.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("\n---ASSESS GRADED DOCUMENTS---")
    input = state["input"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes" and os.getenv("WEB_SEARCH") == "False":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "\n---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("\n---DECISION: GENERATE---")
        return "generate"


# Graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("chat", chat)  # chat with history
# workflow.add_node("generate", generate)  # generatae
workflow.add_node("websearch", web_search)  # web search

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "chat",
    },
)
workflow.add_edge("websearch", "chat")
workflow.add_edge("chat", END)

# Finally, we compile the graph with a checkpointer object.
# This persists the state, in this case in memory.
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
