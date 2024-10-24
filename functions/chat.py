import uuid
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from prompts.chat_prompts import history_context_prompt, system_role_prompt
from prompts.basic_prompts import question_rewriter_system


def get_rag_chain_with_history(retriever, llm, system_prompt=system_role_prompt):
    # Format history prompt
    contextualize_q_system_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", history_context_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Incorporate the retriever into the history question-answering chain.
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_system_prompt
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def get_rag_chain(llm, retriever, system_prompt=system_role_prompt):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Answer the user's question: {input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


def question_rewritter(llm):
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_rewriter_system),
            (
                "human",
                "Here is the initial question: \n\n {input} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()

    return question_rewriter


def generate_response(ai, input, thread_id=str(uuid.uuid4())):
    result = ai.invoke(
        {"input": input},
        config={"configurable": {"thread_id": thread_id}},
    )

    return result
