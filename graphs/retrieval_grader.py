### Retrieval Grader

from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field

from functions.embedding_and_llm import gemini_llm
from prompts.basic_prompts import retrieval_grader_system as system_prompt


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
llm = gemini_llm
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
