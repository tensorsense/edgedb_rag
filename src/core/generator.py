from langchain.chains import create_history_aware_retriever
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic.v1 import BaseModel, Field
from typing import List, Dict


# types used to prompt and verify model's structured output
class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


# general RAG prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "You answer questions about the database called EdgeDB. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Only answer the question using information you can cite directly from the context. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)


# query contextualization prompt for retriever (last message + history -> search query)
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)


def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\nArticle Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)


def get_sessions(session_store: Dict[str, BaseChatMessageHistory]):
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    return get_session_history


def build_generator(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    get_session_history_callable=None,
    history_factory_config=None,
):
    """Create a Conversation RAG chain to use as a backbone for the chatbot

    Args:
        llm: LangChain ChatModel with function calling support.
        retriever: A runnable that fetches relevant documents based on the query.
        get_session_history_callable: A callable that returns relevant chat history
            that the bot can use as context. The implementation will depend on how a
            particular chat interface handles message history.
        history_factory_config: An object that describes what arguments the chain needs
            to use to fetch correct message history from get_session_history_callable

    """
    # 1. Decorate retriever with history-aware functionality (last message + history -> search query -> docs)
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Build a question answering subchain (search query + docs -> answer with quotes)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: format_docs_with_id(x["retrieval_result"]["documents"]))
        )
        | qa_prompt
        | llm.with_structured_output(QuotedAnswer)
    ).with_config(run_name="question_answer_chain")

    # 3. Join retriever and QA chain together
    rag_chain = (
        RunnablePassthrough.assign(retrieval_result=history_aware_retriever)
        .assign(answer=question_answer_chain)
        .with_config(run_name="rag_chain")
    )

    # 4. Verify or set up session history fetching
    if get_session_history_callable is None:
        get_session_history_callable = get_sessions(session_store={})
        history_factory_config = [
            ConfigurableFieldSpec(
                id="session_id",
                annotation=str,
                name="Session ID",
                description="Identifier of the session",
                default="0",
                is_shared=True,
            ),
        ]
    else:
        assert history_factory_config is not None

    # 5. Decorate RAG chain with message history functionality
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history_callable,
        history_factory_config=history_factory_config,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain
