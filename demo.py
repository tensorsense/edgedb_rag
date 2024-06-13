import gradio as gr

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import sys

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.globals import set_debug
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

set_debug(False)
_ = load_dotenv(find_dotenv())
sys.path.append(Path("src").resolve().as_posix())

from src.core.index import Index, ID_KEY
from src.core.retriever import build_retriever
from src.core.generator import build_generator

llm = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment="gpt4o",
    openai_api_version="2023-07-01-preview",
    max_retries=0,
)

embedding_function = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002", api_version="2023-07-01-preview"
)

persist_path = Path("index_storage").resolve()

if persist_path.exists():
    index = Index.from_persist_path(
        persist_path=persist_path,
        embedding_function=embedding_function,
    )
else:
    index = Index.from_metadata(
        metadata_path=Path("resources/doc_metadata.jsonl"),
        lib_path=Path("../docs_md"),
        persist_path=persist_path,
        embedding_function=embedding_function,
    )

retriever = build_retriever(
    llm=llm, vectorstore=index.vectorstore, docstore=index.docstore
)


def parse_history(raw_history):
    history = ChatMessageHistory()
    for human, ai in raw_history:
        history.add_user_message(human)
        history.add_ai_message(ai)

    return history


generator = build_generator(llm=llm, retriever=retriever, get_session_history_callable=parse_history)


def build_message(message_parts):
    message = ""
    message += "Looking for: " + message_parts["search_query"] 
    message += " in " + message_parts["search_category"] if len(message_parts["search_category"]) > 0 else " everywhere"
    message += "\n\n"

    message += message_parts["answer"] + "\n\n"
    
    message += "Cited sources:\n"
    message += message_parts["citations"] + "\n\n"

    message += "All fetched sources:\n"
    message += message_parts["all_sources"]

    return message


def generate(question: str, history):

    response = generator.stream(
        {"input": question},
        config={"configurable": {"raw_history": history}},
    )

    message_parts = {
        "search_query": "",
        "search_category": "",

        "answer": "",
        "citations": "",
        "all_sources": "",
    }

    for segment in response:
        if "retrieval_result" in segment:
            if "documents" in segment["retrieval_result"]:
                document_part = ""
                
                for doc in segment["retrieval_result"]["documents"]:
                    document_part += f"{doc.metadata['source']}\n"

                message_parts["all_sources"] = document_part

            if "search_terms" in segment["retrieval_result"]:
                message_parts["search_query"] = segment["retrieval_result"]["search_terms"].query
                
                if segment["retrieval_result"]["search_terms"].category is not None:
                    message_parts["search_category"] = segment["retrieval_result"]["search_terms"].category

        if "answer" in segment:
            message_parts["answer"] = segment["answer"].answer

            if segment["answer"].citations:
                citation_part = ""
                for citation in segment["answer"].citations:
                     citation_part += "Source " + str(citation.source_id) + "\n\n"
                     citation_part += citation.quote + "\n\n"

                message_parts["citations"] = citation_part

        yield build_message(message_parts)


demo = gr.ChatInterface(
    fn=generate,
)

if __name__ == "__main__":
    demo.launch(share=True, auth=("edgedb", "<pass>"))
