import gradio as gr

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import sys

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.globals import set_debug


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
        lib_path=Path("../../docs_md"),
        persist_path=persist_path,
        embedding_function=embedding_function,
    )

retriever = build_retriever(
    llm=llm, vectorstore=index.vectorstore, docstore=index.docstore
)

generator = build_generator(llm=llm, retriever=retriever)


def generate(question: str, session_id: str):

    response = generator.stream(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )

    for segment in response:

        message = ""

        if "retrieval_result" in segment:
            if "search_terms" in segment["retrieval_result"]:
                message += (
                    "Looking for: " + segment["retrieval_result"]["search_terms"].query
                )

                if segment["retrieval_result"]["search_terms"].category is not None:
                    message += (
                        " in " + segment["retrieval_result"]["search_terms"].category
                    )

        if "answer" in segment:
            message += segment["answer"].answer

            if segment["answer"].citations:
                for citation in segment["answer"].citations:
                    message += "Source" + citation.source_id + citation.quote

        yield message


demo = gr.ChatInterface(
    fn=generate,
)

if __name__ == "__main__":
    demo.launch(share=True, auth=("edgedb", "<pass>"))
