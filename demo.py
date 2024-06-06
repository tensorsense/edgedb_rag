import gradio as gr

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from tqdm import tqdm
import sys
from llama_index.core.llms import ChatMessage, MessageRole


_ = load_dotenv(find_dotenv()) 


sys.path.append(Path("src").resolve().as_posix())

from src.generator_builder import LLMConfig

llms = LLMConfig.from_azure_deployments(
    light="gpt-35-turbo-1106",
    heavy="gpt4o",
    embedding="text-embedding-ada-002",
    api_version="2023-07-01-preview",
)
from llama_index.core import Settings

Settings.embed_model = llms.embed_model
Settings.llm = llms.llamaindex_light


from src.index_builder import build_index

index, full_nodes_dict = build_index(
    persist_path = Path("./index_storage_updated").resolve(),
    collection_name = "index",
    lib_path = Path("../../docs_md").resolve()
)

from src.generator_builder import build_generator

generator = build_generator(
    index=index, full_nodes_dict=full_nodes_dict, llm_config=llms
)


def answer(question: str, history):

    history_llamaindex = []
    for human, ai in history:
        history_llamaindex.append(ChatMessage(content=human, role=MessageRole.USER))
        history_llamaindex.append(ChatMessage(content=ai, role=MessageRole.ASSISTANT))

    response = generator.stream_chat(message=question, chat_history=history_llamaindex)
    message = ''
    for token in response.response_gen:
        message += token
        yield message


demo = gr.ChatInterface(
    fn=answer,
)

if __name__ == '__main__':
    demo.launch(share=True, auth=("edgedb", "<pass>"))
