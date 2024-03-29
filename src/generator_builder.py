from src.core.retriever import build_retriever
from src.core.postprocessor import build_context_filter
from src.core.query_engine import build_query_engine

from llama_index.core.indices.base import BaseIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.schema import BaseNode
from llama_index.core import Settings
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from typing import Dict
from pydantic.v1 import BaseModel


class LLMConfig(BaseModel):
    llamaindex_light: LLM
    llamaindex_heavy: LLM
    langchain_light: BaseChatModel
    langchain_heavy: BaseChatModel
    embed_model: BaseEmbedding

    @classmethod
    def from_openai_model_names(cls, light: str, heavy: str, embedding: str):
        return cls(
            llamaindex_light=OpenAI(
                model=light,
            ),
            llamaindex_heavy=OpenAI(
                model=heavy,
            ),
            langchain_light=ChatOpenAI(
                temperature=0.1,
                model_name=light,
            ),
            langchain_heavy=ChatOpenAI(
                temperature=0.1,
                model_name=heavy,
            ),
            embed_model=OpenAIEmbedding(
                model=embedding,
                embed_batch_size=100,
            ),
        )

    @classmethod
    def from_azure_deployments(
        cls, light: str, heavy: str, embedding: str, api_version: str
    ):
        return cls(
            llamaindex_light=AzureOpenAI(
                azure_deployment=light,
                openai_api_version=api_version,
                max_retries=0,
            ),
            llamaindex_heavy=AzureOpenAI(
                azure_deployment=heavy,
                openai_api_version=api_version,
                max_retries=0,
            ),
            langchain_light=AzureChatOpenAI(
                temperature=0.1,
                azure_deployment=light,
                openai_api_version=api_version,
                max_retries=0,
            ),
            langchain_heavy=AzureChatOpenAI(
                temperature=0.1,
                azure_deployment=heavy,
                openai_api_version=api_version,
                max_retries=0,
            ),
            embed_model=AzureOpenAIEmbedding(
                deployment_name=embedding,
                api_version=api_version,
                embed_batch_size=100,
            ),
        )


def build_generator(
    index: BaseIndex,
    full_nodes_dict: Dict[str, BaseNode],
    llm_config: LLMConfig,
    verbose: bool = False,
) -> BaseChatEngine:
    # 1. Retriever that performs search over index
    retriever = build_retriever(
        index=index,
        full_nodes_dict=full_nodes_dict,
        llm=llm_config.llamaindex_light,
        verbose=verbose,
        top_k=10,
    )

    # 2. Postprocessor that removes irrelevant nodes from context
    context_filter = build_context_filter(
        langchain_light=llm_config.langchain_light,
        langchain_heavy=llm_config.langchain_heavy,
    )

    # 3. Query engine that generates an answer based on the context
    query_engine = build_query_engine(
        retriever=retriever,
        llm=llm_config.llamaindex_heavy,
        postprocessors=[context_filter],
        langchain_light=llm_config.langchain_light,
        langchain_heavy=llm_config.langchain_heavy,
    )

    # 4. Chat engine that condences chat history into a single search query
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        llm=llm_config.llamaindex_light,
        verbose=verbose,
    )

    return chat_engine
