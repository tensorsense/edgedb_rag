from typing import List, Optional
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.llms import LLM

from langchain_core.runnables import RunnableSerializable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


TEMPLATE = """You are a programming assistant who specializes in working with EdgeDB.
    You have expert knowledge about EdgeDB and its integrations, as well as EdgeQL and SDL.
    You are answering questions to help users learn EdgeDB using the official documentation.

    The automated seach system has provided you with context information made of pieces of the documentation below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge about syntax and operators, answer the query.
    
    If the context information does not contain enough information to answer the query, \
    reply with "I cound not find relevant EdgeDB documentation on the subject".

    If the user is asking about certain syntax or operator as if it exists, but no such thing can be found in the context, \
    please clarify the confusion by saying "I have no relevant information about [subject]". \
    Next, offer a solution that strictly follows the context if it is possible, otherwise say nothing at all.

    Always assume the question needs to be answered with EdgeQL and SDL, \
    unless the user's query contains or explicitly asks for another language, \
    even if you find examples in other languages within the context.

    Try to always illustrate your answer with a pair of an example schema and an example query corresponding to it.

    User query: {query_str}
    
    Answer:
    """


class VerifiedResponse(BaseModel):
    references: List[str] = Field(
        description="List of names of pieces of context that were used to produce the response. \
            For every factual statement in the response, try to find the name of the context from which the information came. \
            Can be empty."
    )
    is_faithful: bool = Field(
        description="Based on the references, determine if the response can be considered faithful to the context overall."
    )
    assistant_response: str = Field(
        description="Final system response that is going to be sent to the user. \
            If the response is correct and faithful, reproduce it verbatim. \
            If minor mistakes are present, fix them. \
            If the response is not faithful, replace it with 'Sorry, the system was unable to produce a faithful response'"
    )
    reasoning: Optional[str] = Field(
        description="Additional explanation for the outcome of the verification. Can be empty."
    )


def build_response_chain(
    langchain_light: BaseChatModel, langchain_heavy: BaseChatModel
) -> RunnableSerializable:
    system_template = """You are the assistant that helps rate faithfulness of EdgeDB QA system responses.
        You will be provided a user query denoted by ***, and the QA system response denoted by <<>>.
        You will also receive context made of pieces of official EdgeDB documentation denoted by ---.
        Your job is to determine whether all of the information in the response is present in the context.

        {format_instructions}
        """

    human_template = """***{query}***
        <<{response}>>
        ---{context}---
        """

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=VerifiedResponse)

    system_message = SystemMessagePromptTemplate.from_template(
        template=system_template,
        # partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    human_message = HumanMessagePromptTemplate.from_template(template=human_template)

    prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            human_message,
        ],
    )

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    chain_light = prompt | langchain_light | parser
    chain_heavy = prompt | langchain_heavy | parser
    chain = chain_light.with_fallbacks([chain_heavy])

    return chain


class FilteredQueryEngine(RetrieverQueryEngine):
    def __init__(self, **kwargs) -> None:
        self.chain: RunnableSerializable = None
        super().__init__(**kwargs)

    @classmethod
    def from_args(cls, chain: RunnableSerializable, *args, **kwargs):
        instance = super().from_args(*args, **kwargs)
        instance.chain = chain
        return instance

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        response = super()._query(query_bundle)
        verified_response = self.chain.invoke(
            {
                "query": query_bundle.query_str,
                "response": response.response,
                "context": "---".join(
                    [
                        f"Context {i}:\n{node.node.text}"
                        for i, node in enumerate(response.source_nodes)
                    ]
                ),
            }
        )

        extra_info = {
            "references": verified_response.references,
            "is_faithful": verified_response.is_faithful,
            "reasoning": verified_response.reasoning,
            "original_response": response.response,
        }
        response.metadata = (
            extra_info if response.metadata is None else response.metadata | extra_info
        )
        response.response = verified_response.assistant_response

        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        response = await super()._aquery(query_bundle)

        verified_response = await self.chain.ainvoke(
            {
                "query": query_bundle.query_str,
                "response": response.response,
                "context": "---".join(
                    [
                        f"Context {i}:\n{node.node.text}"
                        for i, node in enumerate(response.source_nodes)
                    ]
                ),
            }
        )
        extra_info = {
            "references": verified_response.references,
            "is_faithful": verified_response.is_faithful,
            "reasoning": verified_response.reasoning,
            "original_response": response.response,
        }
        response.metadata = (
            extra_info if response.metadata is None else response.metadata | extra_info
        )
        response.response = verified_response.assistant_response

        return response


def build_query_engine(
    retriever: BaseRetriever,
    llm: LLM,
    postprocessors: BaseNodePostprocessor,
    langchain_light: BaseChatModel,
    langchain_heavy: BaseChatModel,
) -> FilteredQueryEngine:

    response_chain = build_response_chain(langchain_light, langchain_heavy)

    query_engine = FilteredQueryEngine.from_args(
        chain=response_chain,
        retriever=retriever,
        llm=llm,
        node_postprocessors=postprocessors,
    )

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": PromptTemplate(TEMPLATE)}
    )

    return query_engine
