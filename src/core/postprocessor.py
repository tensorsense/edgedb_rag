from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_core.runnables import RunnableSerializable

from langchain_core.pydantic_v1 import BaseModel, Field

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from typing import Any, List, Optional

# from pydantic import BaseModel, Field


class RatedDocument(BaseModel):
    content: str = Field(description="Text of the document verbatim.")
    is_relevant: bool = Field(
        description="Whether the document contains information necessary to reply to the query."
    )


class FilteredContext(BaseModel):
    documents: List[RatedDocument] = Field(
        description="All provided documents, rated for relevance."
    )


def build_context_filter_chain(langchain_light, langchain_heavy):

    system_template = """You are an assistant that helps filter pieces of EdgeDB documentation.
        Below you will find several pieces of official EdgeDB documentation, each denoted by ---, as well as a user query denoted by ***.
        Your job is to determine for each piece whether or not it contains information necessary to answer user's query.
        Pay specific attention to:

        1. Programming language. Assume by default that the user is expecting to see EdgeQL and SDL in the answer, unless they explicitly said otherwise.
        2. Relevancy of the concepts. For example:

        Query: how do I create a Movie entity with a backlink to characters?

        Relevant (contains an example of a backlink):
        ```sdl
        type User {{
            required email: str;
            multi friends: User;
            blog_links := .<author[is BlogPost];
            comment_links := .<author[is Comment];
        }}
        ```

        Not relevant (does not contain an example of a backlink):
        ```sdl
        type Movie {{
            required title: str {{ constraint exclusive }};
            required release_year: int64;
            multi characters: Person;
        }}
        ```

        
        {format_instructions}
        """

    human_template = """***{query}***
        ---{documents}---
        """

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=FilteredContext)

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


class ContextFilter(BaseNodePostprocessor):
    chain: RunnableSerializable = Field(description="LangChain chain to verify context")

    def __init__(self, chain):
        super().__init__(
            chain=chain,
        )

    @classmethod
    def class_name(cls) -> str:
        return "ContextFilter"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        concat_documents = "---".join([node.node.text for node in nodes])

        filtered_context = self.chain.invoke(
            {
                "query": query_bundle.query_str,
                "documents": concat_documents,
            }
        )

        filtered_nodes = []

        for rated_doc in filtered_context.documents:
            if rated_doc.is_relevant:
                filtered_nodes.append(
                    NodeWithScore(node=TextNode(text=rated_doc.content), score=1.0)
                )

        return filtered_nodes


def build_context_filter(langchain_light, langchain_heavy):
    context_filter_chain = build_context_filter_chain(langchain_light, langchain_heavy)
    context_filter = ContextFilter(chain=context_filter_chain)

    return context_filter
