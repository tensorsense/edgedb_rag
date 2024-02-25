from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from llama_index.core.bridge.pydantic import BaseModel, Field
from typing import List, Dict, Any
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
)

from llama_index.core import PromptTemplate
from llama_index.core.base.response.schema import Response
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.chat_engine.types import BaseChatEngine, AgentChatResponse


ANSWER_RELEVANCY_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the response is relevant to the query.\n"
    "The evaluation should be performed in a step-by-step manner by answering the following questions:\n"
    "1. Is the provided response specific to EdgeDB?\n"
    "2. Does the provided response stick to EdgeQL and SDL by default OR match the programming language specified in the query?\n"
    "3. Does the provided response match the subject matter of the user's query?\n"
    "4. Does the provided response attempt to address the focus or perspective on the subject matter taken on by the user's query?\n"
    "Each question above is worth 1 point. Provide detailed feedback on response according to the criteria questions above. "
    "After your feedback provide a final result by strictly following this format: '[RESULT] followed by the integer number representing the total score assigned to the response'\n\n"
    "Query: \n {query}\n"
    "Response: \n {response}\n"
    "Feedback:"
)

CONTEXT_RELEVANCY_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the retrieved context from the document sources are relevant to the query.\n"
    "The evaluation should be performed in a step-by-step manner by answering the following questions:\n"
    "1. Does the retrieved context default to EdgeQL and SDL OR match the programming language specified in the query?\n"
    "2. Does the retrieved context match the subject matter of the user's query?\n"
    "3. Can the retrieved context be used exclusively to provide a full answer to the user's query?\n"
    "Each question above is worth 2 points, where partial marks are allowed and encouraged. Provide detailed feedback on the response "
    "according to the criteria questions previously mentioned. "
    "After your feedback provide a final result by strictly following this format: "
    "'[RESULT] followed by the float number representing the total score assigned to the response'\n\n"
    "Query: \n {query_str}\n"
    "Context: \n {context_str}\n"
    "Feedback:"
)


class PydanticResponse(BaseModel):
    query: str
    response: str
    metadata: Dict[str, Any]
    source_nodes: List[TextNode]

    @classmethod
    def from_llamaindex_response(cls, query, llamaindex_response):
        if isinstance(llamaindex_response, AgentChatResponse):
            metadata = llamaindex_response.sources[0].raw_output.metadata
        elif isinstance(llamaindex_response, Response):
            metadata = llamaindex_response.metadata
        else:
            raise NotImplementedError
        return cls(
            query=query,
            response=llamaindex_response.response,
            metadata=metadata,
            source_nodes=[node.node for node in llamaindex_response.source_nodes],
        )


def run_queries(generator, queries: List[str], result_path: Path):
    all_responses = []

    for query in tqdm(queries):
        try:
            if issubclass(type(generator), BaseChatEngine):
                llamaindex_response = generator.chat(message=query, chat_history=[])
            elif issubclass(type(generator), BaseQueryEngine):
                llamaindex_response = generator.query(query)
            else:
                raise NotImplementedError
        except Exception as e:
            print(e)
            continue

        response = PydanticResponse.from_llamaindex_response(
                query=query,
                llamaindex_response=llamaindex_response,
            )
        with result_path.open("+a") as f:
            f.write(f"{response.json()}\n")

        all_responses.append(response)

    return all_responses


class Evaluator:
    def __init__(self, llm) -> None:
        self.answer_relevancy = AnswerRelevancyEvaluator(
            llm=llm,
            eval_template=ANSWER_RELEVANCY_TEMPLATE,
            score_threshold=4.0,
        )

        self.context_relevancy = ContextRelevancyEvaluator(
            llm=llm,
            eval_template=CONTEXT_RELEVANCY_TEMPLATE,
            score_threshold=6.0,
        )

        self.faithfulness = FaithfulnessEvaluator(llm=llm)

    def evaluate_response(self, response: PydanticResponse):
        llamaindex_response = response.to_llamaindex_response()
        query = response.query

        faithfulness = self.faithfulness.evaluate_response(response=llamaindex_response)
        answer_relevancy = self.answer_relevancy.evaluate_response(
            query=query, response=llamaindex_response
        )
        context_relevancy = self.context_relevancy.evaluate_response(
            query=query, response=llamaindex_response
        )

        return {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_relevancy": context_relevancy,
        }
