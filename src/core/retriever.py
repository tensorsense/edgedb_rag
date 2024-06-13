from typing import List, Optional
from operator import itemgetter
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)


class Search(BaseModel):
    """Search over documentation for EdgeDB database."""

    query: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    category: Optional[str] = Field(
        None,
        description="The section to which a piece of documentation belongs. One of ['edgedb_general', 'ddl', 'integrations'].",
    )


class FilteredMultiVectorRetriever(MultiVectorRetriever):

    _new_arg_supported: bool = True
    _expects_other_args: bool = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        _filter=None,
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """

        _filter = _filter if _filter is not None else {}
        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, filter=_filter, **self.search_kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search(
                query, filter=_filter, **self.search_kwargs
            )

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        return [d for d in docs if d is not None]


def build_retriever(llm, vectorstore, docstore):

    system = """You are an expert at converting user questions into database queries. \
    You have access to documentation for EdgeDB, the database. \
    Given a question, return a list of database queries optimized to retrieve the most relevant results.

    If there are acronyms or words you are not familiar with, do not try to rephrase them."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    structured_llm = llm.with_structured_output(Search)
    query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

    filtered_retriever = FilteredMultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
    )

    def retrieve_with_filter(retreiver_chain):
        def do_retrieve(search):
            return retreiver_chain.invoke(
                search.query, _filter={"category": {"$eq": search.category}}
            )

        return do_retrieve

    extract_search_terms = RunnableParallel(
        input=RunnablePassthrough(),
        search_terms=query_analyzer,
    )

    retriever_chain = extract_search_terms | RunnablePassthrough.assign(
        documents=itemgetter("search_terms")
        | RunnableLambda(retrieve_with_filter(filtered_retriever))
    )

    return retriever_chain
