from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.retrievers.multi_vector import SearchType
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.stores import BaseStore, ByteStore
from typing import Any, Optional, List
from langchain.chains.query_constructor.base import _format_attribute_info
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.vectorstores import VectorStore
from langchain_core.stores import BaseStore
from langchain_core.language_models.chat_models import BaseChatModel


class SelfQueryMultiVectorRetriever(SelfQueryRetriever):

    byte_store: Optional[ByteStore] = None
    """The lower-level backing storage layer for the parent documents"""
    docstore: Optional[BaseStore[str, Document]]
    """The storage interface for the parent documents"""
    id_key: Optional[str] = "doc_id"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        structured_query = self.query_constructor.invoke(
            {"query": query}, config={"callbacks": run_manager.get_child()}
        )

        if self.verbose:
            print(f"Generated Query: {structured_query}")
        new_query, search_kwargs = self._prepare_query(query, structured_query)

        # docs = self.vectorstore.search(query, self.search_type, **search_kwargs)
        # docs = self._get_docs_with_query(new_query, search_kwargs)

        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                new_query, **search_kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search(new_query, **search_kwargs)

        # print(new_query)
        # print(search_kwargs)
        # print(sub_docs)

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        return [d for d in docs if d is not None]

    @classmethod
    def from_llm(
        cls,
        byte_store: ByteStore,
        id_key: str,
        **kwargs: Any,
    ) -> "SelfQueryMultiVectorRetriever":
        instance = super().from_llm(**kwargs)
        instance.docstore = byte_store
        instance.id_key = id_key
        return instance


def build_retriever(
    llm: BaseChatModel, vectorstore: VectorStore, docstore: BaseStore, id_key: str
) -> SelfQueryMultiVectorRetriever:
    
    metadata_field_info = [
        AttributeInfo(
            name="category",
            description="The section to which a piece of documentation belongs. One of ['edgedb_general', 'ddl', 'integrations']. Most likely you will need to pick the first class to avoid duplication",
            type="string",
        ),
    ]
    document_content_description = (
        "Piece of EdgeDB documentation describing some functionality of the database."
    )

    retriever = SelfQueryMultiVectorRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        byte_store=docstore,
        id_key=id_key,
    )

    return retriever
