from pathlib import Path
from tqdm import tqdm
from enum import Enum
import uuid
import re

from typing import List, Dict
from pydantic.v1 import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.stores import BaseStore
from langchain_core.embeddings import Embeddings
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader


ID_KEY = "doc_id"


class DocCategory(Enum):
    EDGEDB_GENERAL = "edgedb_general"
    EDGEQL_AND_SDL = "edgeql_and_sdl"
    DDL = "ddl"
    INTEGRATIONS = "integrations"
    OTHER = "other"


class DocMetadata(BaseModel):
    url: str = Field(description="Path to doc relative to root.")
    category: DocCategory = Field(
        default=DocCategory.EDGEDB_GENERAL,
        description="Parent category to which the doc belongs.",
    )


class Docs(BaseModel):
    full: Dict[str, Document] = Field(description="Mapping of UUIDs to full documents")
    children: List[Document] = Field(
        description="List of child documents carrying parent UUID in metadata"
    )


class PersistentDocStore(BaseModel):
    doc_map: Dict[str, Document] = Field(default_factory=dict)


class Index(BaseModel):
    """Index for searching documents"""

    class Config:
        arbitrary_types_allowed = True

    vectorstore: VectorStore = Field(
        description="Stores no-code chunks of text and their embeddings. Performs similarity search."
    )
    docstore: BaseStore = Field(description="Stores full parent documents.")
    embedding_function: Embeddings = Field(
        description="Embedding function used to project documents and queries."
    )

    @staticmethod
    def read_docs_from_metadata(metadata_path: Path, lib_path: Path) -> Docs:
        # 1. Load metadata from disk
        with metadata_path.open("r") as f:
            metadatas = [DocMetadata.parse_raw(raw_line) for raw_line in f.readlines()]

        def extract_non_code_text(markdown_string: str) -> str:
            # Define a regex pattern to match triple backticks code blocks
            code_block_pattern = r"```.*?```"

            # Remove code blocks from the Markdown string
            non_code_text = re.sub(
                code_block_pattern, "", markdown_string, flags=re.DOTALL
            )

            return non_code_text

        # 2. Read original documents from disk according to paths found in metadata
        full_docs: List[Document] = []

        for metadata in metadatas:
            loader = TextLoader(lib_path / metadata.url)
            doc_chunks = loader.load()
            for doc in doc_chunks:
                doc.metadata["category"] = metadata.category.value
            full_docs.extend(doc_chunks)

        # 3. Generate doc ids used to link child documents with their parents
        full_docs_dict = {str(uuid.uuid4()): doc for doc in full_docs}

        # 4. Create child docs by removing code from full documents
        no_code_docs = [
            Document(
                page_content=extract_non_code_text(full_doc.page_content),
                metadata=full_doc.metadata | {ID_KEY: doc_id},
            )
            for doc_id, full_doc in full_docs_dict.items()
        ]

        return Docs(full=full_docs_dict, children=no_code_docs)

    @staticmethod
    def create_vectorstore(
        persist_directory: str, embedding_function: Embeddings
    ) -> VectorStore:
        # The vectorstore for the child chunks
        vectorstore = Chroma(
            collection_name="child_docs",
            persist_directory=persist_directory,
            embedding_function=embedding_function,
        )
        return vectorstore

    @staticmethod
    def create_docstore(persistent_docstore: PersistentDocStore) -> BaseStore:
        # The storage layer for the parent documents
        docstore = InMemoryByteStore()
        docstore.mset([(k, v) for k, v in persistent_docstore.doc_map.items()])
        return docstore

    @classmethod
    def from_metadata(
        cls,
        metadata_path: Path,
        lib_path: Path,
        persist_path: Path,
        embedding_function: Embeddings,
    ) -> "Index":
        """Build new index using a library of docs and a metadata JSON file"""

        # 1. Read and preprocess documents
        docs = cls.read_docs_from_metadata(metadata_path, lib_path)

        # 2. Create and populate vectorstore
        vectorstore = cls.create_vectorstore(
            persist_directory=persist_path.resolve().as_posix(),
            embedding_function=embedding_function,
        )

        def batcher(seq, size):
            return (seq[pos : pos + size] for pos in range(0, len(seq), size))

        batches = list(batcher(docs.children, 100))
        for batch in tqdm(batches, total=len(batches)):
            vectorstore.add_documents(batch)

        # 3. Create and populate docstore for full documents, and save it to disk
        persistent_docstore = PersistentDocStore(doc_map=docs.full)
        docstore = cls.create_docstore(persistent_docstore=persistent_docstore)

        with persist_path.joinpath("docstore.json").open("w") as f:
            f.write(persistent_docstore.json())

        return cls(
            vectorstore=vectorstore,
            docstore=docstore,
            embedding_function=embedding_function,
        )

    @classmethod
    def from_persist_path(cls, persist_path: Path, embedding_function: Embeddings):
        """Load existing index from disk"""
        vectorstore = cls.create_vectorstore(
            persist_directory=persist_path.resolve().as_posix(),
            embedding_function=embedding_function,
        )
        with persist_path.joinpath("docstore.json").open("r") as f:
            persistent_docstore = PersistentDocStore.parse_raw(f.read())
        docstore = cls.create_docstore(persistent_docstore=persistent_docstore)
        return cls(
            vectorstore=vectorstore,
            docstore=docstore,
            embedding_function=embedding_function,
        )
