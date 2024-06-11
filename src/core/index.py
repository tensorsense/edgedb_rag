from pathlib import Path
import re
import json
from tqdm import tqdm
from typing import Tuple, Dict

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Document,
)
from llama_index.core.schema import IndexNode, TextNode, BaseNode
from llama_index.core.indices.base import BaseIndex
from chromadb.api.models.Collection import Collection
from pydantic import BaseModel, Field
from enum import Enum


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


def extract_non_code_text(markdown_string: str) -> str:
    # Define a regex pattern to match triple backticks code blocks
    code_block_pattern = r"```.*?```"

    # Remove code blocks from the Markdown string
    non_code_text = re.sub(code_block_pattern, "", markdown_string, flags=re.DOTALL)

    return non_code_text


def save_to_disk(
    lib_path: Path,
    metadata_path: Path,
    persist_path: Path,
    collection_name: str,
) -> Tuple[BaseIndex, Dict[str, BaseNode]]:
    documents = SimpleDirectoryReader(lib_path.as_posix(), recursive=True).load_data()

    # Load metadata from disk
    with metadata_path.open("r") as f:
        metadatas = [
            DocMetadata.model_validate_json(raw_line) for raw_line in f.readlines()
        ]
        metadatas_dict = {doc_metadata.url: doc_metadata for doc_metadata in metadatas}

    # Set category for each parsed LlamaIndex document
    docs_with_meta = []

    for doc in documents:
        rel_path = (
            Path(doc.metadata["file_path"]).resolve().relative_to(lib_path).as_posix()
        )

        doc.metadata["section"] = metadatas_dict[rel_path].category.value
        docs_with_meta.append(doc)

    # Embed only text, store full docs
    stored_nodes = []
    full_nodes_dict = {}

    for doc in docs_with_meta:
        non_code_text = extract_non_code_text(doc.text)
        if len(non_code_text) < len(doc.text):
            doc_info = {k: v for k, v in doc.dict().items() if k not in ["id_", "text"]}
            index_node = IndexNode.from_text_node(
                index_id=doc.node_id,
                node=TextNode(
                    text=non_code_text,
                    **doc_info,
                ),
            )

            stored_nodes.append(index_node)
            full_nodes_dict[doc.node_id] = doc
        else:
            stored_nodes.append(doc)

    # save to disk

    chroma_client = chromadb.PersistentClient(path=persist_path.as_posix())
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        stored_nodes, storage_context=storage_context, show_progress=True
    )

    full_nodes_path = persist_path.joinpath("full_nodes.json")

    serializable = {k: v.json() for k, v in full_nodes_dict.items()}
    with full_nodes_path.open("w") as f:
        json.dump(serializable, f)

    return index, full_nodes_dict


def load_from_disk(
    persist_path: Path,
    collection_name: str,
) -> Tuple[BaseIndex, Dict[str, BaseNode]]:
    # load from disk
    chroma_client = chromadb.PersistentClient(path=persist_path.as_posix())
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
    )

    full_nodes_path = persist_path.joinpath("full_nodes.json")
    with full_nodes_path.open("r") as f:
        serializable = json.load(f)

    full_nodes_dict = {k: Document.parse_raw(v) for k, v in serializable.items()}

    return index, full_nodes_dict


def add_document(
    index: BaseIndex,
    full_nodes_dict: Dict[str, BaseNode],
    persist_path: Path,
    doc_text: str,
    doc_section: str,
) -> None:

    new_doc_no_code = extract_non_code_text(doc_text)
    doc = TextNode(text=doc_text)
    doc.metadata["section"] = doc_section

    doc_info = {k: v for k, v in doc.dict().items() if k not in ["id_", "text"]}

    index_node = IndexNode.from_text_node(
        index_id=doc.node_id,
        node=TextNode(
            text=new_doc_no_code,
            **doc_info,
        ),
    )

    index.insert_nodes([index_node])
    full_nodes_dict[doc.node_id] = doc

    full_nodes_path = persist_path.joinpath("full_nodes.json")

    serializable = {k: v.json() for k, v in full_nodes_dict.items()}
    with full_nodes_path.open("w") as f:
        json.dump(serializable, f)
