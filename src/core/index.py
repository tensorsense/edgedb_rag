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

# Section mapping for each document in the index

DOC_MAP = {
    "changelog": "other",
    "cli": "edgedb_general",
    "clients": "integrations",
    "datamodel": "edgeql_and_sdl",
    "edgeql": "edgeql_and_sdl",
    "glossary": "other",
    "guides/auth": "edgedb_general",
    "guides/cheatsheet/admin": "edgedb_general",
    "guides/cheatsheet/aliases": "edgeql_and_sdl",
    "guides/cheatsheet/annotations": "edgeql_and_sdl",
    "guides/cheatsheet/boolean": "edgeql_and_sdl",
    "guides/cheatsheet/cli": "edgedb_general",
    "guides/cheatsheet/delete": "edgeql_and_sdl",
    "guides/cheatsheet/functions": "edgeql_and_sdl",
    "guides/cheatsheet/insert": "edgeql_and_sdl",
    "guides/cheatsheet/index": "other",
    "guides/cheatsheet/link_properties": "edgeql_and_sdl",
    "guides/cheatsheet/objects": "edgeql_and_sdl",
    "guides/cheatsheet/repl": "edgedb_general",
    "guides/cheatsheet/select": "edgeql_and_sdl",
    "guides/cheatsheet/update": "edgeql_and_sdl",
    "guides/cloud": "edgedb_general",
    "guides/contributing": "edgedb_general",
    "guides/datamigrations": "edgedb_general",
    "guides/deployment": "edgedb_general",
    "guides/index": "other",
    "guides/migrations": "ddl",
    "guides/tutorials": "integrations",
    "index": "other",
    "intro/cli": "edgedb_general",
    "intro/clients": "integrations",
    "intro/edgeql": "edgeql_and_sdl",
    "intro/index": "other",
    "intro/instances": "edgedb_general",
    "intro/migrations": "edgedb_general",
    "intro/projects": "edgedb_general",
    "intro/quickstart": "edgedb_general",
    "intro/schema": "edgeql_and_sdl",
    "reference/admin": "edgedb_general",
    "reference/bindings": "edgeql_and_sdl",
    "reference/ddl": "ddl",
    "reference/edgeql": "edgeql_and_sdl",
    "reference/index": "other",
    "reference/protocol": "edgedb_general",
    "reference/sdl": "edgeql_and_sdl",
    "reference/connection": "edgedb_general",
    "reference/environment": "edgedb_general",
    "reference/projects": "edgedb_general",
    "reference/edgedb_toml": "edgedb_general",
    "reference/dsn": "edgedb_general",
    "reference/dump_format": "edgedb_general",
    "reference/backend_ha": "edgedb_general",
    "reference/configuration": "edgedb_general",
    "reference/http": "edgedb_general",
    "reference/sql_support": "edgedb_general",
    "stdlib": "edgeql_and_sdl",
}


def extract_non_code_text(markdown_string: str) -> str:
    # Define a regex pattern to match triple backticks code blocks
    code_block_pattern = r"```.*?```"

    # Remove code blocks from the Markdown string
    non_code_text = re.sub(code_block_pattern, "", markdown_string, flags=re.DOTALL)

    return non_code_text


def save_to_disk(
    lib_path: Path,
    persist_path: Path,
    collection_name: str,
) -> Tuple[BaseIndex, Dict[str, BaseNode]]:
    documents = SimpleDirectoryReader(lib_path.as_posix(), recursive=True).load_data()

    # Set category for each document

    sections = set()
    docs_with_meta = []

    for doc in documents:
        section = str(
            Path(doc.metadata["file_path"]).parent.resolve().relative_to(lib_path)
        )
        doc.metadata["section"] = section
        sections.add(section)
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

    chroma_collection = update_meta(chroma_collection, lib_path)

    return index, full_nodes_dict


def load_from_disk(
    lib_path: Path,
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

    chroma_collection = update_meta(chroma_collection, lib_path)

    return index, full_nodes_dict


def update_meta(chroma_collection: Collection, root_path: Path) -> Collection:
    # Add updated sections to metadata
    doc_map_keys = list(DOC_MAP.keys())

    metas = chroma_collection.peek(5000)["metadatas"]
    ids = chroma_collection.peek(5000)["ids"]

    for i, meta in enumerate(metas):
        rel_path = Path(meta["file_path"]).resolve().relative_to(root_path).as_posix()
        section_key = [key for key in doc_map_keys if rel_path.startswith(key)][
            -1
        ]  # because cli and clients
        meta["section"] = DOC_MAP[section_key]
        chroma_collection.update(ids=ids[i], metadatas=meta)

    return chroma_collection
