from pathlib import Path
from typing import Tuple, Dict
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import BaseNode
from src.core.index import load_from_disk, save_to_disk


def build_index(
    lib_path: Path,
    persist_path: Path,
    collection_name: str,
) -> Tuple[BaseIndex, Dict[str, BaseNode]]:

    if persist_path.exists():
        index, full_nodes_dict = load_from_disk(
            lib_path=lib_path,
            persist_path=persist_path,
            collection_name=collection_name,
        )
    else:
        index, full_nodes_dict = save_to_disk(
            lib_path=lib_path,
            persist_path=persist_path,
            collection_name=collection_name,
        )
    return index, full_nodes_dict
