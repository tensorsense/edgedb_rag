from pathlib import Path
from core.index import load_from_disk, save_to_disk


def build_index(persist_path: Path, collection_name, lib_path):

    if persist_path.exists():
        index, full_nodes_dict = load_from_disk(persist_path, collection_name, lib_path)
    else:
        index, full_nodes_dict = save_to_disk(persist_path, collection_name, lib_path)

    return index, full_nodes_dict
