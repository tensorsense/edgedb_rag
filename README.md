# EdgeDB Retrieval Augmented Generator

## Overview

### 1. Indexing

**For every document** in the library:

1. Assign section based on its contents such as "EdgeQL and SDL", "DDL", "Integrations" etc.
2. Remove code snippets and calculate embeddings of what remains.
3. Store embeddings in vector store. Put section and link to the full (text_code) document into metadata.

**For a query**:

1. Using an LLM infer the section in which the answer to the query is most likely to be.
2. Filter stored embeddings that belong to that section.
3. Embed the query and perform vector search on filtered embeddings.
4. For every embedding that gets matched, get the full document that it links to.

![](assets/edgedb_index.png)

### 2. Generation

1. Run documents retrieved on the previous step through the LLM. Ask it to determine whether each of them is relevant to the query.
2. Remove documents that were classified as irrelevant.
3. With remaining documents as context, use LLM to answer the query based only on information that is present in them.
4. Finally, ask an LLM to validate whether the answer generated on the previous step is faithful to the documents that were given as context.
5. If the answer was classified as faithful, return it. Otherwise replace it with a fail message.

![](assets/edgedb_gen.png)

## Setup

1. Install requirements

```bash
pip3 install requirements.txt
```

2. Configure `OPENAI_API_KEY` in the environment.

## Configure LLMs

```python
from src.generator_builder import LLMConfig

llms = LLMConfig.from_azure_deployments(
    light="gpt-35-turbo-1106",
    heavy="gpt-4-1106",
    embedding="text-embedding-ada-002",
    api_version="2023-07-01-preview",
)
```

We're using three different text models within the pipeline.

1. `light` (e.g. `gpt-35-turbo` or a fine-tune) is used for query rewriting, context filtering and answer validation.
2. `heavy` (e.g. `gpt-4`) is used for final answer generation. It is also used as a fallback in case `light` fails to adhere to the output structure.
3. `embedding` is a Sentence Transformer model used to calculate embeddings for documents and queries.

Note that we're using LlamaIndex and LangChain within the pipeline, and each of them uses their own abstractions for the models.

Finally, set `embed_model` and `llm` to be used globally across LlamaIndex by default.
In particular, they are used to calculate query embeddings and refine context during generation.

```python
from llama_index.core import Settings

Settings.embed_model = llms.embed_model
Settings.llm = llms.llamaindex_light
```

## Build the index

```python
from src.index_builder import build_index

index, full_nodes_dict = build_index(
    lib_path = Path("../../docs_md").resolve()
    persist_path = Path("index_storage").resolve(),
    collection_name = "dev1",
)
```

- `lib_path` is a directory that contains Markdown chunks of documentation.
- `persist_path` is a directory where both ChromaDB and LlamaIndex are storing documents and their embeddings.
- `collection_name` is for ChromaDB to identify embeddings that belong to this particular index.

If `persist_path` exists, the script will attempt to load the index from there. Otherwise, it will build one from scratch.

- ChromaDB only stores document embeddings and metadata, but not the original text.
- Before calculating embeddings, we remove code snippets from text. We keep original (text + code) documents in the `full_nodes_dict`.
- `index` is a LlamaIndex object that talks to ChromaDB to perform embedding seach and metadata filtering, and then fetches the documents based on the output.

## Build the generator

```python
from src.generator_builder import build_generator

generator = build_generator(
    index=index, full_nodes_dict=full_nodes_dict, llm_config=llms
)
```

## Run a query

## Run evaluation



