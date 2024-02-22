from llama_index.core.indices.vector_store.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

# from llama_index.core.indices.vector_store.retrievers.auto_retriever.prompts import (
#     PREFIX,
#     EXAMPLES,
#     SUFFIX,
# )
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataInfo,
    VectorStoreInfo,
    VectorStoreQuerySpec,
)

CUSTOM_META_DESCRIPTION = f"Section in which the documentation page is found. Can be one of 'edgeql_and_sdl', 'edgedb_general', 'ddl', 'integrations' or 'other'."

VECTOR_STORE_INFO = VectorStoreInfo(
    content_info="Documentation for EdgeDB database.",
    metadata_info=[
        MetadataInfo(
            name="section",
            type="str",
            description=CUSTOM_META_DESCRIPTION,
        ),
    ],
)


CUSTOM_PREFIX = """\
Your goal is to structure the user's query denoted by *** to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:

{schema_str}

edgeql_and_sdl: queries about defining a schema and querying data in EdgeDB or generally about EdgeQL and SDL languages.
edgedb_general: queries about EdgeDB setup, general usage and administration.
ddl: queries about DDL, EdgeDB's low-level data definition language.
integrations: queries about EdgeDB integrations with various programming languages and frameworks.
other: queries about EdgeDB that do not belong to any of the above categories.

The query string should contain only text that is expected to match the contents of \
documents. Any conditions in the filter should not be mentioned in the query as well.

Make sure that filter only refers to attribute that exist in the data source.
Make sure that filter takes into account the description of attribute.
Make sure that filter is only used as needed. \
By default, assume the query belongs to \
edgeql_and_sdl if it mentions writing schema or query, or \
edgedb_general otherwise. \

If the user's query explicitly mentions number of documents to retrieve, set top_k to \
that number, otherwise do not set top_k.

"""

example_query = "Can I cast an array to a different type?"

example_output = VectorStoreQuerySpec(
    query="cast an array to a different type",
    filters=[
        MetadataFilter(key="section", value="edgeql_and_sdl"),
    ],
)

example_query_2 = "How do I create a database using CLI?"

example_output_2 = VectorStoreQuerySpec(
    query="create a database using CLI",
    filters=[
        MetadataFilter(key="section", value="edgedb_general"),
    ],
)

CUSTOM_EXAMPLES = f"""\
<< Example 1. >>
Data Source:
```json
{VECTOR_STORE_INFO.json(indent=4)}
```

User Query:
{example_query}

Structured Request:
```json
{example_output.json()}


<< Example 2. >>
Data Source:
```json
{VECTOR_STORE_INFO.json(indent=4)}
```

User Query:
{example_query_2}

Structured Request:
```json
{example_output_2.json()}

```
""".replace(
    "{", "{{"
).replace(
    "}", "}}"
)


CUSTOM_SUFFIX = """

Data Source:
```json
{info_str}
```

User Query:
***{query_str}***

Structured Request:
"""


def build_retriever(
    index,
    full_nodes_dict,
    llm,
    verbose: bool = True,
    top_k: int = 10,
):

    retriever = VectorIndexAutoRetriever(
        index,
        vector_store_info=VECTOR_STORE_INFO,
        similarity_top_k=top_k,
        verbose=verbose,
        llm=llm,
    )

    retriever.update_prompts(
        {
            "prompt": PromptTemplate(
                template=CUSTOM_PREFIX + CUSTOM_EXAMPLES + CUSTOM_SUFFIX,
                prompt_type=PromptType.VECTOR_STORE_QUERY,
            )
        }
    )

    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": retriever},
        node_dict=full_nodes_dict,
        verbose=False,
    )

    return recursive_retriever
