# EdgeDB Retrieval Augmented Generator 🦜

## Getting started

1. Install dependencies: `pip3 install -r requirements.txt`
2. Set your `OPENAI_API_KEY` in the environment, if necessary.
3. Prepare the docs in Markdown format. Split longer documents into sections, so that each documents covers one specific concept.
4. Prepare the metadata file. Use the following JSON lines format: 

	`{"url":"relative/path/to/doc.md","category":"edgedb_general"}`
	
	The current list of categories goes like this:
	
	- `edgeql_and_sdl`
	- `ddl`
	- `integrations`
	- `edgedb_general`
	- `other`

4. See `build_rag.ipynb` for a step-by-step configuration guide.
5. See `demo.py` for an example chatbot implementation with answer streaming.

## System overview

### 1. Index

The index consists of two components:

- vectorstore to store document embeddings and perform vector search.
- docstore to store full documents.

To build the index, we need to have the documents stored as Markdown files, as well as the metadata JSON file. **For every document in metadata file**:

1. Load the document.
2. Calculate an embedding and store in the vectorstore.
3. Store the original full document in the docstore (represented by a JSON file).

### 2. Conversational RAG

This is the backbone of the chatbot application. It is represented by a LangChain chain that takes each user message through multiple processing stages before generating a response.

The stages include:

1. **Contextualization**: Turning a message into a search query using chat history.
2. **Query analysis**: Breaking down the search query into a similarity search part and a filter.
3. **Retrieval**: Using vectorstore to retrieve relevant documents with similarity search.
4. **Generation**: Producing the final answer based on documents and chat history.

Stages 1, 2 and 4 perform one LLM request each, with stage 4 doing the heavy lifting of synthesizing the answer.

## Integration steps

### Initialization

1. Build the index:

    ```python
    index = Index.from_metadata(
        metadata_path=Path("resources/doc_metadata.jsonl"),
        lib_path=Path("../docs_md"),
        persist_path=persist_path,
        embedding_function=embedding_function,
    )
    ```

2. Use index to build a retriever

    ```python
    retriever = build_retriever(
        llm=llm, vectorstore=index.vectorstore, docstore=index.docstore
    )
    ```

3. Set up a history callable. 
   The RAG itself does not store or handle chat history in any way. Instead it calls this callable with arguments specified in the config to get relevant chat history for every generation.

    ```python
    def parse_history(raw_history):
        # parses message history from the list of pairs of strings
        history = ChatMessageHistory()
        for human, ai in raw_history:
            history.add_user_message(human)
            history.add_ai_message(ai)

        return history

    # description parse_history arguments
    history_factory_config=[
        ConfigurableFieldSpec(
            id="raw_history",
            annotation=List,
            name="Raw chat message history",
            description="List of messages coming from frontend",
            default=[],
            is_shared=True,
        ),
    ]
    ```

    ```python
    # example call with this setup
    response = generator.stream(
        {"input": question},
        config={"configurable": {"raw_history": history}},  # this is where we attach raw history coming from the frontend
    )
    ```

4. Use the retriever and the history callable to build the generator

    ```python
    generator = build_generator(
        llm=llm,
        retriever=retriever,
        get_session_history_callable=parse_history,
        history_factory_config=history_factory_config,
    )
    ```

### Generating answers

1. Use LangChain's `stream` to get a streaming response.

    - Make sure to pass in the newest user message, as well as previous chat history.
    - Using chat history enables the chatbot to handle followup question and have multi-turn conversations.
    - The chatbot searches for documents relevant to the latest message every time before generating the answer.

    ```python
    response = generator.stream(
        {"input": question},
        config={"configurable": {"raw_history": history}},
    )
    ```

2. Iterate over the response:

    ```python
    for segment in response:
        # deal with the answer
    ```

### Response structure

The overall structure of the response is this:

```python
{
	'input': 'In TypeScript, how do I do an insert?',
	'chat_history': [],
	'retrieval_result': {
		'input': 'In TypeScript, how do I do an insert?',
		'search_terms': Search(
			query='insert in TypeScript',
			category='integrations'
		),
		'documents': [
			Document(
				page_content='Full text of the doc',
				metadata={
					'source':'path/to/doc.md',
					'category': 'integrations'
				}
			),
		]
	},
	'answer': QuotedAnswer(
		answer='To perform an insert in TypeScript using EdgeDB...',
		citations=[
			Citation(
				source_id=0,
				quote='Verbatim quote from the doc')
			]
	)
}
```

Note Pydantic types used by LangChain under the hood.

```python
from langchain_core.documents import Document
from src.core.retriever import Search
from src.core.generator import Citation, QuotedAnswer
```

However, when streaming you are not going to get this entire dictionary all at once. LangChain is going to stream the output of the query analysis first, then proceed to stream the answer.

For an example of handling LangChain streaming output see `demo.py`.