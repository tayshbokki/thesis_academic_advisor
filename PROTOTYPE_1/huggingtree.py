# Load environment variables (including HF_TOKEN)
from dotenv import load_dotenv
load_dotenv()

import os
model=InferenceClientModel(token=os.environ["HF_TOKEN"])
import sys

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import datasets
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# Load the Hugging Face documentation dataset
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# Filter to include only Transformers documentation
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

# Convert dataset entries to Document objects with metadata
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

# Split documents into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Characters per chunk
    chunk_overlap=50,  # Overlap between chunks to maintain context
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],  # Priority order for splitting
)
docs_processed = text_splitter.split_documents(source_docs)

print(f"Knowledge base prepared with {len(docs_processed)} document chunks")

from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        # Initialize the retriever with our processed documents
        self.retriever = BM25Retriever.from_documents(
            docs, k=10  # Return top 10 most relevant documents
        )

    def forward(self, query: str) -> str:
        """Execute the retrieval based on the provided query."""
        assert isinstance(query, str), "Your search query must be a string"

        # Retrieve relevant documents
        docs = self.retriever.invoke(query)

        # Format the retrieved documents for readability
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# Initialize our retriever tool with the processed documents
retriever_tool = RetrieverTool(docs_processed)

from smolagents import InferenceClientModel, CodeAgent

# Initialize the agent with our retriever tool
agent = CodeAgent(
    tools=[retriever_tool],  # List of tools available to the agent
    model=InferenceClientModel(),  # Default model "Qwen/Qwen3-Next-80B-A3B-Thinking"
    max_steps=4,  # Limit the number of reasoning steps
    verbosity_level=2,  # Show detailed agent reasoning
)

# To use a specific model, you can specify it like this:
# model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct")

# Ask a question that requires retrieving information
question = "For a transformers model training, which is slower, the forward or the backward pass?"

# Run the agent to get an answer
agent_output = agent.run(question)

# Display the final answer
print("\nFinal answer:")
print(agent_output)