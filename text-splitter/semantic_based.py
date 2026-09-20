from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings
)

from langchain_experimental.text_splitter import (
    SemanticChunker
)

from dotenv import load_dotenv
load_dotenv()

# Create Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# Create semantic splitter
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

# Sample text
text = """
Python is a programming language.
Python is used for machine learning and AI.

The Eiffel Tower is located in Paris.
Paris is the capital of France.
"""

# Split text
chunks = splitter.split_text(text)

# Print chunks
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk)