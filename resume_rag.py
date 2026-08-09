"""
RAG tool built directly on the chromadb client (no LangChain).

Mirrors the same pipeline as Tony's original rag_chatbot.py:
  load PDF -> chunk text -> embed -> store in Chroma -> retrieve on query

Why no LangChain here: this agent project's environment had install issues
with LangChain on Python 3.14, and going straight to the chromadb client
avoids that entirely while also being a stronger "I understand what's
underneath the framework" signal for interviews.

Embeddings use OpenAI (text-embedding-3-small) since it's the cheapest,
most standard option and doesn't require setting up a second embeddings
provider. The Anthropic API is still what powers the agent's reasoning and
tool-use loop in agent.py -- this file only produces a `resume_lookup`
tool function and its OpenAI-compatible function-schema dict.
"""

import os
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from openai import OpenAI

load_dotenv()

PDF_PATH = "Tony_Testa_Resume_mthree.pdf"   # put your resume PDF in this same folder
CHUNK_SIZE = 500       # characters per chunk, matches original script
CHUNK_OVERLAP = 50

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def load_and_chunk_pdf(path: str) -> list[str]:
    """Extract text from the PDF and split into overlapping chunks."""
    reader = PdfReader(path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    chunks = []
    start = 0
    while start < len(full_text):
        end = start + CHUNK_SIZE
        chunks.append(full_text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP  # overlap so context isn't cut mid-sentence
    return [c for c in chunks if c.strip()]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of text chunks via OpenAI's embeddings endpoint."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def build_resume_index(pdf_path: str = PDF_PATH):
    """
    Builds an in-memory Chroma collection from the resume PDF, once, at
    startup -- same lifecycle as the original Chroma.from_documents() call.
    Returns a queryable collection.
    """
    chunks = load_and_chunk_pdf(pdf_path)
    embeddings = embed_texts(chunks)

    chroma_client = chromadb.Client()  # in-memory, ephemeral -- matches original script
    collection = chroma_client.create_collection(name="resume")

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    print(f"  [resume_rag] indexed {len(chunks)} chunks from {pdf_path}")
    return collection


# Built once when this module is imported, reused across all tool calls.
_resume_collection = None


def get_resume_collection():
    global _resume_collection
    if _resume_collection is None:
        _resume_collection = build_resume_index()
    return _resume_collection


def resume_lookup(query: str, n_results: int = 3) -> str:
    """
    The actual tool function the agent calls. Embeds the query, retrieves
    the most relevant resume chunks, and returns them as context for the
    model to answer from -- the "R" and "A" of RAG. The "G" (generation)
    happens back in agent.py when the model reads this returned text.
    """
    collection = get_resume_collection()
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    matched_chunks = results["documents"][0]
    if not matched_chunks:
        return "No relevant information found in the resume."

    return "\n---\n".join(matched_chunks)


# Tool schema for agent.py to register alongside calculator and lookup
RESUME_TOOL_SCHEMA = {
    "name": "resume_lookup",
    "description": "Search Tony's resume for relevant information to answer "
    "a question about his work history, education, skills, or background. "
    "Use this for any question about Tony's resume, experience, or qualifications.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in the resume, e.g. "
                "'work experience' or 'education and certifications'",
            }
        },
        "required": ["query"],
    },
}


if __name__ == "__main__":
    # Quick standalone test, mirrors how rag_chatbot.py let you ask questions directly
    print("Building resume index...")
    get_resume_collection()
    print("\nTry it:")
    for q in ["What companies has Tony worked for?", "What is Tony's GPA?"]:
        print(f"\nQ: {q}")
        print(resume_lookup(q))
