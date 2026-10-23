
import time
from typing import List, Dict, Any

import google.generativeai as genai
import chromadb
from chromadb.config import Settings

from ..config.settings import Config
from ..config.prompts import get_system_prompt


class GeminiResponseGenerator:
    """RAG chatbot for UCSB Engineering information."""

    def __init__(self, collection_name: str = None):
        self.config = Config()
        self.collection_name = collection_name or self.config.COLLECTION_NAME

        genai.configure(api_key=self.config.GOOGLE_API_KEY)
        self.embedding_model = self.config.EMBEDDING_MODEL
        self.chat_model = self.config.GEMINI_MODEL

        self.chroma_client = chromadb.PersistentClient(
            path=self.config.CHROMADB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.chroma_client.get_collection(
            name=self.collection_name
        )
        print(f"Connected to ChromaDB collection: {self.collection_name}")
        print(f"Collection contains {self.collection.count()} documents")

        self.model = genai.GenerativeModel(self.chat_model)
        self.system_prompt = get_system_prompt("ucsb_engineering")

    def get_query_embedding(
        self, query: str, retries: int = 1
    ) -> List[float]:
        """Create a query embedding using Gemini."""
        for attempt in range(retries):
            try:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=query,
                    task_type="retrieval_query",
                )
                return result["embedding"]
            except Exception as e:
                print(f"Query embedding failed: {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(1)

        raise RuntimeError("Could not create query embedding.")

    def retrieve_relevant_docs(
        self, query: str, n_results: int = None
    ) -> Dict[str, Any]:
        """Try vector retrieval first, then local keyword search."""
        if n_results is None:
            n_results = self.config.DEFAULT_RETRIEVAL_COUNT

        # Try semantic retrieval. This requires Google API quota.
        try:
            count = self.collection.count()
            if count:
                query_embedding = self.get_query_embedding(query)
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, count),
                    include=["documents", "metadatas", "distances"],
                )

                docs = (results.get("documents") or [[]])[0]
                metadatas = (results.get("metadatas") or [[]])[0]

                formatted = [
                    {
                        "content": doc,
                        "metadata": metadata or {},
                        "rank": i + 1,
                    }
                    for i, (doc, metadata) in enumerate(
                        zip(docs, metadatas)
                    )
                ]

                if formatted:
                    return {
                        "query": query,
                        "results": formatted,
                        "total_found": len(formatted),
                    }
        except Exception as e:
            print(f"Vector search unavailable; using local search: {e}")

        # Local search reads saved text and does not call Google.
        try:
            saved = self.collection.get(
                include=["documents", "metadatas"]
            )
            documents = saved.get("documents") or []
            metadatas = saved.get("metadatas") or []

            stop_words = {
                "what", "which", "where", "when", "does", "about",
                "tell", "show", "give", "with", "from", "that",
                "this", "the", "and", "for", "are", "how", "can",
                "you", "into", "have", "has", "its", "who", "why",
                "please", "information", "objective", "major",
            }

            terms = {
                word.strip(".,?!:;()[]{}\"'").lower()
                for word in query.split()
            }
            terms = {
                word for word in terms
                if len(word) > 2 and word not in stop_words
            }

            matches = []
            for doc, metadata in zip(documents, metadatas):
                doc = doc or ""
                metadata = metadata or {}
                meta_text = " ".join(
                    str(value) for value in metadata.values()
                ).lower()
                text = doc.lower()

                score = sum(
                    3 for term in terms if term in meta_text
                ) + sum(
                    1 for term in terms if term in text
                )

                if score:
                    matches.append((score, doc, metadata))

            matches.sort(key=lambda item: item[0], reverse=True)

            formatted = [
                {
                    "content": doc,
                    "metadata": metadata,
                    "rank": i + 1,
                }
                for i, (_, doc, metadata) in enumerate(
                    matches[:n_results]
                )
            ]

            return {
                "query": query,
                "results": formatted,
                "total_found": len(formatted),
            }

        except Exception as e:
            print(f"Local keyword search failed: {e}")
            return {"query": query, "results": [], "total_found": 0}

    def format_context(
        self, retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """Format retrieved documents as context."""
        if not retrieved_docs:
            return "No relevant documents found."

        context_parts = []
        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            content = doc.get("content", "")
            doc_type = metadata.get("type", "document")

            if doc_type == "course":
                header = (
                    f"COURSE: {metadata.get('course_name', 'Unknown')} - "
                    f"{metadata.get('course_title', '')}"
                )
            elif doc_type == "program":
                header = f"PROGRAM: {metadata.get('program', 'Unknown')}"
            elif doc_type == "department":
                header = (
                    f"DEPARTMENT: "
                    f"{metadata.get('department', 'Unknown')}"
                )
            else:
                header = f"DOCUMENT: {doc_type.title()}"

            context_parts.append(f"--- {header} ---\n{content}")

        return "\n\n".join(context_parts)

    def generate_response(
        self, query: str, max_retries: int = 1
    ) -> Dict[str, Any]:
        """Generate a Gemini answer or return matching saved text."""
        retrieval_results = {
            "query": query,
            "results": [],
            "total_found": 0,
        }

        try:
            retrieval_results = self.retrieve_relevant_docs(query)

            docs = retrieval_results["results"]
            if not docs:
                return {
                    "query": query,
                    "response": (
                        "I couldn't find matching information in the saved "
                        "UCSB Engineering documents. Try different keywords."
                    ),
                    "sources": [],
                    "success": True,
                }

            context = self.format_context(docs)
            prompt = f"""{self.system_prompt}

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

Answer using the context above. If the context does not contain the answer,
explain what information is available without inventing facts."""

            # Gemini generation may fail if its quota is exhausted.
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)

                    sources = []
                    for doc in docs:
                        metadata = doc.get("metadata", {})
                        sources.append({
                            "type": metadata.get("type", "document"),
                            "department": metadata.get(
                                "department", "Unknown"
                            ),
                            "title": (
                                metadata.get("course_title")
                                or metadata.get("course_name")
                                or metadata.get("program")
                                or metadata.get("department")
                                or "UCSB Engineering document"
                            ),
                        })

                    return {
                        "query": query,
                        "response": response.text,
                        "sources": sources,
                        "context_docs_used": len(docs),
                        "success": True,
                    }

                except Exception as e:
                    print(f"Gemini generation failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    else:
                        raise

        except Exception as e:
            print(f"Response generation error: {e}")

        # Fallback: show matching text already stored in ChromaDB.
        docs = retrieval_results.get("results", [])
        if docs:
            excerpts = []
            sources = []

            for doc in docs[:3]:
                metadata = doc.get("metadata", {}) or {}
                content = doc.get("content", "") or ""
                excerpts.append(content[:800])
                sources.append({
                    "type": metadata.get("type", "document"),
                    "department": metadata.get("department", "Unknown"),
                    "title": (
                        metadata.get("course_title")
                        or metadata.get("course_name")
                        or metadata.get("program")
                        or metadata.get("department")
                        or "UCSB Engineering document"
                    ),
                })

            return {
                "query": query,
                "response": (
                    "Gemini is currently unavailable. Here is matching text "
                    "from your saved UCSB Engineering documents:\n\n"
                    + "\n\n---\n\n".join(excerpts)
                ),
                "sources": sources,
                "context_docs_used": len(docs),
                "success": True,
            }

        return {
            "query": query,
            "response": (
                "I couldn't retrieve an answer right now. The Google API "
                "may be unavailable or the saved documents may not match "
                "your question. Please try different keywords."
            ),
            "sources": [],
            "success": False,
        }

    def chat_session(self):
        """Run a command-line chat session."""
        print("UCSB Engineering Chatbot")
        print("Type 'quit' to exit.")

        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() in {"quit", "exit", "q"}:
                    print("Goodbye!")
                    break
                if not query:
                    continue

                result = self.generate_response(query)
                print(f"\nBot: {result['response']}")

                for source in result.get("sources", [])[:3]:
                    print(
                        f"Source: {source['title']} "
                        f"({source['department']})"
                    )

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Chat error: {e}")

    def batch_test_queries(
        self, test_queries: List[str]
    ) -> List[Dict[str, Any]]:
        """Run multiple test queries."""
        results = []
        for i, query in enumerate(test_queries, 1):
            print(f"\nTesting {i}/{len(test_queries)}: {query}")
            results.append(self.generate_response(query))
        return results

    def print_test_results(self, results: List[Dict[str, Any]]):
        """Print batch test results."""
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. QUERY: {result['query']}")
            print(f"   SUCCESS: {result['success']}")
            print(
                f"   SOURCES: {result.get('context_docs_used', 0)} documents"
            )
            print(f"   RESPONSE: {result['response'][:200]}...")

            for source in result.get("sources", [])[:2]:
                print(
                    f"   - {source['title']} ({source['department']})"
                )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        try:
            return {
                "collection_name": self.collection_name,
                "document_count": self.collection.count(),
                "status": "connected",
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "status": "error",
                "error": str(e),
            }


if __name__ == "__main__":
    try:
        generator = GeminiResponseGenerator()
        print("\nChoose testing mode:")
        print("1. Interactive chat")
        print("2. Batch test")
        print("3. Single query")

        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":
            generator.chat_session()
        elif choice == "2":
            queries = [
                "What computer science courses are available?",
                "Tell me about the mechanical engineering program",
                "What are the prerequisites for CS 162?",
                "Show me electrical engineering courses about circuits",
                "What programs does the Materials department offer?",
                "How many units is ECE 10A?",
                "What courses can I take for machine learning?",
                "Tell me about the Computer Engineering program",
            ]
            generator.print_test_results(
                generator.batch_test_queries(queries)
            )
        elif choice == "3":
            query = input("Enter your query: ").strip()
            if query:
                result = generator.generate_response(query)
                print(f"\nResponse: {result['response']}")
                for source in result.get("sources", []):
                    print(
                        f"- {source['title']} ({source['department']})"
                    )
        else:
            print("Invalid choice.")
    except Exception as e:
        print(f"Error initializing response generator: {e}")
