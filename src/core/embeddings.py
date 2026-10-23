
import os
import json
import time
from typing import List, Dict, Any

import google.generativeai as genai
import chromadb
from chromadb.config import Settings

from src.config.settings import Config


class GeminiDocumentEmbedder:
    """Embed documents using Google Gemini and store them in ChromaDB."""

    def __init__(self, collection_name: str = None):
        self.config = Config()

        is_valid, error_msg = self.config.validate_config()
        if not is_valid:
            raise ValueError(f"Configuration error: {error_msg}")

        genai.configure(api_key=self.config.GOOGLE_API_KEY)

        self.embedding_model = self.config.EMBEDDING_MODEL
        self.collection_name = collection_name or self.config.COLLECTION_NAME

        self.chroma_client = chromadb.PersistentClient(
            path=self.config.CHROMADB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "UCSB College of Engineering documents"
            },
        )

        print(f"Initialized ChromaDB collection: {self.collection_name}")
        print(f"ChromaDB path: {self.config.CHROMADB_PATH}")

    def get_embedding(
        self, text: str, retries: int = 3
    ) -> List[float]:
        """Get a text embedding with retry logic."""

        for attempt in range(retries):
            try:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document",
                )
                return result["embedding"]

            except Exception as e:
                print(
                    f"Embedding attempt {attempt + 1} "
                    f"failed: {e}"
                )

                if attempt < retries - 1:
                    error_message = str(e).lower()

                    if "quota" in error_message or "rate" in error_message:
                        wait_time = 60
                    else:
                        wait_time = 2 ** attempt

                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError("Failed to generate embedding.")

    def chunk_text(
        self,
        text: str,
        max_chars: int = 1000,
        overlap: int = 100,
    ) -> List[str]:
        """Split text into chunks while always advancing the start position."""

        if not text or not text.strip():
            return []

        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero.")

        # Overlap must be smaller than the chunk size.
        overlap = max(0, min(overlap, max_chars - 1))

        if len(text) <= max_chars:
            return [text.strip()]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + max_chars, text_length)

            if end < text_length:
                sentence_end = text.rfind(".", start, end)

                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    word_end = text.rfind(" ", start, end)

                    if word_end > start:
                        end = word_end

            # Guarantee that the end position moves forward.
            if end <= start:
                end = min(start + max_chars, text_length)

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            # Guarantee that the next start is greater than the current start.
            if end >= text_length:
                break

            next_start = max(start + 1, end - overlap)
            start = next_start

        return chunks

    def embed_documents(
        self,
        documents_file: str = None,
        batch_size: int = 5,
    ):
        """Embed documents and store them in ChromaDB."""

        if documents_file is None:
            documents_file = os.path.join(
                self.config.DATA_PATH,
                "processed_documents.json",
            )

        if not os.path.exists(documents_file):
            print(f"Error: {documents_file} not found!")
            print("Run the data processor first:")
            print("python -m src.data.data_processor")
            return 0

        with open(documents_file, "r", encoding="utf-8") as file:
            documents = json.load(file)

        if not isinstance(documents, list):
            raise ValueError(
                "processed_documents.json must contain a list of documents."
            )

        print(f"Processing {len(documents)} documents...")

        # Start with an empty collection for a fresh indexing run.
        try:
            existing_ids = self.collection.get(include=[])["ids"]

            if existing_ids:
                self.collection.delete(ids=existing_ids)
                print(f"Cleared {len(existing_ids)} existing chunks.")
        except Exception as e:
            print(f"Could not clear existing collection: {e}")
            raise

        batch_docs = []
        batch_embeddings = []
        batch_metadatas = []
        batch_ids = []
        processed_count = 0

        for i, doc in enumerate(documents):
            doc_id = doc.get("doc_id", f"document_{i}")

            try:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})

                if not isinstance(content, str) or not content.strip():
                    print(f"Skipping empty document: {doc_id}")
                    continue

                chunks = self.chunk_text(content)

                print(
                    f"Document {i + 1}/{len(documents)}: "
                    f"{doc_id} — {len(chunks)} chunks"
                )

                for chunk_idx, chunk in enumerate(chunks):
                    print(
                        f"Creating embedding for chunk "
                        f"{chunk_idx + 1}/{len(chunks)} "
                        f"of {doc_id}..."
                    )

                    embedding = self.get_embedding(chunk)

                    if len(chunks) > 1:
                        chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                    else:
                        chunk_id = doc_id

                    batch_docs.append(chunk)
                    batch_embeddings.append(embedding)
                    batch_metadatas.append({
                        **metadata,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                    })
                    batch_ids.append(chunk_id)
                    processed_count += 1

                    if len(batch_docs) >= batch_size:
                        self._add_batch_to_collection(
                            batch_docs,
                            batch_embeddings,
                            batch_metadatas,
                            batch_ids,
                        )

                        print(
                            f"Added batch of {len(batch_docs)} "
                            f"chunks to collection."
                        )

                        batch_docs = []
                        batch_embeddings = []
                        batch_metadatas = []
                        batch_ids = []

                    # Small delay to reduce request frequency.
                    time.sleep(1)

            except Exception as e:
                print(f"Error processing document {doc_id}: {e}")
                continue

        # Save any remaining chunks.
        if batch_docs:
            self._add_batch_to_collection(
                batch_docs,
                batch_embeddings,
                batch_metadatas,
                batch_ids,
            )
            print(
                f"Added final batch of {len(batch_docs)} chunks "
                "to collection."
            )

        final_count = self.collection.count()

        print("\nEmbedding complete!")
        print(f"Successfully processed chunks: {processed_count}")
        print(f"Total chunks in collection: {final_count}")

        if final_count == 0:
            print("Warning: No chunks were added.")
            print("Check processed_documents.json.")

        return final_count

    def _add_batch_to_collection(
        self,
        docs: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str],
    ):
        """Add a batch of documents to ChromaDB."""

        self.collection.add(
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def test_retrieval(
        self,
        query: str,
        n_results: int = None,
    ):
        """Test retrieval using a sample query."""

        if n_results is None:
            n_results = self.config.DEFAULT_RETRIEVAL_COUNT

        try:
            query_result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query",
            )

            query_embedding = query_result["embedding"]
            available_count = self.collection.count()

            if available_count == 0:
                print("The collection is empty. Embed documents first.")
                return

            n_results = min(n_results, available_count)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )

            print(f"\nQuery: {query}")
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            print(f"Found {len(documents)} results:")

            for i, (document, metadata) in enumerate(
                zip(documents, metadatas),
                start=1,
            ):
                doc_type = metadata.get("type", "unknown")

                if doc_type == "course":
                    identifier = metadata.get(
                        "course_name", "Unknown Course"
                    )
                elif doc_type == "program":
                    identifier = metadata.get(
                        "program", "Unknown Program"
                    )
                elif doc_type == "department":
                    identifier = metadata.get(
                        "department", "Unknown Department"
                    )
                else:
                    identifier = metadata.get(
                        "title", "Unknown"
                    )

                print(f"\n{i}. {doc_type.title()}: {identifier}")
                print(
                    "   Department: "
                    f"{metadata.get('department', 'Unknown')}"
                )
                print(f"   Preview: {document[:200]}...")

        except Exception as e:
            print(f"Error testing retrieval: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""

        try:
            return {
                "collection_name": self.collection_name,
                "document_count": self.collection.count(),
                "chromadb_path": self.config.CHROMADB_PATH,
                "embedding_model": self.embedding_model,
            }
        except Exception as e:
            return {"error": str(e)}

    def query_documents(
        self,
        query: str,
        n_results: int = None,
        include_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """Retrieve relevant documents for a query."""

        if n_results is None:
            n_results = self.config.DEFAULT_RETRIEVAL_COUNT

        try:
            query_result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query",
            )

            query_embedding = query_result["embedding"]
            available_count = self.collection.count()

            if available_count == 0:
                return {"query": query, "results": {}, "count": 0}

            n_results = min(n_results, available_count)

            include_list = ["documents", "metadatas", "distances"]

            if include_embeddings:
                include_list.append("embeddings")

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=include_list,
            )

            return {
                "query": query,
                "results": results,
                "count": len(results["documents"][0]),
            }

        except Exception as e:
            return {"error": str(e), "query": query}


if __name__ == "__main__":
    print("Starting UCSB Document Embedding Process")
    print("=" * 50)

    try:
        embedder = GeminiDocumentEmbedder()

        stats = embedder.get_collection_stats()
        print(f"Current collection stats: {stats}")

        document_count = embedder.embed_documents()

        if document_count > 0:
            print("\n" + "=" * 50)
            print("Testing retrieval with sample queries:")
            print("=" * 50)

            test_queries = [
                "What computer science courses are available?",
                "Tell me about the mechanical engineering program",
                "What are the requirements for electrical engineering?",
                "Show me courses about machine learning",
            ]

            for query in test_queries:
                embedder.test_retrieval(query)
                print("-" * 30)
        else:
            print(
                "\nNo documents embedded. Check your processed data "
                "and any errors above."
            )

    except Exception as e:
        print(f"Error during embedding process: {e}")
        print("Check your Google API key, processed documents,")
        print("and ChromaDB configuration.")
