"""
ETL Pipeline: Embedder & Vector Store
Embeds extracted chunks and stores them in ChromaDB with hybrid search support.
"""

import os
import json
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from etl.pdf_processor import TextChunk, ExtractedTable


class VectorStore:
    """ChromaDB-based vector store with BM25 hybrid search."""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model_name

        print(f"[VectorStore] Loading embedding model: {embedding_model_name}")
        self.embed_model = SentenceTransformer(embedding_model_name)

        print(f"[VectorStore] Initializing ChromaDB at: {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Create/get collection
        self.collection = self.client.get_or_create_collection(
            name="cyber_ireland_2022",
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 index (rebuilt on load)
        self._bm25_corpus: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_metadata: list[dict] = []

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_model.encode(text, normalize_embeddings=True).tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""
        return self.embed_model.encode(texts, normalize_embeddings=True).tolist()

    def add_chunks(self, chunks: list[TextChunk], batch_size: int = 50):
        """Add text chunks to the vector store."""
        print(f"[VectorStore] Adding {len(chunks)} chunks to vector store...")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            embeddings = self.embed_texts(documents)
            metadatas = [
                {
                    "page_number": c.page_number,
                    "chunk_type": c.chunk_type,
                    "section": c.section,
                    # Store table metadata as JSON string if present
                    "table_id": c.metadata.get("table_id", ""),
                    "caption": c.metadata.get("caption", ""),
                    "markdown": c.metadata.get("markdown", "")[:1000],
                    "raw_data_json": c.metadata.get("raw_data_json", "")[:5000],
                }
                for c in batch
            ]

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            if (i + batch_size) % 100 == 0 or i + batch_size >= len(chunks):
                print(
                    f"[VectorStore] Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks"
                )

        # Rebuild BM25 index
        self._build_bm25_index()
        print(f"[VectorStore] Done. Collection size: {self.collection.count()}")

    def _build_bm25_index(self):
        """Build BM25 index from all documents in the collection."""
        print("[VectorStore] Building BM25 keyword index...")
        results = self.collection.get(include=["documents", "metadatas"])
        if not results["documents"]:
            return

        self._bm25_corpus = results["documents"]
        self._bm25_ids = results["ids"]
        self._bm25_metadata = results["metadatas"]

        # Tokenize for BM25
        tokenized = [doc.lower().split() for doc in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized)
        print(f"[VectorStore] BM25 index built with {len(self._bm25_corpus)} documents")

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        filter_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid search combining vector similarity and BM25 keyword matching.

        Returns ranked list of results with scores.
        """
        # Ensure BM25 index exists
        if self._bm25_index is None:
            self._build_bm25_index()

        # --- Vector search ---
        where_filter = {"chunk_type": filter_type} if filter_type else None
        query_embedding = self.embed_text(query)

        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 2, self.collection.count()),
            include=["documents", "metadatas", "distances"],
            where=where_filter,
        )

        # Build vector score map (convert distance to similarity)
        vector_scores: dict[str, float] = {}
        if vector_results["ids"] and vector_results["ids"][0]:
            for idx, doc_id in enumerate(vector_results["ids"][0]):
                # ChromaDB cosine distance: lower = more similar
                distance = vector_results["distances"][0][idx]
                similarity = 1.0 - distance
                vector_scores[doc_id] = similarity

        # --- BM25 search ---
        bm25_scores: dict[str, float] = {}
        if self._bm25_index and self._bm25_corpus:
            tokenized_query = query.lower().split()
            scores = self._bm25_index.get_scores(tokenized_query)

            # Normalize BM25 scores
            max_score = max(scores) if max(scores) > 0 else 1.0
            for idx, score in enumerate(scores):
                if score > 0:
                    bm25_scores[self._bm25_ids[idx]] = score / max_score

        # --- Combine scores (Reciprocal Rank Fusion-inspired) ---
        all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        combined: list[tuple[str, float]] = []

        for doc_id in all_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            b_score = bm25_scores.get(doc_id, 0.0)
            combined_score = (vector_weight * v_score) + (bm25_weight * b_score)
            combined.append((doc_id, combined_score))

        # Sort by combined score descending
        combined.sort(key=lambda x: x[1], reverse=True)

        # Fetch full results for top-k
        top_ids = [doc_id for doc_id, _ in combined[:top_k]]
        if not top_ids:
            return []

        full_results = self.collection.get(
            ids=top_ids,
            include=["documents", "metadatas"],
        )

        # Build result list maintaining order
        id_to_result = {}
        for i, doc_id in enumerate(full_results["ids"]):
            id_to_result[doc_id] = {
                "id": doc_id,
                "content": full_results["documents"][i],
                "metadata": full_results["metadatas"][i],
            }

        results = []
        for doc_id, score in combined[:top_k]:
            if doc_id in id_to_result:
                result = id_to_result[doc_id].copy()
                result["score"] = round(score, 4)
                result["vector_score"] = round(vector_scores.get(doc_id, 0.0), 4)
                result["bm25_score"] = round(bm25_scores.get(doc_id, 0.0), 4)
                results.append(result)

        return results

    def search_tables(self, query: str, top_k: int = 5) -> list[dict]:
        """Search specifically within table chunks."""
        return self.hybrid_search(
            query, top_k=top_k, filter_type="table"
        )

    def get_page_content(self, page_number: int) -> list[dict]:
        """Get all content from a specific page."""
        results = self.collection.get(
            where={"page_number": page_number},
            include=["documents", "metadatas"],
        )
        return [
            {
                "id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i],
            }
            for i in range(len(results["ids"]))
        ]


def run_etl_pipeline(
    pdf_path: str,
    persist_dir: str = "./chroma_db",
    data_dir: str = "./data",
    embedding_model: str = "all-MiniLM-L6-v2",
):
    """Run the full ETL pipeline: extract, embed, store."""
    from etl.pdf_processor import extract_pdf, save_extraction_results

    # Step 1: Extract
    print("=" * 60)
    print("[ETL] Step 1: Extracting PDF content...")
    print("=" * 60)
    chunks, tables = extract_pdf(pdf_path)
    save_extraction_results(chunks, tables, data_dir)

    # Step 2: Embed & Store
    print("=" * 60)
    print("[ETL] Step 2: Embedding and storing in vector database...")
    print("=" * 60)
    store = VectorStore(persist_dir=persist_dir, embedding_model_name=embedding_model)
    store.add_chunks(chunks)

    # Step 3: Verify
    print("=" * 60)
    print("[ETL] Step 3: Verification...")
    print("=" * 60)
    print(f"[ETL] Collection count: {store.collection.count()}")

    # Test search
    test_queries = [
        "total number of jobs",
        "Pure-Play cybersecurity firms South-West",
        "2030 job target CAGR growth",
    ]
    for q in test_queries:
        results = store.hybrid_search(q, top_k=3)
        print(f"\n[ETL] Test query: '{q}'")
        for r in results:
            print(f"  Score: {r['score']:.4f} | Page: {r['metadata']['page_number']} | Type: {r['metadata']['chunk_type']}")
            print(f"  Content: {r['content'][:120]}...")

    print("\n" + "=" * 60)
    print("[ETL] Pipeline complete!")
    print("=" * 60)

    return store
