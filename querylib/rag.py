from pathlib import Path
import orjson
import faiss
import numpy as np
import os
import functools
import concurrent.futures
from typing import List, Dict, Any, Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.schema import TextNode

# Fix the thread count calculation (avoid potential AttributeError)
THREAD_COUNT = os.cpu_count() // 2 + 1 if os.cpu_count() else 4
faiss.omp_set_num_threads(THREAD_COUNT)


class DocumentationRAG:
    """
    Optimized CPU-only Retrieval-Augmented Generation (RAG) system for
    querying documentation.
    """

    def __init__(
        self,
        json_path: str,
        embedding_file: Optional[str] = None,
        batch_size: int = 64,
        similarity_top_k: int = 5,
    ):
        """
        Initializes the RAG system.

        Args:
            json_path (str): Path to the JSON file containing structured documentation.
            embedding_file (str, optional): Path to load/save embeddings.
            batch_size (int): Batch size for embedding processing.
            similarity_top_k (int): Number of top results to return.
        """
        self.file_path = str(Path(json_path))
        self._documents = []
        self._index = None
        self._query_engine = None
        self._embedding_file = str(Path(embedding_file)) if embedding_file else None
        self.batch_size = batch_size
        self.similarity_top_k = similarity_top_k

        # Preload JSON data
        self._data = self._load_json()
        self._setup_rag(embedding_file=self._embedding_file)

    def _load_json(self) -> Dict[str, Any]:
        """Load JSON data efficiently using orjson."""
        try:
            with open(self.file_path, "rb") as f:
                return orjson.loads(f.read())
        except Exception as e:
            raise ValueError(f"Error loading JSON: {e}")

    def _extract_docs(self, data: Any) -> List[Document]:
        """
        Iteratively extracts documentation fields from the JSON structure.
        Uses a non-recursive approach for better performance.

        Args:
            data (dict or list): JSON content.

        Returns:
            list[Document]: Extracted documentation as Document objects.
        """
        stack = [data] if isinstance(data, dict) else list(data)
        documents = []

        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue

            if "summary" in node and node.get("summary"):
                name = node.get("name", "")
                node_type = node.get("type", "")
                summary = node.get("summary", "")

                text_content = f"{name} ({node_type}): {summary}"

                if "returns" in node and node["returns"]:
                    returns = node["returns"]
                    text_content += f"\nReturns: {returns.get('name', '')} ({returns.get('type', '')})"

                metadata = {
                    "path": node.get("path", ""),
                    "name": name,
                    "type": node_type,
                    "is_class": node_type == "class",
                    "is_function": node_type == "function",
                }

                # Include only non-empty metadata fields to reduce memory usage
                if returns := node.get("returns"):
                    metadata["returns"] = returns

                if raises := node.get("raises"):
                    metadata["raises"] = raises

                if params := node.get("params"):
                    metadata["params"] = params

                documents.append(Document(text=text_content, metadata=metadata))

            # Add methods to the stack, if they exist
            if "methods" in node and node["methods"]:
                stack.extend(node["methods"])

        return documents

    def _compute_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings in batches using parallel processing."""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Use ThreadPoolExecutor for parallel embedding computation
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=THREAD_COUNT
            ) as executor:
                batch_embeddings = list(
                    executor.map(self.embed_model.get_text_embedding, batch)
                )

            all_embeddings.extend(batch_embeddings)

            # Optional progress indication for large datasets
            if i % (self.batch_size * 10) == 0 and i > 0:
                print(f"Processed {i}/{len(texts)} embeddings")

        return np.array(all_embeddings, dtype=np.float32)

    def _setup_rag(self, embedding_file: Optional[str] = None):
        """
        Sets up the optimized RAG system with optional embedding file storage.

        Args:
            embedding_file (str, optional): Path to store/load precomputed embeddings.
        """
        # Disable LLM usage
        Settings.llm = None

        # Extract documents
        self._documents = self._extract_docs(self._data)
        texts = [doc.text for doc in self._documents]

        # Initialize embedding model with optimized settings
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        Settings.embed_model = self.embed_model

        # Load or compute embeddings
        if embedding_file and os.path.exists(embedding_file):
            try:
                embeddings = np.load(embedding_file)
                print(f"Loaded embeddings from {embedding_file}")
            except Exception as e:
                print(f"Error loading embeddings, computing new ones: {e}")
                embeddings = self._compute_embeddings_batch(texts)
                np.save(embedding_file, embeddings)
        else:
            # Default embedding file path if not provided
            if not embedding_file:
                embedding_file = self.file_path.rsplit(".", 1)[0] + "_embeddings.npy"

            # Compute embeddings in optimized batches
            embeddings = self._compute_embeddings_batch(texts)

            # Save embeddings
            np.save(embedding_file, embeddings)
            print(f"Embedding file saved at {embedding_file}")

        # Convert embeddings to contiguous NumPy array for FAISS efficiency
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # FAISS: Use HNSW index with optimized parameters
        embed_dim = embeddings.shape[1]

        # Create optimized index for CPU performance
        # M=16 provides good balance between quality and speed for CPU
        # efConstruction=80 for reasonable build time but good quality
        faiss_index = faiss.IndexHNSWFlat(embed_dim, 16)
        faiss_index.hnsw.efConstruction = 80
        faiss_index.hnsw.efSearch = 64  # Affects search speed/quality tradeoff

        # Add vectors to index
        faiss_index.add(embeddings)

        # Create nodes with pre-computed embeddings for efficiency
        nodes = []
        for i, doc in enumerate(self._documents):
            node = TextNode(text=doc.text, metadata=doc.metadata)
            nodes.append(node)

        # Store in FAISS VectorStore
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)

        # Create VectorStoreIndex with Nodes
        self._index = VectorStoreIndex(nodes=nodes, vector_store=self.vector_store)

        # Create an optimized query engine
        self._query_engine = self._index.as_query_engine(
            similarity_top_k=self.similarity_top_k, response_mode="compact"
        )

        # Clean up data not needed after initialization
        self._data = None  # Free memory from the original JSON data

    @functools.lru_cache(
        maxsize=128
    )  # Use lru_cache instead of cache for better control
    def ask_query(self, query: str) -> Dict[str, Any]:
        """
        Queries the documentation using FAISS-based retrieval.
        Results are cached for efficiency.

        Args:
            query (str): The user query.

        Returns:
            dict: Retrieved documents with metadata and relevance scores.
        """
        response = self._query_engine.query(query)

        # Process results efficiently
        results = [
            {
                "text": node.node.text,
                "score": float(node.score),  # Convert numpy float to Python float
                "metadata": node.node.metadata,
            }
            for node in response.source_nodes
        ]

        return {
            "query": query,
            "results": results,
        }

    def clear_cache(self):
        """Clear the query cache when needed."""
        self.ask_query.cache_clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the RAG system."""
        return {
            "document_count": len(self._documents),
            "cache_info": self.ask_query.cache_info()._asdict(),
            "embedding_dim": self.vector_store.faiss_index.d,
        }
