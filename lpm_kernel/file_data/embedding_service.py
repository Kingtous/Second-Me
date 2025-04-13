from typing import List, Tuple
import chromadb
from chromadb.utils import embedding_functions
import os
from .dto.chunk_dto import ChunkDTO
from lpm_kernel.common.llm import LLMClient
from lpm_kernel.file_data.document_dto import DocumentDTO
from typing import List, Dict, Optional
from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


class EmbeddingService:
    def __init__(self):
        chroma_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.llm_client = LLMClient()

        # document level collection
        self.document_collection = self.client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine", "dimension": 1536}
        )

        # chunk level collection
        self.chunk_collection = self.client.get_or_create_collection(
            name="document_chunks", metadata={"hnsw:space": "cosine", "dimension": 1536}
        )

    def generate_document_embedding(self, document: DocumentDTO) -> List[float]:
        """Process document level embedding and store in ChromaDB"""
        try:
            if not document.raw_content:
                logger.warning(
                    f"Document {document.id} has no content to process embedding"
                )
                return None

            # get embedding
            logger.info(f"Generating embedding for document {document.id}")
            embeddings = self.llm_client.get_embedding([document.raw_content])

            if embeddings is None or len(embeddings) == 0:
                logger.error(f"Failed to get embedding for document {document.id}")
                return None

            embedding = embeddings[0]
            logger.info(f"Successfully got embedding for document {document.id}")

            # store to ChromaDB
            try:
                logger.info(f"Storing embedding for document {document.id} in ChromaDB")
                self.document_collection.add(
                    documents=[document.raw_content],
                    ids=[str(document.id)],
                    embeddings=[embedding.tolist()],
                    metadatas=[
                        {
                            "title": document.title or document.name,
                            "mime_type": document.mime_type,
                            "create_time": document.create_time.isoformat()
                            if document.create_time
                            else None,
                            "document_size": document.document_size,
                            "url": document.url,
                        }
                    ],
                )
                logger.info(f"Successfully stored embedding for document {document.id}")

                # verify embedding storage
                result = self.document_collection.get(
                    ids=[str(document.id)], include=["embeddings"]
                )
                if not result or result["embeddings"] is None:
                    logger.error(
                        f"Failed to verify embedding storage for document {document.id}"
                    )
                    return None
                logger.info(f"Verified embedding storage for document {document.id}")

                return embedding

            except Exception as e:
                logger.error(f"Error storing document embedding in ChromaDB: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error processing document embedding: {str(e)}")
            raise

    def generate_chunk_embeddings(self, chunks: List[ChunkDTO]) -> List[ChunkDTO]:
        """Process chunk level embeddings"""
        """
        Store in ChromaDB, the structure is as follows:
        documents=[c.content for c in unprocessed_chunks],
                    ids=[str(c.id) for c in unprocessed_chunks],
                    embeddings=embeddings.tolist(),
                    metadatas=[
                        {
                            "document_id": str(c.document_id),
                            "topic": c.topic or "",
                            "tags": ",".join(c.tags) if c.tags else "",
                        }
                        for c in unprocessed_chunks
                    ],
        """
        try:
            unprocessed_chunks = [c for c in chunks if not c.has_embedding]
            if not unprocessed_chunks:
                logger.info("No unprocessed chunks found")
                return chunks

            logger.info(f"Processing embeddings for {len(unprocessed_chunks)} chunks")

            contents = [c.content for c in unprocessed_chunks]
            logger.info("Getting embeddings from LLM service... {}".format(contents))
            embeddings = self.llm_client.get_embedding(contents)

            if embeddings is None or len(embeddings) == 0:
                logger.error("Failed to get embeddings from LLM service")
                return chunks

            logger.info(f"Successfully got embeddings with shape: {embeddings.shape}")

            try:
                logger.info("Adding embeddings to ChromaDB...")
                self.chunk_collection.add(
                    documents=[c.content for c in unprocessed_chunks],
                    ids=[str(c.id) for c in unprocessed_chunks],
                    embeddings=embeddings.tolist(),
                    metadatas=[
                        {
                            "document_id": str(c.document_id),
                            "topic": c.topic or "",
                            "tags": ",".join(c.tags) if c.tags else "",
                        }
                        for c in unprocessed_chunks
                    ],
                )
                logger.info("Successfully added embeddings to ChromaDB")

                # verify embeddings storage
                for chunk in unprocessed_chunks:
                    result = self.chunk_collection.get(
                        ids=[str(chunk.id)], include=["embeddings"]
                    )
                    if result and result["embeddings"] is not None:
                        chunk.has_embedding = True
                        logger.info(f"Verified embedding for chunk {chunk.id}")
                    else:
                        logger.warning(
                            f"Failed to verify embedding for chunk {chunk.id}"
                        )
                        chunk.has_embedding = False

            except Exception as e:
                logger.error(f"Error storing embeddings in ChromaDB: {str(e)}", exc_info=True)
                for chunk in unprocessed_chunks:
                    chunk.has_embedding = False
                raise

            return chunks

        except Exception as e:
            logger.error(f"Error processing chunk embeddings: {str(e)}", exc_info=True)
            raise

    def get_chunk_embedding_by_chunk_id(self, chunk_id: int) -> Optional[List[float]]:
        """Get the corresponding embedding vector by chunk_id

        Args:
            chunk_id (int): chunk ID

        Returns:
            List[float]: embedding vector, return None if not found

        Raises:
            ValueError: when chunk_id is invalid
            Exception: other errors
        """
        try:
            if not isinstance(chunk_id, int) or chunk_id < 0:
                raise ValueError("Invalid chunk_id")

            # query from ChromaDB
            result = self.chunk_collection.get(
                ids=[str(chunk_id)], include=["embeddings"]
            )

            if not result or result["embeddings"] is None:
                logger.warning(f"No embedding found for chunk {chunk_id}")
                return None

            return result["embeddings"][0]

        except Exception as e:
            logger.error(f"Error getting embedding for chunk {chunk_id}: {str(e)}")
            raise

    def get_document_embedding_by_document_id(
        self, document_id: int
    ) -> Optional[List[float]]:
        """Get the corresponding embedding vector by document_id

        Args:
            document_id (int): document ID

        Returns:
            List[float]: embedding vector, return None if not found

        Raises:
            ValueError: when document_id is invalid
            Exception: other errors
        """
        try:
            if not isinstance(document_id, int) or document_id < 0:
                raise ValueError("Invalid document_id")

            # query from ChromaDB
            result = self.document_collection.get(
                ids=[str(document_id)], include=["embeddings"]
            )

            if not result or result["embeddings"] is None:
                logger.warning(f"No embedding found for document {document_id}")
                return None

            return result["embeddings"][0]

        except Exception as e:
            logger.error(
                f"Error getting embedding for document {document_id}: {str(e)}"
            )
            raise

    def search_similar_chunks(
        self, query: str, limit: int = 5
    ) -> List[Tuple[ChunkDTO, float]]:
        """Search similar chunks, return list of ChunkDTO objects and their similarity scores

        Args:
            query (str): query text
            limit (int, optional): return result limit. Defaults to 5.

        Returns:
            List[Tuple[ChunkDTO, float]]: return list of (ChunkDTO, similarity score), sorted by similarity score in descending order

        Raises:
            ValueError: when query parameters are invalid
            Exception: other errors
        """
        try:
            if not query or not query.strip():
                raise ValueError("Query string cannot be empty")

            if limit < 1:
                raise ValueError("Limit must be positive")

            # calculate query text embedding
            query_embedding = self.llm_client.get_embedding([query])
            if query_embedding is None or len(query_embedding) == 0:
                raise Exception("Failed to generate embedding for query")

            # query ChromaDB
            results = self.chunk_collection.query(
                query_embeddings=[query_embedding[0].tolist()],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

            if not results or not results["ids"]:
                return []

            # convert results to ChunkDTO objects
            similar_chunks = []
            for i in range(len(results["ids"])):
                chunk_id = results["ids"][0][i]  # ChromaDB returns nested lists
                document_id = results["metadatas"][0][i]["document_id"]
                content = results["documents"][0][i]
                topic = results["metadatas"][0][i].get("topic", "")
                tags = (
                    results["metadatas"][0][i].get("tags", "").split(",")
                    if results["metadatas"][0][i].get("tags")
                    else []
                )

                # calculate similarity score (ChromaDB returns distances, need to convert to similarity)
                similarity_score = (
                    1 - results["distances"][0][i]
                )  # assume using Euclidean distance or cosine distance

                chunk = ChunkDTO(
                    id=int(chunk_id),
                    document_id=int(document_id),
                    content=content,
                    topic=topic,
                    tags=tags,
                    has_embedding=True,
                )

                similar_chunks.append((chunk, similarity_score))

            # sort by similarity score in descending order
            similar_chunks.sort(key=lambda x: x[1], reverse=True)

            return similar_chunks

        except ValueError as ve:
            logger.error(f"Invalid input parameters: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            raise
