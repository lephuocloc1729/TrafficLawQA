from uuid import uuid4
from typing import List, Dict, Optional, Union

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse


class QdrantVectorStore:
    def __init__(self, uri: str):
        self.uri = uri
        self.client = QdrantClient(url=self.uri, check_compatibility=False)

    def create_collection(
        self,
        collection_name: str,
        embedding_size: int,
        distance: Union[str, models.Distance] = "cosine",
    ) -> None:
        if isinstance(distance, str):
            distance = self._parse_distance_metric(distance)

        try:
            if not self.client.collection_exists(collection_name=collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size,
                        distance=distance,
                    ),
                    on_disk_payload=True,
                )
        except UnexpectedResponse as e:
            raise RuntimeError(f"Failed to create collection '{collection_name}': {e}")

    def add_embeddings(
        self,
        collection_name: str,
        embeddings: List[List[float]],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[Union[str, int]]] = None,
        batch_size: int = 32,
    ) -> None:
        if payloads and len(payloads) != len(embeddings):
            raise ValueError("Payloads must match the number of embeddings.")

        if ids and len(ids) != len(embeddings):
            raise ValueError("IDs must match the number of embeddings.")

        points = [
            models.PointStruct(
                id=ids[i] if ids else str(uuid4()),
                vector=embedding,
                payload=payloads[i] if payloads else None,
            )
            for i, embedding in enumerate(embeddings)
        ]

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch)

    def query(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[models.ScoredPoint]:
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

    @staticmethod
    def _parse_distance_metric(metric: str) -> models.Distance:
        """Convert a distance metric string to the corresponding Qdrant Distance enum."""
        metric = metric.lower()
        match metric:
            case "cosine":
                return models.Distance.COSINE
            case "euclid":
                return models.Distance.EUCLID
            case "dot":
                return models.Distance.DOT
            case "manhattan":
                return models.Distance.MANHATTAN
            case _:
                raise ValueError(f"Unsupported distance metric: '{metric}'")
