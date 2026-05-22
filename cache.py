# cache.py
import hashlib
import time
from collections import defaultdict
from typing import Dict, Any, Optional

import numpy as np
import redis.asyncio as aioredis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

from config import ModelName, MODEL_PRICING

# ============================================================================
# 1. Thread-Isolated Embedding Setup
# ============================================================================
print("Loading sentence-transformers embedding framework...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384
INDEX_NAME = "semantic_cache"


# ============================================================================
# 2. Redis Vector Cache Engine (Async-Native)
# ============================================================================
class RedisSemanticCache:
    def __init__(self, redis_url: str = "redis://localhost:6379", distance_threshold: float = 0.14, ttl_seconds: int = 3600):
        """
        For COSINE distance metrics in Redis VSS:
        - 0.0 is an identical match.
        - Distance values below 0.14-0.16 generally represent close semantic matches.
        """
        self.redis_url = redis_url
        self.threshold = distance_threshold
        self.ttl = ttl_seconds
        self.pool = aioredis.ConnectionPool.from_url(redis_url, decode_responses=True)
        self.client = aioredis.Redis(connection_pool=self.pool)
        self.hits = 0
        self.misses = 0

    async def initialize_index(self):
        """Creates the VSS Index using raw search fields if it does not exist."""
        try:
            await self.client.ft(INDEX_NAME).info()
            print("Redis vector semantic index detected.")
        except Exception:
            schema = (
                TextField("prompt"),
                TextField("response"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )
            # Index hashes beginning with 'cache:'
            await self.client.ft(INDEX_NAME).create_index(
                fields=schema,
                definition=IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH),
            )
            print("Successfully initialized fresh Redis VSS index mappings.")

    def _get_embedding(self, text: str) -> bytes:
        """Encodes string to raw FLOAT32 vector bytes using sentence-transformers."""
        return embedding_model.encode(text).astype(np.float32).tobytes()

    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Queries Redis Vector Space using Async K-Nearest Neighbors."""
        query_vector = self._get_embedding(query)
        
        # Redis Vector Search syntax: Find 1 nearest neighbor
        base_query = "*=>[KNN 1 @embedding $vec_param AS vector_distance]"
        q = (
            Query(base_query)
            .return_fields("prompt", "response", "vector_distance")
            .sort_by("vector_distance")
            .paging(0, 1)
            .dialect(2)
        )
        
        try:
            results = await self.client.ft(INDEX_NAME).search(q, query_params={"vec_param": query_vector})
            if results.docs:
                match = results.docs[0]
                distance = float(match.vector_distance)
                
                if distance <= self.threshold:
                    self.hits += 1
                    # Invert distance back to a standard user-facing similarity score
                    similarity = round(1.0 - distance, 4)
                    return {
                        "response": match.response,
                        "similarity": similarity
                    }
        except Exception as e:
            print(f"Cache lookup anomaly suppressed: {e}")

        self.misses += 1
        return None

    async def put(self, query: str, response: str):
        """Persists the payload deterministically with an active system TTL."""
        digest = hashlib.md5(query.encode()).hexdigest()
        doc_id = f"cache:{digest}"
        
        mapping = {
            "prompt": query,
            "response": response,
            "embedding": self._get_embedding(query)
        }
        
        # Execute pipeline non-blockingly via async context managers
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.hset(doc_id, mapping=mapping)
            pipe.expire(doc_id, self.ttl)
            await pipe.execute()


# ============================================================================
# 3. Cost Accounting Telemetry Tracker
# ============================================================================
class CostTracker:
    def __init__(self):
        self.total_requests = 0
        self.total_cost_usd = 0.0
        self.cost_by_user = defaultdict(float)

    def record(self, user_id: str, model_enum: ModelName, input_t: int, output_t: int):
        pricing = MODEL_PRICING.get(model_enum, {"input": 0.0, "output": 0.0})
        cost = ((input_t / 1_000_000) * pricing["input"]) + ((output_t / 1_000_000) * pricing["output"])
        
        self.total_requests += 1
        self.total_cost_usd += cost
        self.cost_by_user[user_id] += cost
        return cost

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "top_users": dict(sorted(self.cost_by_user.items(), key=lambda x: x[1], reverse=True)[:5])
        }