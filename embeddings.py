import math
from fastembed import TextEmbedding

# Load a lightweight, high-performance embedding model (BAAI/bge-small-en-v1.5)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_embedding(text: str) -> list[float]:
    """Generate a vector embedding for a given text strings."""
    # fastembed returns a generator of numpy arrays
    embeddings = list(embedding_model.embed([text]))
    return embeddings[0].tolist()

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate the cosine similarity between two vector lists."""
    dot_product = sum(a*b for a, b in zip(vec_a,vec_b))
    norm_a = math.sqrt(sum(a*a for a in vec_a))
    norm_b = math.sqrt(sum(b*b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product/(norm_a * norm_b)