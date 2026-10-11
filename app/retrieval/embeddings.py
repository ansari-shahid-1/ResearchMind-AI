"""Sentence-transformers wrapper. The model is loaded once per process and
kept around, because loading all-MiniLM takes a couple of seconds and we do
not want that on every request.

Vectors are L2-normalised so cosine and dot product agree, which lets us
switch the Chroma space without re-embedding anything.
"""

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=2)
def _load(model_name: str):
    # imported lazily - torch is slow to import and the eval scripts don't
    # always need it
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    model = _load(model_name or settings.embed_model)
    vectors = model.encode(
        list(texts),
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def embed_one(text: str, model_name: str | None = None) -> list[float]:
    return embed([text], model_name)[0]


def dimension(model_name: str | None = None) -> int:
    return _load(model_name or settings.embed_model).get_sentence_embedding_dimension()
