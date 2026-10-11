"""Central config. Everything tunable lives here so the eval harness can sweep it."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="RESEARCHMIND_", extra="ignore"
    )

    # models (served locally by ollama + sentence-transformers)
    chat_model: str = "qwen2.5:3b"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_host: str = "http://localhost:11434"

    # chunking - sizes are in characters. Crude but it keeps the tokenizer
    # dependency out of the ingest path, and every model we use here is
    # sub-8k context anyway.
    chunk_size: int = 900
    chunk_overlap: int = 160

    # retrieval
    top_k: int = 6
    dense_candidates: int = 20
    sparse_candidates: int = 20
    rrf_k: int = 60

    # agent
    max_agent_steps: int = 6
    temperature: float = 0.1

    # Output caps. Without these a 3B model will occasionally not stop: one
    # eval run generated 41k tokens on a single question and took 7 minutes.
    # The cap is the difference between a slow answer and a hung app.
    max_answer_tokens: int = 450
    max_helper_tokens: int = 250

    # storage
    upload_dir: Path = DATA_DIR / "uploads"
    chroma_dir: Path = DATA_DIR / "chroma"
    registry_db: Path = DATA_DIR / "registry.db"
    collection: str = "papers"

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
