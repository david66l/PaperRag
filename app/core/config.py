"""Centralized configuration using pydantic-settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PAPERRAG_",
        case_sensitive=False,
    )

    # --- Paths ---
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_dir: Path = Field(default=Path("data"))
    index_dir: Path = Field(default=Path("indexes"))

    @property
    def abs_data_dir(self) -> Path:
        if self.data_dir.is_absolute():
            return self.data_dir
        return self.project_root / self.data_dir

    @property
    def abs_index_dir(self) -> Path:
        if self.index_dir.is_absolute():
            return self.index_dir
        return self.project_root / self.index_dir

    # --- Ingestion ---
    load_limit: int = Field(default=0, description="Max papers to load, 0 = unlimited")
    default_data_file: str = Field(default="arxiv-metadata-oai-snapshot.json")

    # --- Chunking ---
    chunk_strategy: Literal["metadata", "fixed", "recursive"] = "metadata"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Embedding ---
    embedding_provider: Literal["local", "api"] = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 256
    embedding_api_url: str = ""
    embedding_api_key: str = ""

    # --- FAISS ---
    faiss_index_type: Literal["flat", "ivf"] = "flat"
    faiss_nprobe: int = 10

    # --- Retrieval ---
    top_k_dense: int = 30
    top_k_bm25: int = 30
    top_n_fused: int = 20
    top_n_final: int = 10
    top_n_context: int = 5
    context_max_tokens: int = 3000

    # --- Fusion ---
    fusion_strategy: Literal["weighted", "rrf"] = "rrf"
    dense_weight: float = 0.6
    bm25_weight: float = 0.3
    metadata_weight: float = 0.1
    rrf_k: int = 60

    # --- Reranker ---
    rerank_enabled: bool = False
    reranker_provider: Literal["local", "api", "none"] = "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_api_url: str = ""
    reranker_api_key: str = ""

    # --- LLM ---
    llm_provider: Literal["openai_compatible", "anthropic"] = "openai_compatible"
    llm_api_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000


def get_settings() -> Settings:
    return Settings()
