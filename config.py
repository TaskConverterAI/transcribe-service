import os
import logging
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    device: str = "cpu"
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    whisper_model: str = "large"
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN", "")

    max_file_size: int = 25 * 1024 * 1024  # 25MB
    supported_formats: List[str] = [
        "audio/wav", "audio/mp3", "audio/m4a", "audio/ogg",
        "audio/flac", "audio/aac", "video/mp4"
    ]

    diarization_min_segment: float = 0.5  # Минимальная длительность сегмента спикера (сек)
    diarization_merge_gap: float = 0.3    # Максимальный зазор для объединения смежных сегментов одного спикера

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )
