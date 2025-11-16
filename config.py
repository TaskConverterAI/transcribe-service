import os
import logging
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    device: str = "cpu"

    transcribe_model_path: str = os.getenv("VOSK_MODEL_PATH", "models/vosk-model-ru-0.42")
    transcribe_model_url: str = os.getenv("VOSK_MODEL_URL", "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip")

    # Диаризация (улучшенная)
    enable_speaker_recognition: bool = True
    speaker_auto_detect: bool = True  # Автоопределение числа спикеров
    speaker_min: int = 2              # Минимум спикеров при автоопределении
    speaker_max: int = 8              # Максимум спикеров при автоопределении
    speaker_force_num: int = 0        # Если >0, принудительно число спикеров
    diarization_min_segment: float = 0.5
    diarization_merge_gap: float = 0.3
    speaker_min_duration: float = 0.4  # Минимальная длина сегмента речи для учёта

    # LLM
    tasks_model: str = os.getenv("TASKS_MODEL", "gpt-oss:120b-cloud")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "1c52a28399814d52a8e1961b5d71e438.MFK5Y1VlK2cRGIiEw_vnijQ4")


    max_file_size: int = 500 * 1024 * 1024
    supported_formats: List[str] = [
        "audio/wav", "audio/mp3", "audio/m4a", "audio/ogg",
        "audio/flac", "audio/aac", "video/mp4"
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
