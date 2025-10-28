import os
import asyncio
from typing import Dict, List, Any, Optional
import tempfile
import time
import logging
from contextlib import asynccontextmanager
import functools

import torch
import whisper
import librosa
import soundfile as sf
from pyannote.audio import Pipeline
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import get_settings, setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск TranscribeAPI...")
    try:
        await transcription_service.load_models()
        logger.info("Сервис готов к работе")
    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")

    yield

    logger.info("Завершение работы сервиса")


app = FastAPI(
    title="TranscribeAPI",
    description="API для транскрибации аудио с диаризацией спикеров",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SegmentModel(BaseModel):
    start: float = Field(..., description="Время начала сегмента в секундах")
    end: float = Field(..., description="Время окончания сегмента в секундах")
    text: str = Field(..., description="Текст сегмента")
    speaker: str = Field(..., description="Идентификатор спикера")
    confidence: float = Field(..., description="Уровень уверенности")


class SpeakerModel(BaseModel):
    speaker_id: str = Field(..., description="Идентификатор спикера")
    total_duration: float = Field(..., description="Общая длительность речи спикера")
    segments_count: int = Field(..., description="Количество сегментов спикера")
    full_text: str = Field(..., description="Полный текст спикера")


class ModelInfoModel(BaseModel):
    whisper_model: str = Field(..., description="Используемая модель Whisper")
    diarization_model: str = Field(..., description="Используемая модель диаризации")
    device: str = Field(..., description="Устройство обработки (CPU/GPU)")
    whisper_processing_time: float = Field(..., description="Время транскрибации")
    diarization_processing_time: float = Field(..., description="Время диаризации")
    alignment_processing_time: float = Field(..., description="Время совмещения")


class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Полный транскрибированный текст")
    speakers: List[SpeakerModel] = Field(..., description="Информация о спикерах")
    segments: List[SegmentModel] = Field(..., description="Сегменты с привязкой к спикерам")
    processing_time: float = Field(..., description="Общее время обработки в секундах")
    audio_duration: float = Field(..., description="Длительность аудио в секундах")
    language: str = Field(..., description="Определенный язык")
    ai_info: ModelInfoModel = Field(..., description="Информация о моделях и производительности")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Статус сервиса")
    timestamp: float = Field(..., description="Временная метка проверки")
    models_loaded: Dict[str, bool] = Field(..., description="Статус загрузки моделей")
    system_info: Optional[Dict[str, Any]] = Field(None, description="Системная информация")


class TranscriptionService:
    def __init__(self):
        self.whisper_model = None
        self.diarization_pipeline = None
        self.device = self._determine_device()
        self.model_loading_lock = asyncio.Lock()
        logger.info(f"Инициализация сервиса. Устройство: {self.device}")

    def _determine_device(self) -> str:
        if settings.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return settings.device

    async def load_models(self):
        async with self.model_loading_lock:
            if self.whisper_model is not None and self.diarization_pipeline is not None:
                return

            try:
                logger.info("Начало загрузки моделей...")

                logger.info(f"Загрузка Whisper модели: {settings.whisper_model}")
                self.whisper_model = whisper.load_model(settings.whisper_model, device=self.device)
                logger.info("Whisper модель загружена")

                logger.info("Загрузка модели диаризации...")
                if not settings.huggingface_token:
                    raise ValueError("HUGGINGFACE_TOKEN не установлен в переменных окружения")

                try:
                    from huggingface_hub import login as hf_login
                    try:
                        hf_login(token=settings.huggingface_token, add_to_git_credential=False)
                        logger.info("Аутентификация через huggingface_hub.login выполнена.")
                    except Exception as auth_err:
                        logger.warning(f"Не удалось выполнить login: {auth_err}. Продолжаем без него.")
                except ImportError:
                    logger.info("Функция login недоступна в установленной версии huggingface_hub.")

                diarization_pipeline = Pipeline.from_pretrained(settings.diarization_model,
                                                                use_auth_token=settings.huggingface_token)

                if diarization_pipeline is None:
                    raise RuntimeError("Не удалось загрузить модель диаризации ни одним доступным способом.")

                self.diarization_pipeline = diarization_pipeline

                if self.device == "cuda" and self.diarization_pipeline is not None:
                    self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))

                logger.info("Все модели загружены")

            except Exception as e:
                logger.error(f"Ошибка загрузки моделей: {e}")
                self.whisper_model = None
                self.diarization_pipeline = None
                raise e

    @staticmethod
    def validate_audio_file(file: UploadFile) -> None:
        if not file.content_type or not any(
                file.content_type.startswith(fmt.split('/')[0]) for fmt in settings.supported_formats
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неподдерживаемый формат файла. Поддерживаемые: {settings.supported_formats}"
            )

        if file.size and file.size > settings.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Файл слишком большой. Максимальный размер: {settings.max_file_size // (1024 * 1024)}MB"
            )

    async def process_audio_file(self, file: UploadFile) -> TranscriptionResponse:
        start_time = time.time()
        temp_files = []

        try:
            if not self.whisper_model or not self.diarization_pipeline:
                await self.load_models()

            self.validate_audio_file(file)

            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.tmp')
            temp_files.append(temp_input.name)

            content = await file.read()
            temp_input.write(content)
            temp_input.close()

            # Предобработка
            processed_audio = self._preprocess_audio(temp_input.name)
            temp_files.append(processed_audio)

            # Получение длительности
            audio_duration = librosa.get_duration(path=processed_audio)
            logger.info(f"Обработка аудио длительностью {audio_duration:.2f}с")

            # Параллельная обработка
            whisper_task = asyncio.create_task(self._transcribe_async(processed_audio))
            diarization_task = asyncio.create_task(self._diarize_async(processed_audio))

            whisper_result, diarization_result = await asyncio.gather(
                whisper_task, diarization_task
            )

            # Совмещение результатов
            alignment_start = time.time()
            segments_with_speakers = self._align_transcription_with_speakers(
                whisper_result['result'], diarization_result['result']
            )
            alignment_time = time.time() - alignment_start

            # Сводка по спикерам
            speakers_summary = self._get_speakers_summary(segments_with_speakers)

            # Формирование ответа
            total_time = time.time() - start_time

            response = TranscriptionResponse(
                text=whisper_result['result']['text'],
                speakers=speakers_summary,
                segments=segments_with_speakers,
                processing_time=round(total_time, 2),
                audio_duration=round(audio_duration, 2),
                language=whisper_result['result'].get('language', 'unknown'),
                ai_info=ModelInfoModel(
                    whisper_model=settings.whisper_model,
                    diarization_model="pyannote/speaker-diarization-3.1",
                    device=self.device,
                    whisper_processing_time=round(whisper_result['time'], 2),
                    diarization_processing_time=round(diarization_result['time'], 2),
                    alignment_processing_time=round(alignment_time, 2)
                )
            )

            logger.info(f"Обработка завершена за {total_time:.2f}с")
            return response

        except Exception as e:
            logger.error(f"Ошибка обработки: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ошибка обработки аудио: {str(e)}"
            )
        finally:
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Не удалось удалить {temp_file}: {e}")

    @staticmethod
    def _preprocess_audio(file_path: str) -> str:
        try:
            audio, sr = librosa.load(file_path, sr=16000)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, audio, 16000)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ошибка предобработки аудио: {str(e)}"
            )

    async def _transcribe_async(self, audio_path: str) -> Dict[str, Any]:
        start_time = time.time()
        loop = asyncio.get_event_loop()
        fn = functools.partial(
            self.whisper_model.transcribe,
            audio_path,
            language='ru',
            task='transcribe',
            word_timestamps=True
        )
        result = await loop.run_in_executor(None, fn)
        processing_time = time.time() - start_time
        return {'result': result, 'time': processing_time}

    async def _diarize_async(self, audio_path: str) -> Dict[str, Any]:
        start_time = time.time()
        loop = asyncio.get_event_loop()
        fn = functools.partial(self.diarization_pipeline, audio_path)
        result = await loop.run_in_executor(None, fn)
        processing_time = time.time() - start_time
        return {'result': result, 'time': processing_time}

    def _align_transcription_with_speakers(self, whisper_result: Dict, diarization: Any) -> List[SegmentModel]:
        segments_with_speakers = []

        for segment in whisper_result['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()

            speaker = self._find_speaker_for_segment(diarization, start_time, end_time)

            segments_with_speakers.append(SegmentModel(
                start=start_time,
                end=end_time,
                text=text,
                speaker=speaker,
                confidence=segment.get('avg_logprob', 0)
            ))

        return segments_with_speakers

    @staticmethod
    def _find_speaker_for_segment(diarization: Any, start_time: float, end_time: float) -> str:
        segment_center = (start_time + end_time) / 2

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.start <= segment_center <= segment.end:
                return speaker

        return "UNKNOWN"

    @staticmethod
    def _get_speakers_summary(segments: List[SegmentModel]) -> List[SpeakerModel]:
        speakers_data = {}

        for segment in segments:
            speaker = segment.speaker
            if speaker not in speakers_data:
                speakers_data[speaker] = {
                    'total_duration': 0,
                    'segments_count': 0,
                    'text_parts': []
                }

            duration = segment.end - segment.start
            speakers_data[speaker]['total_duration'] += duration
            speakers_data[speaker]['segments_count'] += 1
            speakers_data[speaker]['text_parts'].append(segment.text)

        speakers_list = []
        for speaker_id, data in speakers_data.items():
            speakers_list.append(SpeakerModel(
                speaker_id=speaker_id,
                total_duration=round(data['total_duration'], 2),
                segments_count=data['segments_count'],
                full_text=' '.join(data['text_parts'])
            ))

        return speakers_list


transcription_service = TranscriptionService()


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "TranscribeAPI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    models_loaded = {
        "whisper": transcription_service.whisper_model is not None,
        "diarization": transcription_service.diarization_pipeline is not None
    }

    status_value = "healthy" if all(models_loaded.values()) else "unhealthy"

    system_info = {
        "device": transcription_service.device,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__
    }

    if torch.cuda.is_available():
        system_info["gpu_name"] = torch.cuda.get_device_name()
        system_info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)}GB"

    return HealthResponse(
        status=status_value,
        timestamp=time.time(),
        models_loaded=models_loaded,
        system_info=system_info
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(..., description="Аудио файл для транскрибации")):
    return await transcription_service.process_audio_file(file)


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Запуск сервера на {settings.host}:{settings.port}")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )
