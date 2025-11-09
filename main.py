import os
import asyncio
from typing import Dict, List, Any, Optional
import tempfile
import time
import logging
from contextlib import asynccontextmanager
import json
import zipfile
import urllib.request
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

import torch
import librosa
import soundfile as sf
import vosk
import webrtcvad
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import get_settings, setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cpu" if torch.cuda.is_available() else "cpu"))


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
    description="API для транскрибации а��дио с диаризацией спикеров",
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
    transcribe_backend: str = Field(..., description="InferenceClient репозиторий")
    diarization_model: str = Field(..., description="Модель диаризации")
    device: str = Field(..., description="Устройство диаризации (CPU/GPU)")


class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Полный транскрибированный текст")
    speakers: List[SpeakerModel] = Field(..., description="Информация о спикерах")
    segments: List[SegmentModel] = Field(..., description="Сегменты с привязкой к спикерам")
    processing_time: float = Field(..., description="Общее время обработки в секундах")
    duration: float = Field(..., description="Длительность аудио в секундах")
    language: str = Field(..., description="Определенный язык")
    ai_info: ModelInfoModel = Field(..., description="Информация о моделях и производительности")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Статус сервиса")
    timestamp: float = Field(..., description="Временная метка проверки")
    models_loaded: Dict[str, bool] = Field(..., description="Статус загрузки моделей")
    system_info: Optional[Dict[str, Any]] = Field(None, description="Системная информация")


# --- Модели для анализа встречи ---
class TaskModel(BaseModel):
    title: str = Field(..., description="Короткое название задачи")
    description: str = Field(..., description="Полное описание / фраза из диалога")
    assignee: str = Field("UNASSIGNED", description="Ответственный, если определен")
    source_line: int = Field(..., description="Номер строки в исходном тексте")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Полный текст встречи. Формат: строки вида 'SPK1: текст' или произвольный текст")
    language: Optional[str] = Field(None, description="Язык текста (по умолчанию авто)")
    max_tasks: int = Field(20, description="Максимальное количество задач в ответе")
    use_llm: bool = Field(True, description="Использовать LLM для саммари и извлечения задач")


class AnalyzeResponse(BaseModel):
    summary: str = Field(..., description="Сводное описание встречи")
    summary_paragraphs: List[str] = Field(..., description="Абзацы саммари")
    tasks: List[TaskModel] = Field(..., description="Извлеченные задачи")
    participants: List[str] = Field(..., description="Список идентификаторов участников")
    total_lines: int = Field(..., description="Количество обработанных строк")


class TranscriptionService:
    def __init__(self):
        self.transcribe_model = None
        self.llm_model = None
        self.device = self._determine_device()
        self.model_loading_lock = asyncio.Lock()
        logger.info(f"Инициализация сервиса. Устройство: {self.device}")

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

    @staticmethod
    def _determine_device() -> str:
        if settings.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return settings.device

    @staticmethod
    async def _download_transcribe_model():
        """Загружает модель, если она не существует"""
        if os.path.exists(settings.transcribe_model_path):
            return

        logger.info(f"Загрузка модели транскрибации из {settings.transcribe_model_url}")
        os.makedirs(os.path.dirname(settings.transcribe_model_path), exist_ok=True)

        # Загружаем и распаковываем модель
        zip_path = f"{settings.transcribe_model_path}.zip"
        urllib.request.urlretrieve(settings.transcribe_model_url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(settings.transcribe_model_path))

        # Переименовываем папку в нужное имя
        extracted_folder = None
        for item in os.listdir(os.path.dirname(settings.transcribe_model_path)):
            if item.startswith("vosk-model"):
                extracted_folder = os.path.join(os.path.dirname(settings.transcribe_model_path), item)
                break

        if extracted_folder and extracted_folder != settings.transcribe_model_path:
            os.rename(extracted_folder, settings.transcribe_model_path)

        os.unlink(zip_path)
        logger.info("Модель транскрибации успешно загружена")


    def _load_ollama_client(self):
        """Инициализирует Ollama client для задач и саммари."""
        if self.llm_model is not None:
            return
        model_name = settings.tasks_model
        if not model_name:
            logger.warning("tasks_model не задан — LLM клиент не будет создан.")
            return
        try:
            try:
                from ollama import Client  # type: ignore
            except ImportError:
                logger.error("Библиотека ollama не установлена. Добавьте 'ollama' в requirements.txt")
                return
            headers = {}
            if settings.ollama_api_key:
                headers['Authorization'] = 'Bearer ' + settings.ollama_api_key
            host = settings.ollama_base_url.rstrip('/')
            self.llm_model = Client(host=host, headers=headers if headers else None)
            logger.info(f"Ollama клиент инициализирован: host={host}, model={model_name}")

        except Exception as e:
            logger.error(f"Не удалось создать LLM клиент: {e}")
            self.llm_model = None

    async def load_models(self):
        async with self.model_loading_lock:
            if self.transcribe_model is not None:
                return

            try:
                logger.info("Начало загрузки модели транскрибации...")
                await self._download_transcribe_model()

                if not os.path.exists(settings.transcribe_model_path):
                    raise ValueError(f"Модель для транскрибации не найдена по пути: {settings.transcribe_model_path}")

                self.transcribe_model = vosk.Model(settings.transcribe_model_path)
                logger.info("Модель для транскрибации загружена успешно")

            except Exception as e:
                logger.error(f"Ошибка загрузки модели транскрибации: {e}")
                self.transcribe_model = None
                raise e

    @staticmethod
    def _segments_from_text_with_diarization(full_text: str, diarization: Any) -> List[SegmentModel]:
        # Простая нарезка текста по диаризации: каждая метка спикера получает свой кусок (делим равномерно)
        if not full_text:
            # создаём сегменты без текста
            blank = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                blank.append(
                    SegmentModel(start=segment.start, end=segment.end, text='', speaker=speaker, confidence=0.0))
            return blank
        words = full_text.split()
        total_words = len(words)
        diar_segments = list(diarization.itertracks(yield_label=True))
        if not diar_segments:
            return [SegmentModel(start=0.0, end=max(len(full_text) / 10.0, 0.1), text=full_text, speaker='SPK1',
                                 confidence=0.0)]
        alloc = []
        # Распределяем слова пропорционально длительности сегмента
        total_duration = sum((seg.end - seg.start) for seg, _, _ in diar_segments)
        used = 0
        for i, (seg, _, speaker) in enumerate(diar_segments):
            duration = seg.end - seg.start
            if total_duration <= 0:
                share = total_words // len(diar_segments)
            else:
                share = int(round((duration / total_duration) * total_words))
            if i == len(diar_segments) - 1:
                share = total_words - used
            segment_words = words[used: used + share]
            used += share
            alloc.append(SegmentModel(
                start=seg.start,
                end=seg.end,
                text=' '.join(segment_words).strip(),
                speaker=speaker,
                confidence=0.0
            ))
        return alloc

    def _voice_activity_detection(self, audio_data: bytes, sample_rate: int = 16000) -> List[tuple]:
        """Детекция голосовой активности с помощью WebRTC VAD"""
        vad = webrtcvad.Vad()
        vad.set_mode(3)  # Самый агрессивный режим

        # Размер фрейма для VAD (10, 20 или 30 мс)
        frame_duration = 30  # мс
        frame_size = int(sample_rate * frame_duration / 1000)

        speech_segments = []
        current_segment_start = None

        for i in range(0, len(audio_data), frame_size * 2):  # *2 для 16-bit audio
            frame = audio_data[i:i + frame_size * 2]
            if len(frame) < frame_size * 2:
                break

            timestamp = i / (sample_rate * 2)

            try:
                is_speech = vad.is_speech(frame, sample_rate)

                if is_speech and current_segment_start is None:
                    current_segment_start = timestamp
                elif not is_speech and current_segment_start is not None:
                    speech_segments.append((current_segment_start, timestamp))
                    current_segment_start = None
            except:
                # Если VAD не работает с этим фреймом, считаем его речью
                if current_segment_start is None:
                    current_segment_start = timestamp

        # Закрываем последний сегмент если он остался открытым
        if current_segment_start is not None:
            speech_segments.append((current_segment_start, len(audio_data) / (sample_rate * 2)))

        return speech_segments

    async def _transcribe_async(self, audio_path: str) -> Dict[str, Any]:
        """Транскрибация одним блоком (без почанковой разбивки)."""
        start_time = time.time()
        if not self.transcribe_model:
            await self.load_models()
        loop = asyncio.get_event_loop()

        def _vosk_transcribe():
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)
            audio_bytes = (audio_data * 32767).astype('int16').tobytes()

            recognizer = vosk.KaldiRecognizer(self.transcribe_model, 16000)

            results = []
            frame_size = 4000  # размер чанка для обработки

            # Обрабатываем аудио чанками
            for i in range(0, len(audio_bytes), frame_size):
                chunk = audio_bytes[i:i + frame_size]

                if recognizer.AcceptWaveform(chunk):
                    result = json.loads(recognizer.Result())
                    if result.get('text'):
                        # Добавляем временные метки
                        timestamp = i / (16000 * 2)  # 2 bytes per sample for 16-bit
                        result['start'] = timestamp
                        result['end'] = timestamp + len(chunk) / (16000 * 2)
                        results.append(result)
            """
            # Обрабатываем последний чанк
            final_result = json.loads(recognizer.FinalResult())
            if final_result.get('text'):
                results.append(final_result)
            """
            return results

        vosk_results = await loop.run_in_executor(None, _vosk_transcribe)
        processing_time = time.time() - start_time

        full_text = ' '.join([r.get('text', '') for r in vosk_results if r.get('text')])

        return {
            'result': full_text,
            'time': processing_time,
            'segments': vosk_results
        }

    async def process_audio_file(self, file: UploadFile) -> TranscriptionResponse:
        start_time = time.time()
        temp_files: List[str] = []
        try:
            if not self.transcribe_model:
                await self.load_models()
            self.validate_audio_file(file)
            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix='.input')
            temp_files.append(tmp_in.name)
            content = await file.read()
            tmp_in.write(content)
            tmp_in.close()
            processed = self._preprocess_audio(tmp_in.name)
            temp_files.append(processed)
            duration = librosa.get_duration(path=processed)

            transcribe_result = await self._transcribe_async(processed)

            segments = transcribe_result['segments']
            print(segments)
            # Create embedding
            def segment_embedding(segment):
                audio = Audio()
                start = segment["start"]
                print(start)
                # Whisper overshoots the end timestamp in the last segment
                end = min(duration, segment["end"])
                clip = Segment(start, end)
                waveform, sample_rate = audio.crop(processed, clip)
                return embedding_model(waveform[None])

            embeddings = np.zeros(shape=(len(segments), 192))
            for i, segment in enumerate(segments):
                embeddings[i] = segment_embedding(segment)
            embeddings = np.nan_to_num(embeddings)
            print(f'Embedding shape: {embeddings.shape}')

            # Find the best number of speakers
            score_num_speakers = {}

            for num_speakers in range(2, 10 + 1):
                clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
                score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
                score_num_speakers[num_speakers] = score
            best_num_speaker = max(score_num_speakers, key=lambda x: score_num_speakers[x])
            print(
                f"The best number of speakers: {best_num_speaker} with {score_num_speakers[best_num_speaker]} score")

            # Assign speaker label - создаём новые словари с назначенными спикерами
            clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
            labels = clustering.labels_

            # Создаём новый список сегментов с назначенными спикерами
            segments_with_speakers = []
            for i, segment in enumerate(segments):
                segment_with_speaker = {
                    'start': segment.get('start', 0.0),
                    'end': segment.get('end', 0.0),
                    'text': segment.get('text', ''),
                    'speaker': 'SPEAKER ' + str(labels[i] + 1),
                    'confidence': 0.0
                }
                segments_with_speakers.append(segment_with_speaker)

            # Преобразуем словари в объекты SegmentModel
            segment_models = []
            for seg_dict in segments_with_speakers:
                segment_models.append(SegmentModel(
                    start=seg_dict['start'],
                    end=seg_dict['end'],
                    text=seg_dict['text'],
                    speaker=seg_dict['speaker'],
                    confidence=seg_dict['confidence']
                ))

            speakers_summary = self._get_speakers_summary(segment_models)
            total_time = time.time() - start_time
            resp = TranscriptionResponse(
                text=transcribe_result['result'] or '',
                speakers=speakers_summary,
                segments=segment_models,  # Используем преобразованные объекты
                processing_time=round(total_time, 2),
                duration=round(duration, 2),
                language='ru',
                ai_info=ModelInfoModel(
                    transcribe_backend='Vosk',
                    diarization_model='VAD + MFCC clustering',
                    device=self.device
                )
            )
            logger.info(f"Транскрипция завершена за {total_time:.2f}s")
            return resp
        except Exception as e:
            logger.error(f"Ошибка обработки аудио: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка обработки аудио: {e}")
        finally:
            for f in temp_files:
                try:
                    if os.path.exists(f):
                        os.unlink(f)
                except Exception:
                    pass

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

    @staticmethod
    def _strip_llm_think(text: str) -> str:
        """Удаляет внутреннее рассуждение (think) оставляя финальный вывод."""
        if not text:
            return text
        import re
        text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
        cleaned_lines: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r'^(thought|thinking|reasoning|analysis)[:\-]', stripped, flags=re.IGNORECASE):
                continue
            if stripped.lower().startswith('internal reasoning'):
                continue
            if stripped.lower().startswith('final answer:'):
                stripped = stripped[len('final answer:'):].strip()
            cleaned_lines.append(stripped)
        if cleaned_lines and re.match(r'^(answer|ответ)[:\-]', cleaned_lines[0], flags=re.IGNORECASE):
            cleaned_lines[0] = re.sub(r'^(answer|ответ)[:\-]\s*', '', cleaned_lines[0], flags=re.IGNORECASE)
        return '\n'.join(cleaned_lines).strip()

    def _llm_chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        self._load_ollama_client()
        if self.llm_model is None:
            return ''
        model_name = model or settings.tasks_model
        try:
            resp = self.llm_model.chat(model_name, messages=messages, format="json")
            return resp.message.content
        except Exception as e:
            logger.error(f"Ошибка вызова LLM: {e}")
            return ''

    def _llm_extract_tasks(self, lines: List[str], max_tasks: int) -> List[TaskModel]:
        raw_lines = lines
        self._load_ollama_client()
        if self.llm_model is None:
            logger.info("LLM модель не задана.")
            return []

        numbered = [f"{i + 1}. {l}" for i, l in enumerate(raw_lines)]
        convo = "\n".join(numbered)
        system_prompt = (
            "Ты извлекаешь реальные задачи из диалога. Формат: TASK|source_line|title|assignee|description. Только такие строки." \
            " title ≤6 слов; assignee из строки или UNASSIGNED; description заканчивается оригинальной фразой. Игнорируй факты и вопросы.")
        user_prompt = f"Диалог:\n{convo}\n\nИзвлеки до {max_tasks} задач:"
        raw = self._llm_chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], model=settings.tasks_model)

        tasks: Dict[tuple, TaskModel] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("TASK|"):
                continue
            parts = line.split('|')
            if len(parts) < 5:
                continue
            _, source_line, title, assignee, description = parts[:5]
            try:
                sl = int(source_line)
            except ValueError:
                sl = 0
            key = (sl, title.strip())
            if key in tasks:
                continue
            tasks[key] = TaskModel(
                title=(title.strip() or 'Задача')[:100],
                description=description.strip(),
                assignee=assignee.strip() or 'UNASSIGNED',
                source_line=sl
            )
            if len(tasks) >= max_tasks:
                break

        return sorted(tasks.values(), key=lambda t: (t.source_line, t.title))[:max_tasks]

    def _llm_summarize(self, speaker_text_map: Dict[str, List[str]]) -> List[str]:
        joined = []
        for spk, txts in speaker_text_map.items():
            snippet = ' '.join(txts)
            if len(snippet) > 5000:
                snippet = snippet[:5000].rsplit(' ', 1)[0] + '…'
            joined.append(f"{spk}: {snippet}")
        convo = "\n".join(joined)

        system_prompt = (
            "Ты аналитик встречи. Верни структурированное саммари: [Общее резюме] абзац; [Ключевые решения] буллеты; [Основные темы] буллеты. Без выдумок.")
        user_prompt = f"Диалог:\n{convo}\n\nСформируй саммари."
        raw = self._llm_chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], model=settings.tasks_model)

        print(raw)
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return lines[:20]

    def analyze_text(self, req: AnalyzeRequest) -> AnalyzeResponse:
        raw_text = req.text.strip()
        if not raw_text:
            raise HTTPException(status_code=400, detail="Пустой текст для анализа")

        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        speaker_text_map: Dict[str, List[str]] = {}
        for line in lines:
            if ':' in line[:30]:
                prefix, content = line.split(':', 1)
                speaker = prefix.strip()
                text_part = content.strip()
            else:
                speaker = 'UNKNOWN'
                text_part = line
            speaker_text_map.setdefault(speaker, []).append(text_part)
        participants = list(speaker_text_map.keys())

        summary_paragraphs = self._llm_summarize(speaker_text_map)
        tasks = self._llm_extract_tasks(lines, max_tasks=req.max_tasks)

        summary = '\n'.join(summary_paragraphs)
        return AnalyzeResponse(
            summary=summary,
            summary_paragraphs=summary_paragraphs,
            tasks=tasks,
            participants=participants,
            total_lines=len(lines)
        )


transcription_service = TranscriptionService()


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "TranscribeAPI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "analyze": "/analyze"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    models_loaded = {
        "transcribe_model": transcription_service.transcribe_model is not None,
        "voice_activity_detection": True
    }

    status_value = "healthy" if all(models_loaded.values()) else "unhealthy"

    system_info = {
        "device": transcription_service.device,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "transcribe_model_path": settings.transcribe_model_path,
        "model_exists": os.path.exists(settings.transcribe_model_path)
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


@app.post("/analyze", response_model=AnalyzeResponse, summary="Анализ текста встречи")
async def analyze_meeting(req: AnalyzeRequest):
    try:
        return transcription_service.analyze_text(req)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ошибка анализа текста")
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {e}")


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
