# transcribe-service

## Описание
Сервис для транскрибации аудио и аналитики встреч на базе моделей Whisper (распознавание речи), Pyannote (диаризация спикеров) и LLM (саммари/задачи). Предоставляет REST API:
- `/transcribe` — принимает аудиофайл (multipart/form-data) и возвращает детализированную транскрипцию с сегментами и спикерами.
- `/analyze` — анализирует текст встречи (например, сформированный из транскрипта) и выдаёт:
  1. Саммари в нескольких абзацах
  2. Список извлечённых задач (title, description, assignee, source_line)
  3. Участников и служебную статистику

Есть два режима анализа `/analyze`:
- Эвристический (use_llm=false) — быстрый, на ключевых словах.
- LLM (use_llm=true, по умолчанию) — использует HuggingFace модель (например Flan-T5) для генерации саммари и выделения задач.

## Установка
```bash
git clone https://github.com/your-username/transcribe-service.git
cd transcribe-service
pip install -r requirements.txt
```

## Переменные окружения
Создайте `.env` или экспортируйте переменные:

| Переменная | Назначение | Значение по умолчанию |
|-----------|------------|-----------------------|
| `HUGGINGFACE_TOKEN` | Токен доступа к моделям (нужен для pyannote и приватных моделей) | пусто (обязательно указать) |
| `ANALYSIS_MODEL` | HF модель для text2text генерации (саммари/задачи) | `google/flan-t5-base` |
| `ANALYSIS_MAX_NEW_TOKENS` | Лимит новых токенов при генерации | `512` |
| `ANALYSIS_TEMPERATURE` | Температура (если модель поддерживает) | `0.3` |
| `ANALYSIS_TOP_P` | Top-p (если поддерживается) | `0.9` |
| `WHISPER_MODEL` | Модель Whisper (large, medium, small и т.д.) | задаётся в config.py (large) |
| `DEVICE` | cpu или cuda (либо auto в config) | cpu |

Пример `.env`:
```env
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
ANALYSIS_MODEL=google/flan-t5-base
ANALYSIS_MAX_NEW_TOKENS=384
ANALYSIS_TEMPERATURE=0.3
ANALYSIS_TOP_P=0.9
```

## Запуск
```bash
python main.py
```
Сервер по умолчанию слушает на `0.0.0.0:8000`.
Документация Swagger: `http://localhost:8000/docs`

## Эндпоинты
### 1. /transcribe (POST)
Принимает аудио файл. Пример запроса (curl):
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Accept: application/json" \
  -F "file=@test/lol.wav;type=audio/wav"
```
Пример части ответа:
```json
{
  "text": "... полный текст ...",
  "speakers": [
    {"speaker_id": "SPEAKER_00", "total_duration": 12.34, "segments_count": 5, "full_text": "..."}
  ],
  "segments": [
    {"start": 0.0, "end": 3.2, "text": "Пример фразы", "speaker": "SPEAKER_00", "confidence": -0.12}
  ],
  "processing_time": 5.87,
  "audio_duration": 30.12,
  "language": "ru",
  "ai_info": {"whisper_model": "large", "diarization_model": "pyannote/speaker-diarization-3.1", ...}
}
```

### 2. /analyze (POST)
Анализ текста встречи. Тело запроса:
```json
{
  "text": "SPK1: Нужно подготовить отчёт.\nSPK2: Надо обновить API.",
  "max_tasks": 10,
  "use_llm": true
}
```
Запрос:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "SPK1: Нужно подготовить отчёт по продажам.\nSPK2: Надо обновить CRM интеграцию.", "max_tasks": 5, "use_llm": true}'
```
Ответ (пример):
```json
{
  "summary": "SPK1: ...\n\nSPK2: ...",
  "summary_paragraphs": ["SPK1: ...", "SPK2: ...", "Общий обзор: ..."],
  "tasks": [
    {
      "title": "подготовить отчёт по продажам",
      "description": "Нужно подготовить отчёт по продажам.",
      "assignee": "SPK1",
      "source_line": 1
    }
  ],
  "participants": ["SPK1", "SPK2"],
  "total_lines": 2
}
```
Поле `use_llm` (true/false) переключает режим анализа.

### Извлечение задач через DeepSeek (опционально)
Установите переменные окружения:
```env
HF_TOKEN=hf_ваш_токен
TASKS_MODEL=deepseek-ai/DeepSeek-R1
TASKS_PROVIDER=novita
```
При вызове `/analyze` с `use_llm=true` задачи будут извлекаться через удалённую модель DeepSeek. Формат возврата: `TASK|source_line|title|assignee|description` парсится автоматически.
Если модель не вернёт строки формата `TASK|...` будет использован эвристический fallback.

## Формирование текста встречи из транскрипта
Рекомендуемый формат: одна реплика = одна строка, `SPEAKER_ID: текст`. Для получения такого формата из `/transcribe` используйте массив `segments` (там есть `speaker` и `text`).

## Fallback механизмы
- Если LLM недоступна или произошла ошибка генерации: автоматически используется эвристический алгоритм.
- Ограничение длины входного текста: при превышении порогов текст обрезается с добавлением `…`.

## Производительность и рекомендации
- `google/flan-t5-base` умеренно лёгкая модель. Для ускорения можно выбрать меньшую: `google/flan-t5-small`.
- Для больших совещаний ( > 6000 символов) саммари сокращается.
- Whisper `large` требует значительных ресурсов; при ограничениях используйте `medium` или `small`.

## Пример сценария end-to-end
1. Отправить аудио на `/transcribe`.
2. Из ответа собрать строки формата `speaker: text`.
3. Передать их на `/analyze` с `use_llm=true`.
4. Сохранить задачи в вашу систему управления (Jira, ClickUp и т.п.).

## Быстрый запуск с тестовым клиентом
Скрипт `test/test_client.py` автоматизирует:
- multipart запрос на `/transcribe`
- формирование текста встречи
- запрос на `/analyze`
Запуск:
```bash
python test/test_client.py
```

## Ошибки и отладка
- Проверить `/health` для статуса загрузки моделей.
- Логи выводятся в stdout (см. `setup_logging` в `config.py`).
- Убедитесь, что `HUGGINGFACE_TOKEN` действительно имеет доступ к `pyannote/speaker-diarization-3.1`.

## Требования
- Python 3.8+
- Все зависимости из `requirements.txt`
- GPU (опционально) для ускорения Whisper и диаризации

## Лицензия
MIT License
