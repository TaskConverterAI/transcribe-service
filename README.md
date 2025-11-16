# transcribe-service

## Описание
Сервис для транскрибации аудио и аналитики встреч на базе моделей Vosk (распознавание речи), Pyannote (диаризация спикеров) и LLM (саммари/задачи). Предоставляет REST API:
- `/transcribe` — принимает путь до аудиофайла и возвращает детализированную транскрипцию с сегментами и спикерами.
- `/analyze` — анализирует текст встречи (например, сформированный из транскрипта) и выдаёт:
  1. Саммари в нескольких абзацах
  2. Список извлечённых задач (title, description, assignee, source_line)

## Как собрать

1. Установка
```bash
git clone https://github.com/your-username/transcribe-service.git
cd transcribe-service
pip install -r requirements.txt
```

2. Настройка модели VOSK
```bash
# Распаковать архив с моделью (если еще не распакован)
cd models
unzip vosk-model-ru-0.42.zip

# Модель должна находиться по пути: models/vosk-model-ru-0.42/
# Путь настраивается в config.py через переменную transcribe_model_path
```

3. Настройка устройства для обработки
   
   **Для ускорения обработки при наличии GPU:**
   - В файле `config.py` измените параметр `device` с `"cpu"` на `"gpu"`
   ```python
   device: str = "gpu"  # Вместо "cpu" для ускорения на GPU
   ```
   
   **Примечание:** По умолчанию используется CPU. GPU рекомендуется для больших объемов данных и значительно ускоряет процесс транскрибации.

4. Настройка Ollama Cloud

   a) Установка клиента:
   - Скачайте и установите клиент Ollama с https://ollama.com/download
   
   b) Получение API ключа:
   - Зарегистрируйтесь на https://ollama.com
   - Перейдите на https://ollama.com/settings/keys
   - Создайте новый API ключ и сохраните его
   
   c) Настройка:
   ```bash
   # Авторизация в Ollama Cloud
   ollama signin
   
   # Запуск модели
   ollama run gpt-oss:120b-cloud
   ```
   
   d) Конфигурация:
   - Добавьте ваш API ключ в переменную окружения `OLLAMA_API_KEY` или в config.py
   - По умолчанию используется модель `gpt-oss:120b-cloud`
