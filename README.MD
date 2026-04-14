# Audio to speech transcriber

HTTP-сервис поверх WhisperX для транскрибации аудио. Работает с файлами, лежащими в смонтированной `MEDIA_FOLDER` (см. `-v` при запуске контейнера).

## Build image
```
docker build -t audio-transcriber app -f Dockerfile
```

## Run container
```
docker run -d --name audio-transcriber \
  --env-file /home/user/transcriber_app/.env \
  --gpus all \
  -p 8565:8565 \
  -v /mnt/meeting_records/audio:/records \
  audio-transcriber
```

### Переменные окружения
| Имя | Назначение | Default |
|---|---|---|
| `APP_PORT` | Порт Flask | `3000` |
| `APP_DEBUG` | Flask debug | `False` |
| `MEDIA_FOLDER` | Папка с аудио и артефактами (`.lock`, `_transcription.txt`, `_chunks/`, `_error.log`) | — |
| `CHUNK_LENGTH_MS` | Длина чанка при нарезке | `120000` |
| `CHUNK_OVERLAP_MS` | Перекрытие соседних чанков (нужно, чтобы слова на границе не обрезались) | `3000` |
| `BATCH_SIZE` | Whisper batch size | `8` |
| `COMPUTE_TYPE` | Whisper compute type | `float16` |
| `HF_TOKEN` | Токен HuggingFace (нужен для диаризации) | — |
| `LOGGING_LEVEL` | Уровень логов | `DEBUG` |

---

## API

Все аудио-файлы хранятся в `MEDIA_FOLDER`. Ниже `{audio_name}` — имя файла внутри этой папки (например, `2025-01-19_22-54-48.mp3`).

Пайплайн один и тот же для всех способов запуска:
1. создаётся `.lock`-файл — признак «в работе»;
2. исходник режется на чанки (`CHUNK_LENGTH_MS`) в подпапку `{stem}_chunks/` с перекрытием `CHUNK_OVERLAP_MS` — каждый следующий чанк начинается на `overlap` мс раньше конца предыдущего, чтобы слова на границе не обрезались. Дубли из зоны overlap отбрасываются при сборке результата (у каждого чанка, кроме последнего, обрезаются сегменты, попадающие в хвостовой overlap — их «чистую» версию выдаёт следующий чанк с полным контекстом впереди);
3. каждый чанк прогоняется через Whisper + align (опционально диаризация);
4. результат пишется в `{stem}_transcription.txt`;
5. `.lock` снимается по завершении; при ошибке пишется `{stem}_error.log`.

Параллельные транскрибации сериализуются внутренним локом (OOM protection).

### `GET /`
Health check. Возвращает `OK`.

### `GET /get_transcription`
Запускает транскрибацию файла, **который уже лежит** в `MEDIA_FOLDER`.

Параметры (query или form):
- `audio_name` — имя файла в `MEDIA_FOLDER`.

Ответы:
```
200 {"status": "OK", "statusfile": "{audio_name}.lock"}
200 {"status": "IN_PROGRESS", "message": "..."}   // уже запущен
200 {"status": "ERROR", "message": "File ... not found on host"}
```

Пример:
```
curl "http://0.0.0.0:8565/get_transcription?audio_name=2025-01-19_22-54-48.mp3"
```

### `POST /upload_and_transcribe`
Синхронный эндпоинт для внешних сервисов: принимает файл multipart-ом, прогоняет его через тот же пайплайн (chunk → Whisper → align) и возвращает текст прямо в ответе. Никакие артефакты в `MEDIA_FOLDER` не остаются — файл и чанки живут во временной директории и удаляются по завершении запроса.

Поведение при занятости: если параллельно уже идёт другая транскрибация, запрос **не обрывается** — сервис ждёт освобождения внутреннего лока до 10 секунд. Если за это время очередь не освободилась, возвращается `503 BUSY` (безопасно повторить позже).

Запрос:
- `Content-Type: multipart/form-data`
- поле `file` — аудио-файл (обязательно)
- поле `speakers_count` — включает диаризацию с `max_speakers=N` (опционально)

Ответы:
```
200 {"status": "OK", "transcription_text": "<текст>"}
400 {"status": "ERROR", "message": "No file provided ..."}
503 {"status": "BUSY", "message": "Another transcription is in progress, try again later"}
500 {"status": "ERROR", "message": "<exception>"}
```

Пример:
```
curl -X POST http://0.0.0.0:8565/upload_and_transcribe \
  -F "file=@./meeting.mp3"
```

Замечание: транскрибация выполняется в рамках HTTP-запроса целиком, поэтому на клиенте нужно выставить достаточный read timeout (обычно несколько минут на час записи).

### `GET /get_status`
Опрос статуса ранее запущенной транскрибации.

Параметры:
- `statusfile` — значение из ответа `/get_transcription` (`{audio_name}.lock`).

Ответы:
```
200 {"status": "IN PROGRESS"}
200 {"status": "OK", "transcription_text": "<текст>"}
200 {"status": "ERROR", "message": "..."}
```

Важно: при статусе `OK` сервис удаляет `_transcription.txt` и `_chunks/` — текст нужно сохранить у вызывающей стороны после первого успешного ответа.

Пример:
```
curl "http://0.0.0.0:8565/get_status?statusfile=2025-01-19_22-54-48.mp3.lock"
```

---

## Типовой сценарий интеграции из внешнего сервиса

1. `POST /upload_and_transcribe` с файлом → получить `statusfile`.
2. Периодически `GET /get_status?statusfile=...` (например, раз в 10–30 с).
3. При `status=OK` — забрать `transcription_text` и сохранить у себя.
4. При `status=ERROR` — залогировать `message`.

---

## Size of image layers
| CMD | SIZE |
|---|---|
| `apt install -y cuda-toolkit` | 7.82 GB |
| `pip install -r requirements.txt` | 9.47 GB |
| `pip install nvidia-pyindex && pip install nvidia-cudnn` | 1.46 GB |
| `python app_transcriber.py load` (preload моделей) | 5.61 GB |

Overall: ~25 GB
