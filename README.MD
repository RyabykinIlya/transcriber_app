# Audio to speech transcriber

Build image
```
docker build -t audio-transcriber .
```
Run container
```
docker run --name audio-transcriber -p 8565:8565 -v /myrecords_path:/mnt/myrecords/ audio-transcriber
```

Use app
```
curl http://0.0.0.0:8565/get_transcription?audio_path=/mnt/myrecords/2025-01-19_22-54-48.mp3
```
```
{
  "status": "OK",
  "statusfile": "/mnt/myrecords/2025-01-19_22-54-48.mp3.lock"
}
```

then get transcription text
```
curl http://0.0.0.0:8565/get_status?statusfile=/mnt/myrecords/2025-01-19_22-54-48.mp3.lock
```

```
{
  "status": "OK",
  "transcription_text": "<text here>"
}
```
