# Audio to speech transcriber

### Build image
```
docker build -t audio-transcriber .
```
### Run container
```
docker run -d --name audio-transcriber --env-file /home/user/transcriber_app/.env --gpus all -p 8565:8565 -v /mnt/meeting_r
ecords/audio:/records audio-transcriber
```

### Use app
```
curl http://0.0.0.0:8565/get_transcription?audio_name=2025-01-19_22-54-48.mp3
```
```
{
  "status": "OK",
  "statusfile": "2025-01-19_22-54-48.mp3.lock"
}
```

then get transcription text
```
curl http://0.0.0.0:8565/get_status?statusfile=2025-01-19_22-54-48.mp3.lock
```

```
{
  "status": "OK",
  "transcription_text": "<text here>"
}
```

### Size of image layers:
|CMD|SIZE|
|---|---|
|RUN /bin/sh -c apt-get update &&    apt install -y cuda-toolkit|7.82GB|
|RUN /bin/sh -c pip install -r requirements.txt|9.47GB|
|RUN /bin/sh -c pip install nvidia-pyindex && pip install nvidia-cudnn |1.46GB|
|RUN /bin/sh -c python app_transcriber.py load|5.61GB |

Overall: 25Gb