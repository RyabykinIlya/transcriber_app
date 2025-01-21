#!/usr/bin/python
# -*- coding: utf-8 -*-

import whisperx
import gc
import torch
import threading
from pydub import AudioSegment
from pydub.utils import make_chunks
from os import mkdir, path, remove, environ, mknod
import shutil
from flask import Flask, request, Response, jsonify
import sys
import logging

# from natsort import natsorted

APP_PORT = environ.get("APP_PORT")
APP_DEBUG = environ.get("APP_DEBUG", False)
HF_TOKEN = environ.get("HF_TOKEN")
MEDIA_FOLDER = environ.get("MEDIA_FOLDER")
lock_extender = ".lock"
transcripted_extender = "_transcription.txt"
chunks_extender = "_chunks"

model_config = {
    "language": "ru",
    "batch_size": int(environ.get("BATCH_SIZE", 16)),
    "compute_type": environ.get("COMPUTE_TYPE", "float16"),
    "device": "cuda",
}

app = Flask(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(environ.get("LOGGING_LEVEL", "INFO"))


def create_chunks(chunk_length_ms, chunks_folder, source_filename):
    # Prepare file to not overload CUDA
    chunk_names = []
    myaudio = AudioSegment.from_file(path.join(MEDIA_FOLDER, source_filename))
    chunks = make_chunks(myaudio, chunk_length_ms)

    audio_name, ext = path.splitext(source_filename)

    try:
        shutil.rmtree(chunks_folder)
    except FileNotFoundError:
        pass

    try:
        mkdir(chunks_folder)
    except FileExistsError:
        pass

    audio_extention = "wav"
    logger.debug(f"{chunks_folder=}")
    for i, chunk in enumerate(chunks):
        chunk_name = "{0}_chunk{1}.{2}".format(audio_name, i, audio_extention)
        logger.debug(f"exporting {chunk_name}")
        chunk.export(path.join(chunks_folder, chunk_name), format=audio_extention)
        chunk_names.append(chunk_name)

    # if chunks was already created
    # chunk_names = natsorted(listdir(chunks_folder))

    return chunk_names


def process_audio(audio_file, speakers_from_to: tuple):
    language_code, batch_size, compute_type, device = model_config.values()
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(
        "large-v2", device, language=language_code, compute_type=compute_type
    )

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, language=language_code, batch_size=batch_size)
    # print(result["segments"])  # before alignment
    # print("lang: ", result["language"])

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(
        language_code=language_code, device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # print(result["segments"])  # after alignment

    if speakers_from_to:
        # 3. Assign speaker labels
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=HF_TOKEN, device=device
        )

        # add min/max number of speakers if known
        # diarize_segments = diarize_model(audio)
        min_speakers = speakers_from_to[0]
        max_speakers = speakers_from_to[1]
        diarize_segments = diarize_model(
            audio, min_speakers=min_speakers, max_speakers=max_speakers
        )

        result = whisperx.assign_word_speakers(diarize_segments, result)

    # delete model if low on GPU resources
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


def transcribe(source_filename, speakers_count=None):
    # audio_name = "deepai2611"
    # audio_extention = "wav"
    # source_file = audio_name + "." + audio_extention
    audio_name, ext = path.splitext(source_filename)
    chunks_folder = path.join(MEDIA_FOLDER, audio_name + chunks_extender)
    transcription_file = path.join(MEDIA_FOLDER, audio_name + transcripted_extender)

    chunk_names = create_chunks(
        chunk_length_ms=int(environ.get("CHUNK_LENGTH_MS", 200000)),
        chunks_folder=chunks_folder,
        source_filename=source_filename,
    )
    logger.debug(f'JOBS TODAY: {", ".join(chunk_names)}')

    try:
        remove(transcription_file)
    except FileNotFoundError:
        pass

    if speakers_count:
        speakers_from_to = (1, speakers_count or 3)
    else:
        speakers_from_to = ()

    for chunk_name in chunk_names:
        audio_file = path.join(chunks_folder, chunk_name)
        logger.debug(f">> processing {audio_file}")
        result = process_audio(audio_file, speakers_from_to=speakers_from_to)

        # print("diarize_segments: ", diarize_segments)
        with open(transcription_file, "a+") as f:
            for segment in result["segments"]:
                # print("SEGMENT: ", segment, "\r\n")
                if "speaker" in segment:
                    f.write("\r\n" + segment["speaker"] + ": " + segment["text"])
                else:
                    f.write("\r\n" + segment["text"])

    remove(path.join(MEDIA_FOLDER, source_filename) + lock_extender)


@app.route("/", methods=["GET"])
def root_page():
    return Response("OK")


@app.route("/get_status", methods=["GET"])
def get_status():
    status_filename = request.values.get("statusfile", None)
    if path.exists(path.join(MEDIA_FOLDER, status_filename)):
        return jsonify({"status": "IN PROGRESS"})
    else:
        filename, ext = path.splitext(status_filename.replace(lock_extender, ""))
        try:
            with open(
                path.join(MEDIA_FOLDER, filename + transcripted_extender), "r"
            ) as f:
                transcription_text = f.read()

            # remove artifacts
            shutil.rmtree(path.join(MEDIA_FOLDER, filename + chunks_extender))
            remove(path.join(MEDIA_FOLDER, filename + transcripted_extender))

            return jsonify(
                {"status": "OK", "transcription_text": transcription_text.strip()}
            )
        except FileNotFoundError:
            return jsonify({"status": "ERROR", "message": "Transcription not found"})


@app.route("/get_transcription", methods=["GET"])
def transcribe_from_mp3():
    audio_name = request.values.get("audio_name", None)

    if not audio_name:
        return jsonify({"status": "ERROR", "message": "No audio name provided"})

    filepath = path.join(MEDIA_FOLDER, audio_name)
    if not path.exists(filepath):
        return jsonify(
            {"status": "ERROR", "message": f"File {filepath} not found on host"}
        )

    # speakers_count = request.values.get("speakers_count", None)
    transcribe_thread = threading.Thread(
        target=transcribe,
        name=f"Transcriber {audio_name}",
        args=(audio_name, None),
    )
    transcribe_thread.start()
    statusfile = audio_name + lock_extender
    mknod(path.join(MEDIA_FOLDER, statusfile))
    # transcription_text = transcribe(audio_path, speakers_count)
    return jsonify({"status": "OK", "statusfile": statusfile})


def main():
    app.run(debug=bool(APP_DEBUG), host="::", port=APP_PORT)


if __name__ == "__main__":
    try:
        # pass someth to load models
        sys.argv[1]
        language_code, batch_size, compute_type, device = model_config.values()
        try:
            whisperx.load_model(
                "large-v2", device, language=language_code, compute_type=compute_type
            )
        except RuntimeError:
            pass

        try:
            whisperx.load_align_model(language_code=language_code, device=device)
        except RuntimeError:
            pass
    except IndexError:
        main()
