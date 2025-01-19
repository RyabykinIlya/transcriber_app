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
from threading import Lock
import sys
from natsort import natsorted

APP_PORT = environ.get("APP_PORT")
APP_DEBUG = environ.get("APP_DEBUG", False)
HF_TOKEN = environ.get("HF_TOKEN")
transcripted_extender = "_transcription.txt"

app = Flask(__name__)
lock = Lock()


def create_chunks(chunk_length_ms, chunks_folder, source_filepath):
    # Prepare file to not overload CUDA
    chunk_names = []
    myaudio = AudioSegment.from_file(source_filepath)
    chunks = make_chunks(myaudio, chunk_length_ms)

    audio_filename = path.basename(source_filepath)
    audio_name, ext = path.splitext(audio_filename)

    try:
        shutil.rmtree(chunks_folder)
    except FileNotFoundError:
        pass

    try:
        mkdir(chunks_folder)
    except FileExistsError:
        pass

    audio_extention = "wav"
    print(f"{chunks_folder=}")
    for i, chunk in enumerate(chunks):
        chunk_name = "{0}_chunk{1}.{2}".format(audio_name, i, audio_extention)
        print("exporting ", chunk_name)
        chunk.export(path.join(chunks_folder, chunk_name), format=audio_extention)
        chunk_names.append(chunk_name)

    # if chunks was already created
    # chunk_names = natsorted(listdir(chunks_folder))

    return chunk_names


def process_audio(audio_file, model_config, speakers_from_to: tuple):
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


def transcribe(source_file, speakers_count=None):
    # audio_name = "deepai2611"
    # audio_extention = "wav"
    # source_file = audio_name + "." + audio_extention
    audio_namepath, ext = path.splitext(source_file)
    chunks_folder = audio_namepath + "_chunks"
    transcription_file = audio_namepath + transcripted_extender

    model_config = {
        "language": "ru",
        "batch_size": int(environ.get("BATCH_SIZE", 16)),
        "compute_type": environ.get("COMPUTE_TYPE", "float16"),
        "device": "cuda",
    }

    chunk_names = create_chunks(
        chunk_length_ms=int(environ.get("CHUNK_LENGTH_MS", 200000)),
        chunks_folder=chunks_folder,
        source_filepath=source_file,
    )
    print("JOBS TODAY: ", ", ".join(chunk_names))

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
        print(">> processing", audio_file)
        result = process_audio(
            audio_file, model_config, speakers_from_to=speakers_from_to
        )

        # print("diarize_segments: ", diarize_segments)
        with open(transcription_file, "a+") as f:
            for segment in result["segments"]:
                # print("SEGMENT: ", segment, "\r\n")
                if "speaker" in segment:
                    f.write("\r\n" + segment["speaker"] + ": " + segment["text"])
                else:
                    f.write("\r\n" + segment["text"])

    remove(source_file + ".lock")


@app.route("/get_status", methods=["GET"])
def get_status():
    statusfile = request.values.get("statusfile", None)
    if path.exists(statusfile):
        return jsonify({"status": "IN PROGRESS"})
    else:
        filepath, ext = path.splitext(statusfile.replace(".lock", ""))
        with open(filepath + transcripted_extender, "r") as f:
            transcription_text = f.read()
        return jsonify({"status": "OK", "transcription_text": transcription_text.strip()})


@app.route("/get_transcription", methods=["GET"])
def transcribe_from_mp3():
    audio_path = request.values.get("audio_path", None)
    # speakers_count = request.values.get("speakers_count", None)
    transcribe_thread = threading.Thread(
        target=transcribe,
        name=f"Transcriber {audio_path}",
        args=(audio_path, None),
    )
    transcribe_thread.start()
    statusfile = audio_path + ".lock"
    mknod(statusfile)
    # transcription_text = transcribe(audio_path, speakers_count)
    return jsonify({"status": "OK", "statusfile": statusfile})


def main():
    app.run(debug=bool(APP_DEBUG), host="::", port=APP_PORT)
 

if __name__ == "__main__":
    try:
        # pass someth to load models
        sys.argv[1]
        transcribe("for_load.mp3")
    except IndexError:
        main()
