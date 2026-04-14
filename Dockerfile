FROM python:3.10-slim-bookworm

# for russian users - use proxy
# COPY .wgetrc /root
# COPY apt.conf /etc/apt/apt.conf
# COPY pip.conf /etc/pip.conf

RUN apt-get update && apt-get upgrade -y 
RUN apt install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev libavdevice-dev gcc wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update &&\
    apt install -y cuda-toolkit

ENV APP_PORT=3000
ENV APP_DEBUG=
WORKDIR /opt
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN rm -rf /root/.cache/pip

COPY docker-cache/.cache/huggingface /root/.cache/huggingface

ENV PATH="/usr/local/cuda/bin${PATH:+:${PATH}}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64"
ENV PYTHONUNBUFFERED=1

# COPY for_load.mp3 .
COPY app_transcriber.py .

# RUN python app_transcriber.py load

CMD [ "python", "-u", "./app_transcriber.py" ]