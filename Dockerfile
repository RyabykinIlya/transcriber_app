FROM python:3.10-slim

# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
COPY cuda-keyring_1.1-1_all.deb .

RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update && apt-get upgrade -y 
RUN apt install -y ffmpeg gcc wget cuda-toolkit
RUN export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}} \
    export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib

ENV APP_PORT=3000
ENV APP_DEBUG=
WORKDIR /opt
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN rm -rf /root/.cache/pip 

COPY for_load.mp3 .
COPY app_transcriber.py .
# RUN python app_transcriber.py load

CMD [ "python", "-u", "./app_transcriber.py" ]