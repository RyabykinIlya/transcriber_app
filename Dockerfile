FROM python:3.10-slim

RUN apt-get update && apt-get upgrade -y 

ENV APP_PORT=3000
ENV APP_DEBUG=
WORKDIR /opt
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN rm -rf /root/.cache/pip 

COPY for_load.mp3 .
COPY app_transcriber.py .
RUN python app_transcriber.py load

CMD [ "python", "-u", "./app_transcriber.py" ]