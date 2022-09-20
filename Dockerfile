FROM python:3

WORKDIR /usr/src/app


COPY low_requirements.txt ./

RUN pip install --no-cache-dir -r low_requirements.txt



