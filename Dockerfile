FROM python:3.7.3


COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN mkdir /app

COPY . /app
WORKDIR /app

# Required to enable heroku ps:exec in Docker
RUN rm /bin/sh && ln -s /bin/bash /bin/sh