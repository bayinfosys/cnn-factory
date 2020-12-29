ARG BASE=tensorflow/keras
FROM $BASE

COPY ./src /src

CMD python3 /src/train.py
