ARG BASE=tensorflow/tensorflow:1.7.0-gpu-py3
FROM $BASE

COPY requirements.txt /tmp/requirements.txt

# we have to install opencv for imgaug
RUN apt-get update && \
    apt-get install -y python-opencv && \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

COPY ./src /src

ENTRYPOINT ["python3", "/src/train.py"]

CMD ["--help"]
