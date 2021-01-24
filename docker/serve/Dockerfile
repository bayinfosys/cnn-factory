ARG BASE=tensorflow/tensorflow:2.2.2-py3
FROM $BASE

COPY requirements.serve.txt /tmp/requirements.txt

# we have to install opencv for imgaug
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir opencv-python-headless && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY ./src /src

WORKDIR /

CMD ["/bin/bash", "-c", "gunicorn --bind 0.0.0.0:5000 --worker-class gevent --access-logfile - src.serve:app"]
