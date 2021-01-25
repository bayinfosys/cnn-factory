ARG BASE=python:3
FROM $BASE

COPY requirements.txt /tmp/requirements.txt

# we have to install opencv for imgaug
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir opencv-python-headless && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY ./src /src


ENTRYPOINT ["python3", "-m", "src.generators"]

CMD ["--help"]
