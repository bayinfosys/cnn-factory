ARG BASE=python:3
FROM $BASE

LABEL maintainer="ed@bayis.co.uk"

RUN apt-get update && \
    apt-get install --no-install-recommends -qqy libgl1-mesa-glx

COPY requirements.txt /tmp/requirements.txt

# we have to install opencv for imgaug
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir opencv-python-headless && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY ./src /src


ENTRYPOINT ["python3", "-m", "src.generators"]

CMD ["--help"]
