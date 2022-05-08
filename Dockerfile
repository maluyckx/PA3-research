FROM nvidia/cuda:11.4.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip libcudnn8 && apt-get -y clean && rm -rf /var/lib/apt/lists/*
#COPY main2.py /tmp/main2.py
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
