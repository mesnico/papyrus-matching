#FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN apt update && apt install -y git
