FROM python:3.11-slim

WORKDIR /training

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

LABEL maintainer="sy"

CMD ["python", "code/train.py"]
