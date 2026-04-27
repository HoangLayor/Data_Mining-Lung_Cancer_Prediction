FROM python:3.11-slim

LABEL maintainer="sy"
LABEL organization="proptit"

ARG MLFLOW_VERSION

# Define virtual env
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir mlflow==${MLFLOW_VERSION} psycopg2-binary charset-normalizer==2.1.0

WORKDIR /mlflow/

EXPOSE 5000

# Ref: https://github.com/mlflow/mlflow/issues/6274#issuecomment-1196106372
CMD mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000