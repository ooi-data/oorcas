FROM prefecthq/prefect:2-python3.10

COPY ./ /tmp/oorcas

RUN pip install prefect-aws
RUN pip install -e /tmp/oorcas