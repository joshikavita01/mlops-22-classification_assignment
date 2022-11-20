#FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
ENTRYPOINT ["python3","./plots_graph.py"]
CMD ["python3", "./plots_graph.py"]