#FROM ubuntu:latest
# FROM python:3.8.1
# COPY ./*.py /exp/
# COPY ./requirements.txt /exp/requirements.txt
# RUN pip3 install --no-cache-dir -r /exp/requirements.txt
# WORKDIR /exp
# CMD ["python3", "./plots_graph.py"]
#FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./svm_gamma=0.001_C=0.5.joblib /exp/svm_gamma=0.001_C=0.5.joblib
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
#CMD ["python3", "./plot_graphs.py"]

#COPY app.py /api

#ENV FLASK_APP=app.py

# Unit tests
# RUN pip install pytest && pytest

EXPOSE 5000
CMD ["python3", "./app.py" ]