FROM python:3.6.7

RUN pip install tf-nightly-2.0-preview==2.0.0-dev20190504
RUN pip install tensorflow_datasets
RUN pip install tfds-nightly==1.0.2.dev201905010105
RUN pip install flask
RUN pip install flask-cors
RUN pip install matplotlib
RUN pip install sklearn