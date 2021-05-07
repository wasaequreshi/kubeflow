FROM tensorflow/tfx:0.26.3
WORKDIR /pipeline
COPY ./ ./
RUN pip install keras 
RUN pip install tensorflow-data-validation
RUN pip install tensorflow-model-analysis
RUN pip install tensorflow-metadata
RUN pip install tensorflow-transform
RUN pip install ml-metadata
RUN pip install apache-beam
RUN pip install pyarrow
RUN pip install tfx-bsl
RUN pip install oauth2client==3.0.0
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"