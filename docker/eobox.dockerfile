FROM osgeo/gdal:ubuntu-small-latest

LABEL maintainer="Ben Mack <ben8mack@gmail.com>"

WORKDIR /tmp

RUN apt-get update \
 && apt-get install python3-pip -y

COPY setup.py requirements-dev-examples.txt ./

RUN pip3 install -r requirements-dev-examples.txt

RUN pip3 install --no-cache-dir \
    jupyter

RUN rm -r ./*

WORKDIR /home/eoboxer

COPY . /home/eoboxer/eo-box

RUN mv /home/eoboxer/eo-box/examples /home/eoboxer/examples

RUN pip3 install --no-cache-dir -e ./eo-box

ENV TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

EXPOSE 8888
CMD ["/usr/local/bin/jupyter", "notebook", "--no-browser", "--port=8888", "--ip=0.0.0.0", \
     "--NotebookApp.token=''", "--allow-root"]