ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Ben Mack <ben8mack@gmail.com>"

USER root

RUN pip install shapely  && \
    pip install geopandas  && \
    pip install rasterio

RUN apt-get update && apt-get install software-properties-common -y
RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable && apt-get update
RUN apt-get install gdal-bin -y && apt-get install libgdal-dev -y
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal && export C_INCLUDE_PATH=/usr/include/gdal

RUN pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') && \
    pip install jupyterhub==1.0.0

RUN apt-get install unrar -y && \
    apt-get install lftp -y && \
    apt-get install libproj-dev -y && \
    apt-get install libgdal-dev -y && \
        apt-get install gdal-bin -y && \
        apt-get install proj-bin -y

RUN apt-get remove pkg-config -y

ENV PROJ_LIB="/opt/conda/share/proj"

WORKDIR /tmp

COPY setup.py requirements-dev-examples.txt ./

RUN pip install --no-cache-dir -r requirements-dev-examples.txt

RUN rm -r ./*

WORKDIR $HOME

USER $NB_UID

COPY . /home/${NB_USER}/eo-box

RUN cp -r ./eo-box/examples /home/${NB_USER}/eo-box-examples

USER root

RUN pip install --no-cache-dir -e ./eo-box

USER $NB_UID

