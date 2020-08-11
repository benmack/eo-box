******
Docker
******

.. note:: This is work in progress. I am happy for any suggestions how to improve the Docker file and handling.

Built upon jupyter/scipy-notebook
=================================

Based on
`Jupyter Docker Stacks' jupyter/scipy-notebook <https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-scipy-notebook) image and inspired by the [scioquiver/notebooks:cgspatial-notebook image](https://github.com/SCiO-systems/cgspatial-notebook>`_
and inspired by
`SCiO-systems/cgspatial-notebook <https://github.com/SCiO-systems/cgspatial-notebook>`_.

With the following changes with respect to *cgspatial-notebook*:

* Built upon *jupyter/scipy-notebook* not jupyter/datascience-notebook.
* `ppa:ubuntugis/ubuntugis-unstable` instead of `ppa:ubuntugis/ppa`
* No R/MaxEnt stuff.
* Installed dependencies in *requirements-dev-examples.txt*.
* Added folders *eobox*, *eo-box-examples* and installed *eobox* in editable mode.

**Build**::

  docker image build -t benmack/eobox-notebook:2020-08-11 -f docker/eobox-notebook.dockerfile .

**Run - Jupyter Notebook** - this is the default.::

  docker run benmack/eobox-notebook:latest

**Run - Bash**::

  docker run --rm benmack/eobox-notebook:latest bash -c ls

**Run - Python**, e.g. test if eobox can be imported::

  docker run --rm benmack/eobox-notebook:latest python -c 'import eobox'

**Run - (interactive) IPython**::

  docker run -it --rm benmack/eobox-notebook:latest ipython

**Run - Jupyter Lab** and hang in the sample data as a volume (assuming you are in the root dir of the repository)::

    docker run -p 8080:8888 -v ${PWD}/eobox/sampledata/data:/home/jovyan/data benmack/eobox-notebook:latest jupyter lab

If you have problems with the default *8888:8888* port you can change this as in the example above.
But in this case you need to open the respective port in your browser (http://127.0.0.1:8080/tree) and enter the token you see in the logs.

See other `Docker Options <https://jupyter-docker-stacks.readthedocs.io/en/latest/using/common.html#docker-options>`_ that might (or might not) work with this image.

Built upon osgeo/gdal:ubuntu-small-latest
=========================================

**Build & run**::

  docker image build -t benmack/eobox:latest -f docker/eobox.dockerfile .

  docker run  -v ${PWD}:/home/eoboxer/host_pwd -p 8888:8888 benmack/eobox:latest

Push a new Docker Image
=======================

Currently:

* Make changes in the image, e.g. *docker/eobox-notebook.dockerfile*.

* Change the respective version file, e.g. version_eobox-notebook

* Push to Docker Hub::

    docker push benmack/eobox-notebook:latest
    docker push benmack/eobox-notebook:0.0.1

**TODO**:

In the future install eobox from a release such that it is clear which exact version and code is in the Dockerfile.

Improve versioning and automize release.
Possible starting point: `How to Version Your Docker Images <https://medium.com/better-programming/how-to-version-your-docker-images-1d5c577ebf54>`_.
