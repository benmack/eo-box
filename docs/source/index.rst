.. eo-box documentation master file, created by
   sphinx-quickstart on Wed Jul 11 10:54:53 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

eo-box
======

**eobox** is a Python package with a small collection of tools for working with Remote Sensing / Earth Observation data. 


Package Overview
----------------

So far, the following subpackages are available:

* **eobox.sampledata** contains small sample data that can be used for playing around and testing.

* **eobox.raster** contains raster processing tools for

    * extracting raster values at given (by vector data) locations,

    * window- / chunk-wise processing of multiple single layer raster files as a stack.

* **eobox.vector** contains vector processing tools for

    * clean convertion of polygons to lines and 

    * distance-to-polygon border calculation.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   install

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples_sampledata
   examples_raster
   examples_vector

.. toctree::
   :maxdepth: 2
   :caption: API Docs

   eobox

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
