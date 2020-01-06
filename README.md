[![Build Status](https://travis-ci.org/benmack/eo-box.svg?branch=master)](https://travis-ci.org/benmack/eo-box)
[![Docs Status](https://readthedocs.org/projects/eo-box/badge/?version=latest)](https://eo-box.readthedocs.io/en/latest/?badge=latest)

# eo-box


**eobox** is a Python package with a small collection of tools for working with Remote Sensing / Earth Observation data. 


## Package Overview

So far, the following subpackages are available:

* **eobox.sampledata** contains small sample data that can be used for playing around and testing.

* **eobox.raster** contains raster processing tools for

    * extracting raster values at given (by vector data) locations,

    * window- / chunk-wise processing of multiple single layer raster files as a stack.

* **eobox.vector** contains vector processing tools for

    * clean convertion of polygons to lines and 

    * distance-to-polygon border calculation.

* **eobox.ml** contains machine learning related tools, e.g.

    * plotting a confusion matrix including with precision and recall

    * extended predict function which returns prediction, confidences, and probabilities.  


## Installation

The package requires Python 3. It can be installed with the following command:

```bash
pip3 install eobox
```

## Documentation

The package documentation can be found at [readthedocs](https://eo-box.readthedocs.io/).
