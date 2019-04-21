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


## Installation

The package requires Python 3. It can be installed with the following command:

```bash
pip install eo-box
```

## Documentation

The package documentation can be found at [readthedocs](https://eo-box.readthedocs.io/).
