[![Build Status](https://travis-ci.org/benmack/eo-box.svg?branch=master)](https://travis-ci.org/benmack/eo-box)
[![Docs Status](https://readthedocs.org/projects/eo-box/badge/?version=latest)](https://eo-box.readthedocs.io/en/latest/?badge=latest)

# eo-box

**eobox** is a Python package with a small collection of tools for working with Remote Sensing / Earth Observation data. 

## Package Overview

The structure of this project has been created following the [eo-learn project of Sinergise](https://github.com/sentinel-hub/eo-learn).
For a package containing diverse functionalities as it is envisaged for this package as well, it is convincing to subdivide the pacakge into ["several subpackages according to different functionalities and external package dependencies"](https://github.com/sentinel-hub/eo-learn).

So far, the following subpackages are available:

- **`eo-box-sampledata`** contains small sample data that can be used for playing around and testing.
- **`eo-learn-raster`** contains raster processing tools for  
    * extracting raster values at given (by vector data) locations,
    * window- / chunk-wise processing of multiple single layer raster files as a stack.


## Installation

The package requires Python 3. It can be installed with the following command:

```bash
pip install eo-learn
```

It is also possible to install the subpackage separately:

```bash
pip install eo-box-sampledata
pip install eo-box-raster
```

## Documentation

The package documentation can be found at [readthedocs](https://eo-box.readthedocs.io/).
