"""
eobox.ml subpackage

Subpackage which mainly builds upon sklearn and mlxtend.
It provides tools for 

* plotting confusion matrix

For more information on the package content, visit [readthedocs](https://eo-box.readthedocs.raster/en/latest/eobox.ml.html).

"""

from .plot import plot_confusion_matrix
from .clf_extension import predict_extended
