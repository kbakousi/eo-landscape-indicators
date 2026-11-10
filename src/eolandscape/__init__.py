"""eolandscape: Earth-observation landscape indicators from open data.

Each module is an independent implementation of the methods described in the
publications cited in its docstring.
"""

from . import anomalies, diversity, fire, fragmentation, indices, landcover, masking, trends, zonal

__all__ = ["anomalies", "diversity", "fire", "fragmentation", "indices", "landcover", "masking", "trends", "zonal"]
__version__ = "0.1.1"
