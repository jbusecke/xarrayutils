from . import utils
from . import numpy_utils
from . import xmitgcm_utils
from .utils import *

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
