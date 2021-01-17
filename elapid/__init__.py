"""Configuration for loading the elapid package"""
import logging as _logging
import sys as _sys

from . import metrics, plot, read
from .ml import sk_tuner
from .utils import maxent, run

_logging.basicConfig(
    level=_logging.WARNING,
    format=("%(asctime)s %(levelname)s %(name)s [%(funcName)s] | %(message)s"),
    stream=_sys.stdout,
)
