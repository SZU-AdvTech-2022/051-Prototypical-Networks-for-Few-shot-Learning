#!/usr/bin/env python3

r"""
A set of high-level algorithm implementations, with easy-to-use API.
"""

from .maml import MAML, maml_update
from .coloral import COLORAL, coloral_update
from .meta_sgd import MetaSGD, meta_sgd_update
from .gbml import GBML
'''
from .lightning import (
    LightningEpisodicModule,
    LightningMAML,
    LightningANIL,
    LightningPrototypicalNetworks,
    LightningMetaOptNet,
)
'''