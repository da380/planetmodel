"""catalogue -- named reference models shipped with planetmodel.

`prem()` is Dziewonski & Anderson's Preliminary Reference Earth Model,
built exactly from its published polynomials; `PREM` is its catalogue
name.
"""
from .prem import PREM, prem

__all__ = ["prem", "PREM"]
