# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


# Notez bien: No need to include f0 -- it's just an example!
def f0(x: np.ndarray) -> np.ndarray:
    return 0.069
    


def f1(x: np.ndarray) -> np.ndarray:
    return -0.062
    


def f2(x: np.ndarray) -> np.ndarray:
    return np.subtract(np.exp(np.exp(x[1])), np.exp(np.multiply(np.log10(x[1]), np.divide(x[1], x[2]))))
    


def f3(x: np.ndarray) -> np.ndarray:
    return np.add(np.negative(x[1]), np.abs(np.subtract(np.abs(np.negative(np.log(x[1]))), np.add(np.log(x[1]), np.cos(-0.703)))))
    


def f4(x: np.ndarray) -> np.ndarray:
    return np.abs(np.exp(np.exp(np.cos(x[1]))))
    


def f5(x: np.ndarray) -> np.ndarray:
    return -0.002
    


def f6(x: np.ndarray) -> np.ndarray:
    return x[1]
    


def f7(x: np.ndarray) -> np.ndarray:
    return np.add(np.negative(np.log10(-0.7674633971181548)), 0.32864064220498235)
    


def f8(x: np.ndarray) -> np.ndarray:
    return np.negative(np.abs(np.negative(np.subtract(np.negative(np.multiply(np.multiply(np.log2(np.log(x[5])), x[1]), np.minimum(x[5], -0.9044879550263116))), np.add(x[5], 0.3132668919881396)))))
    
