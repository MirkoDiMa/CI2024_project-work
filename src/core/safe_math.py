import numpy as np
from typing import Callable, Optional

MAX_EXP = 10
MAX_POWER = 5
MAX_FLOAT = 1e10
MIN_FLOAT = 1e-10

# Support functions
def clip_values(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """Clip values to a specified range."""
    return np.clip(x, min_value, max_value)

# Safe operations
def safe_divide(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Perform element-wise safe division, avoiding division by zero."""
    y_safe = np.where(np.abs(y) < MIN_FLOAT, MIN_FLOAT, y)
    return np.divide(x, y_safe)


def safe_ln(x: np.ndarray) -> np.ndarray:
    """Apply natural logarithm safely, avoiding ln(0) and negative values."""
    return np.log(clip_values(x, MIN_FLOAT, MAX_FLOAT))


def safe_sqrt(x: np.ndarray) -> np.ndarray:
    """Apply square root safely, avoiding sqrt of negative numbers."""
    return np.sqrt(clip_values(x, 0, MAX_FLOAT))


def safe_power(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Raise x to the power p safely, handling edge cases."""
    x_safe = np.clip(x, MIN_FLOAT, MAX_FLOAT)
    p_safe = np.clip(p, -MAX_POWER, MAX_POWER)

    is_fractional_power = (p_safe % 1 != 0)
    x_safe = np.where((x_safe < 0) & is_fractional_power, MIN_FLOAT, x_safe)

    result = np.power(x_safe, p_safe)
    return np.clip(np.nan_to_num(result, nan=MIN_FLOAT, posinf=MAX_FLOAT, neginf=-MAX_FLOAT), -MAX_FLOAT, MAX_FLOAT)


def safe_exp(x: np.ndarray) -> np.ndarray:
    """Apply exponential safely, avoiding overflow."""
    return np.exp(clip_values(x, -MAX_EXP, MAX_EXP))


def safe_log2(x: np.ndarray) -> np.ndarray:
    """Apply log base 2 safely."""
    return np.log2(clip_values(x, MIN_FLOAT, MAX_FLOAT))


def safe_log10(x: np.ndarray) -> np.ndarray:
    """Apply log base 10 safely."""
    return np.log10(clip_values(x, MIN_FLOAT, MAX_FLOAT))


# Operator Class
class Operator:
    """
    Represents a mathematical operator with properties relevant to the project.

    Attributes:
        name (str): Unique name of the operator.
        function (Callable): Function associated with the operator.
        arity (int): Number of operands required (1 for unary, 2 for binary).
        symbol (str): Symbol for the operator (e.g., "+", "-", "*", "/").
        numpy_symbol (str): Symbol used in NumPy for code generation.
        latex_symbol (str): LaTeX representation of the operator.
        cost (float): Computational cost of the operator.
    """

    def __init__(self, name: str, function: Callable, arity: int, symbol: str,
                 numpy_symbol: Optional[str] = None, latex_symbol: Optional[str] = None,
                 cost: float = 1.0):
        self.name = name
        self.function = function
        self.arity = arity
        self.symbol = symbol
        self.numpy_symbol = numpy_symbol or symbol
        self.latex_symbol = latex_symbol or symbol
        self.cost = cost

    def __repr__(self):
        return f"Operator(name={self.name}, arity={self.arity}, symbol='{self.symbol}', cost={self.cost})"

    def compute(self, *args: np.ndarray) -> np.ndarray:
        """Execute the operator with the given inputs."""
        if len(args) != self.arity:
            raise ValueError(f"Operator {self.name} requires {self.arity} arguments, but {len(args)} were given.")
        return self.function(*args)


# Definition of unary operators
UNARY_OPERATORS = [
    Operator(name="neg", function=np.negative, arity=1, symbol="-", numpy_symbol="np.negative", latex_symbol="-x", cost=0.052),
    Operator(name="abs", function=np.abs, arity=1, symbol="abs", numpy_symbol="np.abs", latex_symbol="|x|", cost=0.046),
    Operator(name="ln", function=safe_ln, arity=1, symbol="ln", numpy_symbol="np.log", latex_symbol=r"\ln(x)", cost=0.376),
    Operator(name="log2", function=safe_log2, arity=1, symbol="log2", numpy_symbol="np.log2", latex_symbol=r"\log_2(x)", cost=0.397),
    Operator(name="log10", function=safe_log10, arity=1, symbol="log10", numpy_symbol="np.log10", latex_symbol=r"\log_{10}(x)", cost=0.418),
    Operator(name="sqrt", function=safe_sqrt, arity=1, symbol="sqrt", numpy_symbol="np.sqrt", latex_symbol=r"\sqrt{x}", cost=0.169),
    Operator(name="exp", function=safe_exp, arity=1, symbol="exp", numpy_symbol="np.exp", latex_symbol=r"e^x", cost=0.341),
    Operator(name="sin", function=np.sin, arity=1, symbol="sin", numpy_symbol="np.sin", latex_symbol=r"\sin(x)", cost=0.720),
    Operator(name="cos", function=np.cos, arity=1, symbol="cos", numpy_symbol="np.cos", latex_symbol=r"\cos(x)", cost=0.719),
    Operator(name="tan", function=np.tan, arity=1, symbol="tan", numpy_symbol="np.tan", latex_symbol=r"\tan(x)", cost=0.406),
]

# Definition of binary operators
BINARY_OPERATORS = [
    Operator(name="add", function=np.add, arity=2, symbol="+", numpy_symbol="np.add", latex_symbol="+", cost=0.063),
    Operator(name="sub", function=np.subtract, arity=2, symbol="-", numpy_symbol="np.subtract", latex_symbol="-", cost=0.064),
    Operator(name="mul", function=np.multiply, arity=2, symbol="*", numpy_symbol="np.multiply", latex_symbol=r"\times", cost=0.063),
    Operator(name="div", function=safe_divide, arity=2, symbol="/", numpy_symbol="np.divide", latex_symbol=r"\div", cost=0.150),
    Operator(name="pow", function=safe_power, arity=2, symbol="^", numpy_symbol="np.power", latex_symbol="^", cost=0.896),
    Operator(name="min", function=lambda x, y: np.minimum(x, y), arity=2, symbol="min", numpy_symbol="np.minimum", latex_symbol=r"\min(x, y)", cost=0.061),
    Operator(name="max", function=lambda x, y: np.maximum(x, y), arity=2, symbol="max", numpy_symbol="np.maximum", latex_symbol=r"\max(x, y)", cost=0.062),
]

# Dictionary for quick access
ALL_OPERATORS = {op.name: op for op in UNARY_OPERATORS + BINARY_OPERATORS}
