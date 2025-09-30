"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, Sequence

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# Implement for Task 0.1.


def mul(num1: float, num2: float) -> float:
    return num1 * num2


def id(inp: float) -> float:
    return inp


def add(num1: float, num2: float) -> float:
    return num1 + num2


def neg(num: float) -> float:
    return -num


def lt(num1: float, num2: float) -> float:
    return num1 < num2


def eq(num1: float, num2: float) -> float:
    return num1 == num2


def max(num1: float, num2: float) -> float:
    return num2 if lt(num1, num2) else num1


def is_close(num1: float, num2: float, atol: float = 1e-2) -> float:
    return abs(num1 - num2) < atol


def exp(num: float) -> float:
    return math.exp(num)


def inv(num: float) -> float:
    return 1 / num


def sigmoid(num: float) -> float:
    if lt(num, 0.0):
        return mul(exp(num), inv(add(1.0, exp(num))))
    return inv(add(1.0, exp(neg(num))))


def relu(num: float) -> float:
    return max(0.0, num)


def log(num: float) -> float:
    return math.log(num)


def log_back(num1: float, num2: float) -> float:
    return mul(num2, inv(num1))


def inv_back(num1: float, num2: float) -> float:
    return neg(mul(num2, inv(mul(num1, num1))))


def relu_back(num1: float, num2: float) -> float:
    derivative = 1.0 if lt(0.0, num1) else 0.0
    return mul(derivative, num2)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# Implement for Task 0.3.


def map(f: Callable[[float], float], list_a: Iterable[float]) -> Iterable[float]:
    res = []
    for a in list_a:
        res.append(f(a))
    return res


def zipWith(
    f: Callable[[float, float], float], list_a: Iterable[float], list_b: Iterable[float]
) -> Iterable[float]:
    res = []
    for a, b in zip(list_a, list_b):
        res.append(f(a, b))
    return res


def reduce(f: Callable[[float, float], float], list_a: Iterable[float]) -> float:
    # used https://github.com/python/cpython/blob/282bd0fe98bf1c3432fd5a079ecf65f165a52587/Lib/functools.py#L238
    it = iter(list_a)

    try:
        value = next(it)
    except StopIteration:
        return 0.0

    for element in it:
        value = f(value, element)
    return value


def negList(list_a: Iterable[float]) -> Iterable[float]:
    return map(neg, list_a)


def addLists(list_a: Iterable[float], list_b: Iterable[float]) -> Iterable[float]:
    return zipWith(add, list_a, list_b)


def sum(list_a: Sequence[float]) -> float:
    if len(list_a) == 0:
        return 0
    return reduce(add, list_a)


def prod(list_a: Iterable[float]) -> float:
    return reduce(mul, list_a)
