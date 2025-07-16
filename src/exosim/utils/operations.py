import operator

import numpy as np

ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
    "^": operator.xor,
}


def operate_over_axis(
    matrix: np.ndarray, vector: np.ndarray, axis: int, operation: str = "*"
) -> np.ndarray:
    """
    This function is used to perform an operation over a given axis of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        N dimensional array
    vector : np.ndarray
        1 dimensional array
    axis : int
        axis to operate over
    operation : str, optional
        operator symbol, by default "*"

    Returns
    -------
    np.ndarray
        operated matrix
    """
    axis_index = [np.newaxis] * matrix.ndim
    axis_index[axis] = slice(None)
    axis_index = tuple(axis_index)
    matrix = ops[operation](matrix, vector[axis_index])
    return matrix
