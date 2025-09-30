from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # Implement for Task 1.1.
    vals_list = list(vals)
    vals_list[arg] += epsilon / 2
    f_plus = f(*vals_list)

    vals_list = list(vals)
    vals_list[arg] -= epsilon / 2
    f_minus = f(*vals_list)

    derivative_approx = (f_plus - f_minus) / epsilon
    return derivative_approx


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> List[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # Implement for Task 1.4.
    top_order = []
    visited = set()

    def dfs(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return
        visited.add(v.unique_id)

        for parent in v.parents:
            dfs(parent)

        top_order.append(v)

    dfs(variable)
    return top_order[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Implement for Task 1.4.
    top_order = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    for var in top_order:
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var.unique_id])
        else:
            chain = var.chain_rule(derivatives[var.unique_id])
            for parent, d_parent in chain:
                if parent.unique_id in derivatives:
                    derivatives[parent.unique_id] += d_parent
                else:
                    derivatives[parent.unique_id] = d_parent


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
