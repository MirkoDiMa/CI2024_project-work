import random
import numpy as np
from core.safe_math import ALL_OPERATORS

class Node:
    """
    Class representing a GP tree node.
    - op: operator (string) or None if leaf.
    - value: if leaf, ('x', i) or ('const', c).
    - children: list of nodes (0, 1, or 2 depending on the operator).
    """
    def __init__(self, op=None, value=None, children=None):
        self.op = op
        self.value = value
        self.children = children or []

    def __str__(self):
        if self.op is None:
            return f"x[{self.value[1]}]" if self.is_variable() else str(self.value[1])
        elif len(self.children) == 1:
            return f"{self.op}({self.children[0]})"
        elif len(self.children) == 2:
            return f"({self.children[0]} {self.op} {self.children[1]})"
        return "N/A"

    def is_variable(self):
        return isinstance(self.value, tuple) and self.value[0] == 'x'

    def is_constant(self):
        return isinstance(self.value, tuple) and self.value[0] == 'const'

    def evaluate_tree(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the current node and its children (recursively).
        """
        if self.op is None:
            if self.is_variable():
                return x[self.value[1], :]
            return np.full(x.shape[1], self.value[1], dtype=float)

        operator = ALL_OPERATORS.get(self.op)
        if operator is None:
            raise ValueError(f"Unknown operator: {self.op}")

        if operator.arity == 1:
            return operator.function(self.children[0].evaluate_tree(x))
        elif operator.arity == 2:
            left = self.children[0].evaluate_tree(x)
            right = self.children[1].evaluate_tree(x)
            return operator.function(left, right)

    def tree_to_expression(self) -> str:
        """
        Converts the node and its children into a symbolic representation.
        """
        if self.op is None:
            return f"x[{self.value[1]}]" if self.is_variable() else str(self.value[1])

        operator = ALL_OPERATORS.get(self.op)
        if operator is None:
            raise ValueError(f"Unknown operator: {self.op}")

        if operator.arity == 1:
            return f"{operator.numpy_symbol}({self.children[0].tree_to_expression()})"
        elif operator.arity == 2:
            left = self.children[0].tree_to_expression()
            right = self.children[1].tree_to_expression()
            return f"{operator.numpy_symbol}({left}, {right})"

    def copy_tree(self):
        """
        Creates a deep copy of the current node.
        """
        return Node(op=self.op, value=self.value, children=[child.copy_tree() for child in self.children])

    def tree_size(self) -> int:
        """
        Returns the number of nodes in the tree (including the current node).
        """
        return 1 + sum(child.tree_size() for child in self.children)

    def tree_depth(self) -> int:
        """
        Returns the maximum depth of the tree starting from the current node.
        """
        if not self.children:
            return 1
        return 1 + max(child.tree_depth() for child in self.children)

    @staticmethod
    def generate_random_tree(max_depth: int, n_features: int, grow: bool = True):
        """
        Generates a random tree with a maximum depth, penalizing operators with high computational cost.
        """
        if max_depth == 0:
            return Node(op=None, value=random_variable(n_features) if random.random() < 0.5 else random_constant())

        node_type = random.choice(['unary', 'binary', 'leaf']) if grow else random.choice(['unary', 'binary'])

        if node_type == 'leaf':
            return Node(op=None, value=random_variable(n_features) if random.random() < 0.5 else random_constant())
        elif node_type == 'unary':
            op = Node.weighted_operator_choice(arity=1)
            child = Node.generate_random_tree(max_depth - 1, n_features, grow)
            return Node(op=op, children=[child])
        else:  # node_type == 'binary'
            op = Node.weighted_operator_choice(arity=2)
            left_child = Node.generate_random_tree(max_depth - 1, n_features, grow)
            right_child = Node.generate_random_tree(max_depth - 1, n_features, grow)
            return Node(op=op, children=[left_child, right_child])

    @staticmethod
    def weighted_operator_choice(arity: int):
        """
        Randomly selects an operator, penalizing those with higher costs.
        Args:
            arity (int): Operator arity (1 for unary, 2 for binary).
        Returns:
            str: Name of the selected operator.
        """
        operators = [op for op in ALL_OPERATORS.values() if op.arity == arity]
        costs = np.array([op.cost for op in operators])
        probabilities = 1 / costs  # Inverse of cost to penalize expensive operators
        probabilities /= probabilities.sum()  # Normalize to sum to 1
        return random.choices(operators, weights=probabilities, k=1)[0].name

    @staticmethod
    def get_random_node(node):
        """
        Returns a random node and its parent from the tree.
        """
        all_nodes = []

        def traverse(current, parent):
            all_nodes.append((current, parent))
            for child in current.children:
                traverse(child, current)

        traverse(node, None)
        return random.choice(all_nodes)

    def replace_with(self, new_node):
        """
        Replaces the content of the current node with the content of another node.
        Args:
            new_node (Node): The node that will replace the current node.
        """
        self.op = new_node.op
        self.value = new_node.value
        self.children = [child.copy_tree() for child in new_node.children]

# Support functions

def random_variable(n_features):
    i = random.randint(0, n_features - 1)
    return ('x', i)

def random_constant():
    c = np.round(random.uniform(-1, 1), 3)
    return ('const', c)
