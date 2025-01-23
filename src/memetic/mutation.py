import random
from core.tree import Node, random_variable, random_constant
from core.safe_math import ALL_OPERATORS


class AdaptiveMutationManager:
    """
    Adaptive mutation manager for genetic programming.

    This class dynamically selects and applies different mutation strategies
    based on statistics and adaptation criteria.
    """

    def __init__(self, statistics):
        """
        Initializes the adaptive mutation manager.

        Args:
            statistics (GPStatistics): Object to track statistics and logging during evolution.
        """
        self.statistics = statistics
        self.active_strategy = "simple"  # Default strategy

    def simple_mutation(self, individual: Node, n_features: int) -> Node:
        """
        Simple mutation strategy:
        - Preserves the arity of the operator in the randomly selected node.
        - For leaf nodes, changes the constant or the variable.

        This strategy favors local and simple changes.
        """
        mutant = individual.copy_tree()
        node, _ = Node.get_random_node(mutant)

        if node.op is None:  # Leaf node
            if node.is_variable():
                node.value = random_constant()
            else:
                node.value = random_variable(n_features)
        else:  # Internal node
            current_arity = ALL_OPERATORS[node.op].arity
            valid_ops = [op for op in ALL_OPERATORS.values() if op.arity == current_arity]
            node.op = random.choice(valid_ops).name

        return mutant

    def subtree_mutation(self, individual: Node, n_features: int) -> Node:
        """
        Subtree mutation strategy:
        - Replaces a randomly selected subtree with a new randomly generated tree.

        Favors exploration by introducing entirely new structures.
        """
        mutant = individual.copy_tree()
        node, _ = Node.get_random_node(mutant)

        new_subtree = Node.generate_random_tree(max_depth=3, n_features=n_features, grow=True)
        node.replace_with(new_subtree)

        return mutant

    def shrink_mutation(self, individual: Node, n_features: int) -> Node:
        """
        Shrink mutation strategy:
        - Replaces a subtree with a leaf node, simplifying the structure.

        Useful for reducing excessive complexity (bloat problem).
        """
        mutant = individual.copy_tree()
        node, _ = Node.get_random_node(mutant)

        if node.op is not None:  # Internal nodes only
            node.op = None
            node.value = random.choice([random_constant(), random_variable(n_features)])
            node.children = []

        return mutant

    def diversity_mutation(self, individual: Node, n_features: int) -> Node:
        """
        Diversity mutation strategy:
        - Applies random modifications to 2-5 nodes in the tree.

        Encourages population diversity.
        """
        mutant = individual.copy_tree()
        for _ in range(random.randint(2, 5)):  # Modify 2-5 random nodes
            node, _ = Node.get_random_node(mutant)
            if node.op is None:  # Leaf node
                if node.is_variable():
                    node.value = random_constant()
                else:
                    node.value = random_variable(n_features)
            else:  # Internal node
                current_arity = ALL_OPERATORS[node.op].arity
                valid_ops = [op for op in ALL_OPERATORS.values() if op.arity == current_arity]
                node.op = random.choice(valid_ops).name
        return mutant

    def choose_strategy(self):
        """
        Selects the active mutation strategy based on statistics and logs the reason for the change.
        """
        old_strategy = self.active_strategy
        new_strategy = old_strategy  # Default strategy
        reason = "Default strategy (simple)"

        if self.statistics.generations_no_improvement > 3:
            new_strategy = "diversity"
            reason = "Detected stagnation (no improvement for 7 generations)"
        elif self.statistics.complexity > 12_000:
            new_strategy = "shrink"
            reason = "High complexity (>12,000)"
        elif self.statistics.diversity < 3:
            new_strategy = "subtree"
            reason = "Low diversity (<3)"
        else:
            new_strategy = "simple"
            reason = "Stable conditions"

        # Update the strategy if it has changed
        if old_strategy != new_strategy:
            self.statistics.update_single_strategy(
                strategy_type="mutation",
                old_strategy=old_strategy,
                new_strategy=new_strategy,
                reason=reason
            )

        self.active_strategy = new_strategy

    def mutate(self, individual: Node, n_features: int) -> Node:
        """
        Applies the selected mutation strategy to the individual.
        """
        self.choose_strategy()
        strategies = {
            "simple": self.simple_mutation,
            "subtree": self.subtree_mutation,
            "shrink": self.shrink_mutation,
            "diversity": self.diversity_mutation,
        }

        return strategies[self.active_strategy](individual, n_features)

    def get_active_strategy(self) -> str:
        """
        Returns the currently active mutation strategy.
        """
        return self.active_strategy
