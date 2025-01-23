import random
from core.tree import Node

class AdaptiveCrossoverManager:
    """
    Adaptive crossover manager for Genetic Programming.
    Provides dynamic selection and application of crossover strategies based on evolution statistics.
    """

    def __init__(self, statistics):
        """
        Initialize the adaptive crossover manager.

        Args:
            statistics (GPStatistics): Object tracking evolution statistics and data.
        """
        self.statistics = statistics
        self.active_strategy = "one_point"  # Default strategy

    def _apply_crossover(self, parent1: Node, parent2: Node, swap_logic_fn) -> tuple[Node, Node]:
        """
        Base function to apply crossover to two parents.

        Args:
            parent1 (Node): First parent.
            parent2 (Node): Second parent.
            swap_logic_fn (function): Logic defining how to swap subtrees/nodes.

        Returns:
            tuple[Node, Node]: Two offspring generated after crossover.
        """
        child1 = parent1.copy_tree()
        child2 = parent2.copy_tree()
        node1, _ = Node.get_random_node(child1)
        node2, _ = Node.get_random_node(child2)
        swap_logic_fn(node1, node2)
        return child1, child2

    def subtree_crossover(self, parent1: Node, parent2: Node) -> tuple[Node, Node]:
        """
        Subtree crossover: swaps entire subtrees between two parents.
        """
        def swap_logic(node1, node2):
            node1.op, node2.op = node2.op, node1.op
            node1.value, node2.value = node2.value, node1.value
            node1.children, node2.children = node2.children, node1.children

        return self._apply_crossover(parent1, parent2, swap_logic)

    def one_point_crossover(self, parent1: Node, parent2: Node) -> tuple[Node, Node]:
        """
        One-point crossover: swaps the first child of selected nodes.
        """
        def swap_logic(node1, node2):
            if len(node1.children) >= 1 and len(node2.children) >= 1:
                node1.children[0], node2.children[0] = node2.children[0], node1.children[0]

        return self._apply_crossover(parent1, parent2, swap_logic)

    def uniform_crossover(self, parent1: Node, parent2: Node) -> tuple[Node, Node]:
        """
        Uniform crossover: swaps node properties based on a random probability.
        """
        def swap_logic(node1, node2):
            if random.random() < 0.5:
                node1.op, node2.op = node2.op, node1.op
                node1.value, node2.value = node2.value, node1.value

        return self._apply_crossover(parent1, parent2, swap_logic)

    def blended_crossover(self, parent1: Node, parent2: Node) -> tuple[Node, Node]:
        """
        Blended crossover: averages numerical values of corresponding nodes.
        """
        def swap_logic(node1, node2):
            if isinstance(node1.value, (int, float)) and isinstance(node2.value, (int, float)):
                blend = 0.5 * (node1.value + node2.value)
                node1.value, node2.value = blend, blend

        return self._apply_crossover(parent1, parent2, swap_logic)

    def choose_strategy(self):
        """
        Dynamically selects the active crossover strategy based on statistics.
        """
        old_strategy = self.active_strategy
        new_strategy = "one_point"  # Default strategy
        reason = "Default conditions"

        if self.statistics.generations_no_improvement > 3:
            new_strategy = "subtree"
            reason = "Stagnation detected"
        elif self.statistics.diversity < 0.5:
            new_strategy = "uniform"
            reason = "Low diversity (<0.5)"
        elif self.statistics.complexity > 20:
            new_strategy = "blended"
            reason = "High complexity (>20)"

        if old_strategy != new_strategy:
            self.statistics.update_single_strategy(
                strategy_type="crossover",
                old_strategy=old_strategy,
                new_strategy=new_strategy,
                reason=reason
            )
        self.active_strategy = new_strategy

    def crossover(self, parent1: Node, parent2: Node) -> tuple[Node, Node]:
        """
        Applies the currently active crossover strategy.
        """
        self.choose_strategy()
        strategies = {
            "subtree": self.subtree_crossover,
            "one_point": self.one_point_crossover,
            "uniform": self.uniform_crossover,
            "blended": self.blended_crossover,
        }
        return strategies[self.active_strategy](parent1, parent2)

    def get_active_strategy(self) -> str:
        """
        Returns the currently active crossover strategy.
        """
        return self.active_strategy
