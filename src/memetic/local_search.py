import random
import numpy as np
from core.tree import Node
from core.evaluator import Evaluator
from core.safe_math import ALL_OPERATORS


class LocalSearchManager:
    """
    Modular Local Search Manager with optimized computational efficiency.
    Includes hill climbing, tabu search, simulated annealing, and random improvement.
    """

    def __init__(self, statistics, tabu_size=10, initial_temperature=1.0, cooling_rate=0.95):
        self.statistics = statistics
        self.evaluator = Evaluator()
        self.active_strategy = "simulated_annealing"  # Default strategy

        # Tabu Search parameters
        self.tabu_list = set()  # Use set for O(1) lookups
        self.tabu_size = tabu_size

        # Simulated Annealing parameters
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def tweak(self, individual: Node, n_features: int) -> Node:
        """
        Applies a random modification to the given individual.
        Args:
            individual (Node): The individual to modify.
            n_features (int): The number of input features.
        Returns:
            Node: A modified copy of the individual.
        """
        candidate = individual.copy_tree()
        node, _ = Node.get_random_node(candidate)

        if node.op is not None:  # Internal node
            valid_ops = [op.name for op in ALL_OPERATORS.values() if op.arity == ALL_OPERATORS[node.op].arity]
            node.op = random.choice(valid_ops)
        else:  # Leaf node
            if node.is_variable():
                node.value = ('const', random.uniform(-1, 1))
            else:
                node.value = ('x', random.randint(0, n_features - 1))

        return candidate

    def evaluate_candidates(self, candidates, x, y, bloat_penalty):
        """
        Efficiently evaluates a list of candidates using NumPy vectorized operations.
        """
        fitnesses = [
            self.evaluator.fitness_function(candidate, x, y, bloat_penalty)
            for candidate in candidates
        ]
        return fitnesses

    def hill_climb(self, individual, x, y, bloat_penalty):
        """
        Hill climbing: Accepts only improvements.
        """
        candidate = self.tweak(individual, x.shape[0])
        if self.evaluator.fitness_function(candidate, x, y, bloat_penalty) < self.evaluator.fitness_function(
                individual, x, y, bloat_penalty):
            return candidate
        return individual

    def tabu_search(self, individual, x, y, bloat_penalty):
        """
        Tabu Search: Avoid revisiting solutions using a tabu set.
        """
        candidates = [self.tweak(individual, x.shape[0]) for _ in range(5)]
        candidates = [c for c in candidates if c.tree_to_expression() not in self.tabu_list]

        if not candidates:  # All candidates are tabu
            return individual

        fitnesses = self.evaluate_candidates(candidates, x, y, bloat_penalty)
        best_candidate = candidates[np.argmin(fitnesses)]

        # Update tabu list
        self.tabu_list.add(best_candidate.tree_to_expression())
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop()  # Efficient removal with set

        return best_candidate

    def simulated_annealing(self, individual, x, y, bloat_penalty):
        """
        Simulated Annealing: Accept worse solutions with a probability.
        """
        candidate = self.tweak(individual, x.shape[0])
        current_fitness = self.evaluator.fitness_function(individual, x, y, bloat_penalty)
        new_fitness = self.evaluator.fitness_function(candidate, x, y, bloat_penalty)

        delta = new_fitness - current_fitness
        if delta < 0 or random.random() < np.exp(-delta / self.temperature):
            self.temperature *= self.cooling_rate
            return candidate

        return individual

    def random_improvement(self, individual, x, y, bloat_penalty):
        """
        Random improvement: Tries multiple tweaks and selects the best.
        """
        candidates = [self.tweak(individual, x.shape[0]) for _ in range(5)]
        fitnesses = self.evaluate_candidates(candidates, x, y, bloat_penalty)

        best_candidate = candidates[np.argmin(fitnesses)]
        return best_candidate if min(fitnesses) < self.evaluator.fitness_function(individual, x, y, bloat_penalty) else individual

    def choose_strategy(self):
        """
        Dynamically selects the active local search strategy based on global statistics.
        """
        previous_strategy = self.active_strategy
        reason = "Default strategy"

        if self.statistics.complexity > 15:
            self.active_strategy = "hill_climb"
            reason = "High complexity (>15)"
        elif self.statistics.diversity < 3:
            self.active_strategy = "tabu_search"
            reason = "Low diversity (<3)"
        elif self.temperature > 0.1:
            self.active_strategy = "simulated_annealing"
            reason = "High temperature (>0.1)"
        else:
            self.active_strategy = "random_improvement"
            reason = "Default conditions"

        self.statistics.update_single_strategy(
            strategy_type="local_search",
            old_strategy=previous_strategy,
            new_strategy=self.active_strategy,
            reason=reason
        )

    def local_search(self, individual, x, y, bloat_penalty):
        """
        Applies the dynamically chosen local search strategy to the individual.
        """
        self.choose_strategy()
        strategies = {
            "hill_climb": self.hill_climb,
            "tabu_search": self.tabu_search,
            "simulated_annealing": self.simulated_annealing,
        }
        return strategies[self.active_strategy](individual, x, y, bloat_penalty)
        
    def get_active_strategy(self) -> str:
        """
        Returns the currently active local search strategy.
        """
        return self.active_strategy
