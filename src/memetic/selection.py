import random
import numpy as np
from core.tree import Node
from core.evaluator import Evaluator


class AdaptiveSelectionManager:
    """
    Adaptive selection manager for Genetic Programming.
    Dynamically selects and applies selection strategies based on evolution statistics.
    """

    def __init__(self, statistics):
        """
        Initialize the adaptive selection manager.

        Args:
            statistics (GPStatistics): Object to track evolutionary statistics.
        """
        self.statistics = statistics
        self.active_strategy = "elitist"  # Default strategy

    def _apply_selection(self, population, x, y, bloat_penalty, selection_logic_fn):
        """
        Base function to apply a selection strategy.

        Args:
            population (list[Node]): The population to select from.
            x (np.ndarray): Input data.
            y (np.ndarray): Target output data.
            bloat_penalty (float): Penalty for tree complexity.
            selection_logic_fn (function): Logic defining the selection method.

        Returns:
            Node: The selected individual.
        """
        return selection_logic_fn(population, x, y, bloat_penalty)

    def tournament_selection(self, population, x, y, bloat_penalty, tournament_size=3):
        """
        Tournament selection: selects the best individual from a random subset.
        """
        def logic(pop, x, y, penalty):
            competitors = random.sample(pop, tournament_size)
            fitness_values = [
                Evaluator.fitness_function(ind, x, y, penalty) for ind in competitors
            ]
            best_index = np.argmin(fitness_values)
            return competitors[best_index]

        return self._apply_selection(population, x, y, bloat_penalty, logic)

    def roulette_selection(self, population, x, y, bloat_penalty):
        """
        Roulette wheel selection: selects based on fitness-proportional probabilities.
        """
        def logic(pop, x, y, penalty):
            fitness_values = [
                Evaluator.fitness_function(ind, x, y, penalty) for ind in pop
            ]
            scores = [1 / (1 + f) for f in fitness_values]  # Higher fitness → Higher probability
            total = sum(scores)
            pick = random.random() * total
            current = 0
            for ind, s in zip(pop, scores):
                current += s
                if current > pick:
                    return ind
            return pop[-1]  # Fallback

        return self._apply_selection(population, x, y, bloat_penalty, logic)

    def rank_selection(self, population, x, y, bloat_penalty):
        """
        Rank selection: assigns probabilities based on rank rather than fitness.
        """
        def logic(pop, x, y, penalty):
            fitness_values = [
                Evaluator.fitness_function(ind, x, y, penalty) for ind in pop
            ]
            sorted_indices = np.argsort(fitness_values)
            ranks = np.arange(1, len(pop) + 1)
            probabilities = ranks / ranks.sum()
            return pop[np.random.choice(sorted_indices, p=probabilities)]

        return self._apply_selection(population, x, y, bloat_penalty, logic)

    def elitist_selection(self, population, x, y, bloat_penalty):
        """
        Elitist selection: always selects the best individual.
        """
        def logic(pop, x, y, penalty):
            fitness_values = [
                Evaluator.fitness_function(ind, x, y, penalty) for ind in pop
            ]
            best_index = np.argmin(fitness_values)
            return pop[best_index]

        return self._apply_selection(population, x, y, bloat_penalty, logic)

    def choose_strategy(self):
        """
        Dynamically selects the active selection strategy based on statistics.
        """
        old_strategy = self.active_strategy
        new_strategy = "elitist"  # Default strategy
        reason = "Default conditions"

        if self.statistics.generations_no_improvement > 3:
            new_strategy = "roulette"
            reason = "Stagnation detected"
        elif self.statistics.diversity < 5:
            new_strategy = "tournament"
            reason = "Low diversity detected"
        elif self.statistics.complexity > 10:
            new_strategy = "rank"
            reason = "High complexity detected"

        if old_strategy != new_strategy:
            self.statistics.update_single_strategy(
                strategy_type="selection",
                old_strategy=old_strategy,
                new_strategy=new_strategy,
                reason=reason
            )
        self.active_strategy = new_strategy

    def select(self, population, x, y, bloat_penalty):
        """
        Applies the currently active selection strategy.
        """
        self.choose_strategy()
        strategies = {
            "tournament": self.tournament_selection,
            "roulette": self.roulette_selection,
            "rank": self.rank_selection,
            "elitist": self.elitist_selection,
        }
        return strategies[self.active_strategy](population, x, y, bloat_penalty)

    def get_active_strategy(self):
        """
        Returns the currently active selection strategy.
        """
        return self.active_strategy
