import numpy as np
import csv
from core.evaluator import Evaluator
from utility.plotting import Plotter

class GPStatistics:
    def __init__(self, logger=None):
        """
        Initialize the metrics.
        """
        self.logger = logger  # Optional reference to the logger

        self.history = {
            "generation": [],
            "best_fitness": [],
            "average_fitness": [],
            "diversity": [],
            "complexity": [],
        }
        self.best_fitness = float('inf')
        self.generations_no_improvement = 0
        self.complexity = 0.0
        self.diversity = 0.0
        self.current_generation = 0

        # Track strategy usage
        self.strategy_usage = {
            "selection": {},
            "crossover": {},
            "mutation": {},
            "local_search": {},
        }
        # Save active strategies for the current generation
        self.last_active_strategies = {}

    def update(self, population, x, y, bloat_penalty, best_fitness_current, active_strategies):
        """
        Update statistics based on the current population.

        Args:
            population (list): List of current individuals.
            x (np.ndarray): Input data.
            y (np.ndarray): Expected output data.
            bloat_penalty (float): Penalty for tree complexity.
            best_fitness_current (float): Current best fitness value.
            active_strategies (dict): Active strategies in this generation.
        """
        self.current_generation += 1

        # Stagnation
        if best_fitness_current < self.best_fitness:
            self.best_fitness = best_fitness_current
            self.generations_no_improvement = 0
        else:
            self.generations_no_improvement += 1

        # Complexity
        sizes = [ind.tree_size() for ind in population]
        self.complexity = float(np.mean(sizes))

        # Diversity
        self.diversity = self.calculate_diversity(population, x, y, bloat_penalty)

        # Calculate average fitness
        fitness_values = [
            Evaluator.fitness_function(ind, x, y, bloat_penalty) for ind in population
        ]
        avg_fitness = np.mean(fitness_values)

        # Update strategy usage
        for strategy_type, strategy_name in active_strategies.items():
            self._update_strategy_usage(strategy_type, strategy_name)

        # Save historical data
        self.history["generation"].append(self.current_generation)
        self.history["best_fitness"].append(best_fitness_current)
        self.history["average_fitness"].append(avg_fitness)
        self.history["diversity"].append(self.diversity)
        self.history["complexity"].append(self.complexity)
        # Save active strategies for the current generation
        self.last_active_strategies = active_strategies

    def _update_strategy_usage(self, strategy_type, strategy_name):
        """
        Internal method to increment the usage count of a strategy.
        """
        if strategy_name not in self.strategy_usage[strategy_type]:
            self.strategy_usage[strategy_type][strategy_name] = 0
        self.strategy_usage[strategy_type][strategy_name] += 1

    def update_single_strategy(self, strategy_type: str, old_strategy: str, new_strategy: str, reason: str = ""):
        """
        Update and log a strategy change. Notify the logger as well.

        Args:
            strategy_type (str): Type of strategy (e.g., "mutation", "selection", etc.).
            old_strategy (str): Previous strategy.
            new_strategy (str): New active strategy.
            reason (str, optional): Reason for the change.
        """
        if old_strategy != new_strategy:
            self._update_strategy_usage(strategy_type, new_strategy)
            if self.logger:
                message = [f"{strategy_type.capitalize()} strategy changed from {old_strategy} to {new_strategy}"]
                if reason:
                    message.append(f"Reason: {reason}")
                self.logger.log_message(message)

    def calculate_diversity(self, population, x, y, bloat_penalty) -> float:
        """
        Calculate diversity as the standard deviation of normalized fitness values.

        Args:
            population (list): List of individuals in the population.
            x (np.ndarray): Input data.
            y (np.ndarray): Expected output data.
            bloat_penalty (float): Penalty for tree complexity.

        Returns:
            float: Diversity value.
        """
        evaluator = Evaluator()
        fits = [evaluator.fitness_function(ind, x, y, bloat_penalty) for ind in population]
        fits = np.array(fits)
        fits = fits[np.isfinite(fits)]
        if len(fits) == 0:
            return 0.0
        fits_normalized = (fits - np.min(fits)) / (np.ptp(fits) + 1e-10)
        return float(np.std(fits_normalized))

    def export_history_to_csv(self, file_path):
        """
        Export historical data to a CSV file.

        Args:
            file_path (str): Output CSV file path.
        """
        with open(file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.history.keys())
            writer.writeheader()
            rows = zip(*self.history.values())
            writer.writerows([dict(zip(self.history.keys(), row)) for row in rows])

    def generate_plots(self, output_dir="./output/plots"):
        """
        Generate and save plots based on historical data.

        Args:
            output_dir (str): Directory to save the plots.
        """
        plotter = Plotter(self.history)
        plotter.save_plots(directory=output_dir)

    def get_stats_dict(self):
        """
        Return a dictionary with current metrics.
        """
        return {
            "generation": self.current_generation,
            "best_fitness": self.best_fitness,
            "diversity": self.diversity,
            "complexity": self.complexity,
            "stagnation": self.generations_no_improvement > 5,
        }

    def get_strategy_usage(self):
        """
        Return a dictionary with strategy usage statistics.
        """
        return self.strategy_usage

    def log_current_strategies(self):
        """
        Log a message summarizing the active strategies in the current generation.
        """
        if self.logger and self.last_active_strategies:
            strategies_msg = "Active strategies in Generation {}: ".format(self.current_generation)
            strategies_parts = []
            for strategy_type, strategy in self.last_active_strategies.items():
                strategies_parts.append(f"{strategy_type}={strategy}")
            strategies_msg += ", ".join(strategies_parts)
            self.logger.log_message(strategies_msg)

    def generate_summary(self, output_dir="./output/plots"):
        """
        Generate a readable summary of statistics and save plots.

        Args:
            output_dir (str): Directory to save the plots.
        """
        summary = [
            f"Total generations: {self.current_generation}",
            f"Best fitness achieved: {self.best_fitness:.4f}",
            f"Final diversity: {self.diversity:.4f}",
            f"Final average complexity: {self.complexity:.4f}",
            f"Generations without improvements: {self.generations_no_improvement}",
        ]

        summary.append("\nStrategy usage:")
        for strategy_type, usage in self.strategy_usage.items():
            summary.append(f"  {strategy_type.capitalize()}:")
            for strategy, count in usage.items():
                summary.append(f"    {strategy}: {count} times")

        # Generate and save plots
        self.generate_plots(output_dir=output_dir)
        summary.append(f"\nPlots saved in: {output_dir}")

        return "\n".join(summary)
