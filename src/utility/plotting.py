from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np

class Plotter:
    def __init__(self, plot_dir, plot_dir_prefix="", history=None):
        """
        Initializes the plotter with the directory to save the plots.

        Args:
            plot_dir (str): Base directory to save the plots.
            plot_dir_prefix (str): Prefix to distinguish plots of a specific problem.
            history (dict): Dictionary containing historical data collected by GPStatistics.
        """
        self.plot_dir = plot_dir
        self.plot_dir_prefix = plot_dir_prefix  # Added to avoid error
        os.makedirs(self.plot_dir, exist_ok=True)
        self.history = history


    def save_plot(self, fig, filename):
        """
        Saves the plot in the specified directory.

        Args:
            fig (Figure): Matplotlib figure object.
            filename (str): Filename for the plot.
        """
        # Add the specific prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefixed_filename = f"p_{self.plot_dir_prefix}_{timestamp}_{filename}"
        filepath = os.path.join(self.plot_dir, prefixed_filename)
        fig.savefig(filepath, bbox_inches='tight')
        plt.close(fig)


    def plot_best_fitness(self):
        """
        Plots the trend of the best fitness per generation.
        """
        generations = self.history["generation"]
        best_fitness = self.history["best_fitness"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, best_fitness, label="Best Fitness", linewidth=2, color="blue")
        ax.set_title("Best Fitness Trend")
        ax.set_xlabel("Generations")
        ax.set_ylabel("Best Fitness")
        ax.legend()
        ax.grid()
        self.save_plot(fig, "best_fitness_trend.png")

    def plot_average_fitness(self):
        """
        Plots the evolution of average fitness per generation.
        """
        generations = self.history["generation"]
        avg_fitness = self.history["average_fitness"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, avg_fitness, label="Average Fitness", linewidth=2, linestyle="--", color="orange")
        ax.set_title("Average Fitness Trend")
        ax.set_xlabel("Generations")
        ax.set_ylabel("Average Fitness")
        ax.legend()
        ax.grid()
        self.save_plot(fig, "average_fitness_trend.png")

    def plot_fitness(self, title="Fitness Trend"):
        """
        Plots both the best fitness and average fitness.
        
        Args:
            title (str): Title of the plot.
        """
        generations = self.history["generation"]
        best_fitness = self.history["best_fitness"]
        avg_fitness = self.history["average_fitness"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, best_fitness, label="Best Fitness", linewidth=2)
        ax.plot(generations, avg_fitness, label="Average Fitness", linestyle="--", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Generations")
        ax.set_ylabel("Fitness")
        ax.legend()
        ax.grid()
        self.save_plot(fig, "fitness_trend.png")

    def plot_diversity(self, title="Diversity Trend"):
        """
        Plots the diversity of the population.
        
        Args:
            title (str): Title of the plot.
        """
        generations = self.history["generation"]
        diversity = self.history["diversity"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, diversity, label="Diversity", color="orange", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Generations")
        ax.set_ylabel("Diversity")
        ax.legend()
        ax.grid()
        self.save_plot(fig, "diversity_trend.png")

    def plot_complexity(self, title="Complexity Trend"):
        """
        Plots the average complexity of the population.
        
        Args:
            title (str): Title of the plot.
        """
        generations = self.history["generation"]
        complexity = self.history["complexity"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, complexity, label="Complexity", color="green", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Generations")
        ax.set_ylabel("Complexity")
        ax.legend()
        ax.grid()
        self.save_plot(fig, "complexity_trend.png")

    def plot_operation_frequencies(self, strategy_usage):
        """
        Bar charts showing the relative frequency of operations applied per generation.

        Args:
            strategy_usage (dict): Usage of strategies per operation and generation.
        """
        generations = sorted(list(strategy_usage["selection"].keys()))
        operations = ["selection", "crossover", "mutation"]

        operation_counts = {
            op: [strategy_usage[op].get(g, 0) for g in generations]
            for op in operations
        }

        x = np.arange(len(generations))
        bar_width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, op in enumerate(operations):
            ax.bar(x + i * bar_width, operation_counts[op], width=bar_width, label=op.capitalize())

        ax.set_title("Operation Frequencies per Generation")
        ax.set_xlabel("Generations")
        ax.set_ylabel("Frequency")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(generations, rotation=45)
        ax.legend()
        ax.grid(axis='y')
        self.save_plot(fig, "operation_frequencies.png")

    def plot_exploration_vs_exploitation(self, exploration_counts, exploitation_counts):
        """
        Plots the comparison between exploration and exploitation over time.

        Args:
            exploration_counts (list): Number of new solutions generated per generation.
            exploitation_counts (list): Number of solutions locally improved per generation.
        """
        generations = self.history["generation"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, exploration_counts, label="Exploration (New Solutions)", linewidth=2, color="green")
        ax.plot(generations, exploitation_counts, label="Exploitation (Local Improvement)", linewidth=2, linestyle="--", color="red")
        ax.set_title("Exploration vs Exploitation")
        ax.set_xlabel("Generations")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid()
        self.save_plot(fig, "exploration_vs_exploitation.png")

    def save_all_plots(self, strategy_usage, exploration_counts=None, exploitation_counts=None):
        """
        Generates and saves all relevant plots.

        Args:
            strategy_usage (dict): Usage of strategies per operation and generation.
            exploration_counts (list, optional): Number of new solutions generated per generation.
            exploitation_counts (list, optional): Number of solutions locally improved per generation.
        """
        self.plot_best_fitness()
        self.plot_average_fitness()
        self.plot_fitness()
        self.plot_diversity()
        self.plot_complexity()
        self.plot_operation_frequencies(strategy_usage)
        if exploration_counts and exploitation_counts:
            self.plot_exploration_vs_exploitation(exploration_counts, exploitation_counts)
