import numpy as np
from core.tree import Node

class Evaluator:
    """
    Evaluation module to compute the Mean Squared Error (MSE) and fitness
    of trees generated by Genetic Programming (GP).
    """

    @staticmethod
    def check_validity(array: np.ndarray) -> bool:
        """
        Checks if an array contains NaN, infinity, or values outside the allowed range.

        Args:
            array (np.ndarray): Array of values to validate.

        Returns:
            bool: True if the array is valid, False otherwise.
        """
        return not np.any(np.isnan(array) | np.isinf(array) | (array < -1e6) | (array > 1e6))

    @staticmethod
    def calculate_mse(tree, x, y):
        """
        Calculates the Mean Squared Error (MSE) between the tree's output and expected values.

        Args:
            tree (Node): The tree to evaluate.
            x (np.ndarray): Input array with features along the first dimension.
            y (np.ndarray): Expected output array.

        Returns:
            float: The MSE value.
        """
        try:
            y_pred = Node.evaluate_tree(tree, x)
            if not Evaluator.check_validity(y_pred):
                return float('inf')  # Penalize invalid output
            return np.mean((y - y_pred) ** 2)
        except Exception:
            return float('inf')  # Penalize errors during evaluation

    @staticmethod
    def fitness_function(tree, x, y, bloat_penalty):
        """
        Calculates the fitness function, combining MSE and a penalty for tree size (bloat).

        Args:
            tree (Node): The tree to evaluate.
            x (np.ndarray): Input array with features along the first dimension.
            y (np.ndarray): Expected output array.
            bloat_penalty (float): Penalty applied based on tree size.

        Returns:
            float: The fitness value.
        """
        mse = Evaluator.calculate_mse(tree, x, y)
        size = Node.tree_size(tree)
        penalty = bloat_penalty * size  # Apply penalty for large trees
        return mse + penalty

    @staticmethod
    def evaluate_population(population, x, y, bloat_penalty):
        """
        Evaluates a population of trees and returns their fitness values.

        Args:
            population (list[Node]): List of trees to evaluate.
            x (np.ndarray): Input array with features along the first dimension.
            y (np.ndarray): Expected output array.
            bloat_penalty (float): Penalty applied based on tree size.

        Returns:
            list[float]: List of fitness values for each tree.
        """
        fitness_values = []
        for tree in population:
            fitness = Evaluator.fitness_function(tree, x, y, bloat_penalty)
            fitness_values.append(fitness)
        return fitness_values

    @staticmethod
    def get_best_individual(population, x, y, bloat_penalty):
        """
        Returns the best individual (tree) in the population based on fitness.

        Args:
            population (list[Node]): List of trees to evaluate.
            x (np.ndarray): Input array with features along the first dimension.
            y (np.ndarray): Expected output array.
            bloat_penalty (float): Penalty applied based on tree size.

        Returns:
            tuple: The best tree and its fitness value.
        """
        fitness_values = Evaluator.evaluate_population(population, x, y, bloat_penalty)
        best_index = np.argmin(fitness_values)
        return population[best_index], fitness_values[best_index]
