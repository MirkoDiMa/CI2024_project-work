# ==============================================
# Function to save the best formula to a file
# ==============================================
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

from core.evaluator import Evaluator
from memetic.evolution import GeneticProgramming
from core.statistics import GPStatistics
from memetic_config import BLOAT_PENALTY, N_GENERATIONS, SEED
from utility.logger import Logger
from utility.plotting import Plotter

def initialize_experiment(data_file, base_output_dir):
    """
    Initializes paths, the logger, and directories for an experiment.

    Args:
        data_file (Path): Path to the data file.
        base_output_dir (str): Base output directory.

    Returns:
        dict: Contains the initial configurations for the experiment.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    # Extract the problem ID from the last part of the file name
    problem_id = data_file.stem.split('_')[-1]
    problem_dir = Path(base_output_dir) / f"problem_{problem_id}"
    log_dir = problem_dir / "logs"
    plot_dir = problem_dir / "plots"

    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(
        log_dir=str(log_dir),
        log_file_prefix=f"p_{problem_id}"
    )

    return {
        "problem_id": problem_id,
        "problem_dir": problem_dir,
        "log_dir": log_dir,
        "plot_dir": plot_dir,
        "logger": logger
    }

def load_data(data_file):
    """
    Loads data from the specified file and reduces the sample count if necessary.

    Args:
        data_file (Path): Path to the data file.

    Returns:
        tuple: Loaded and reduced X and Y data.
    """
    data = np.load(data_file)
    x, y = data['x'], data['y']
    
    if x.shape[0] > x.shape[1]:
        x = x.T

    return x, y

def run_genetic_programming(x, y, logger):
    """
    Runs the genetic programming algorithm.

    Args:
        x (np.ndarray): Input data.
        y (np.ndarray): Output data.
        logger (Logger): Logger to track the process.

    Returns:
        tuple: The best individual and GP statistics.
    """
    # Instantiate GPStatistics, passing the logger to track strategy changes
    stats = GPStatistics(logger)
    logger.info("Initializing Genetic Programming.")
    
    gp = GeneticProgramming(
        n_features=x.shape[1],
        generations=N_GENERATIONS,
        bloat_penalty=BLOAT_PENALTY,
        stats=stats
    )

    with tqdm(total=N_GENERATIONS, desc="Genetic Programming", unit="gen") as pbar:
        gp.progress_bar = pbar
        best_individual = gp.run(x, y)

    return best_individual, stats

def save_results(best_individual, stats, output_file, function_name, plot_dir):
    """
    Saves the experiment results.

    Args:
        best_individual (Node): The best individual.
        stats (GPStatistics): GP statistics.
        output_file (str): File to save the formula.
        function_name (str): Name of the function to update.
        plot_dir (Path): Directory for plots.
    """
    evaluator = Evaluator()
    best_expression = best_individual.tree_to_expression()
    update_formula_in_file(
        formula_str=best_expression,
        file_path=output_file,
        function_name=function_name
    )

    # Generate and save all plots using the Plotter
    plotter = Plotter(plot_dir=str(plot_dir), plot_dir_prefix=function_name, history=stats.history)
    plotter.save_all_plots(strategy_usage=stats.strategy_usage)

def update_formula_in_file(formula_str, file_path, function_name):
    """
    Overwrites the `function_name` function in `file_path`
    with `return formula_str`, ensuring the correct NumPy format.

    Args:
        formula_str (str): Formula string to insert.
        file_path (str): Path to the Python file.
        function_name (str): Name of the function to update.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    inside_function = False
    for line in lines:
        if line.strip().startswith(f"def {function_name}"):
            inside_function = True
            new_lines.append(f"def {function_name}(x: np.ndarray) -> np.ndarray:\n")
            new_lines.append(f"    return {formula_str}\n")
            continue
        if inside_function:
            if line.strip() == "" or line.strip().startswith("def "):
                inside_function = False
        if not inside_function:
            new_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(new_lines)

    print(f"Formula updated in {file_path} within the function {function_name}.")
