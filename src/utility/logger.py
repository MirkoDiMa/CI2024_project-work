import logging
import os
import csv
from datetime import datetime
from memetic_config import POP_SIZE, MAX_DEPTH, N_GENERATIONS, TOURNAMENT_SIZE, MUTATION_RATE, CROSSOVER_RATE, ELITISM, BLOAT_PENALTY

class Logger:
    def __init__(self, log_dir="../experiments", log_file_prefix="gp_run"):
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{log_file_prefix}_{self.timestamp}_log.csv")
        self.general_log_file = os.path.abspath(
            os.path.join("..", "docs", f"general_log_{self.timestamp}.txt")
        )



        # Header for the metrics table
        self.metrics_fields = [
            "timestamp",
            "generation",
            "best_fitness",
            "average_fitness",
            "diversity",
            "complexity",
            "selection_strategy",
            "crossover_strategy",
            "mutation_strategy",
            "local_search_algorithm"
        ]
        # Header for the messages table (Algorithm strategies track)
        self.messages_fields = [
            "timestamp",
            "message"
        ]
        # Internal lists to accumulate logs
        self.metrics_logs = []
        self.messages_logs = []

        # Configure logging for console and general file
        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),  # For console output
                logging.FileHandler(self.general_log_file, mode='a')  # For general file output
            ]
        )
        self.logger = logging.getLogger("GPLogger")

    def log_metrics(self, generation=None, best_fitness=None, avg_fitness=None, diversity=None, complexity=None,
                    strategies=None, local_search=None):
        """Accumulates a row of metrics in the internal list."""
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "generation": generation if generation is not None else "",
            "best_fitness": f"{best_fitness:.4f}" if best_fitness is not None else "",
            "average_fitness": f"{avg_fitness:.4f}" if avg_fitness is not None else "",
            "diversity": f"{diversity:.4f}" if diversity is not None else "",
            "complexity": f"{complexity:.4f}" if complexity is not None else "",
            "selection_strategy": strategies.get("selection") if strategies else "",
            "crossover_strategy": strategies.get("crossover") if strategies else "",
            "mutation_strategy": strategies.get("mutation") if strategies else "",
            "local_search_algorithm": local_search if local_search else ""
        }
        self.metrics_logs.append(row)

    def log_message(self, message):
        """
        Accumulates a row of messages in the internal list.
        Args:
            message (str or list): Single message or list of messages (which will be concatenated with " | ").
        """
        if isinstance(message, list):
            message = " | ".join(message)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": message
        }
        self.messages_logs.append(row)

    def info(self, message, generation=None, best_fitness=None, avg_fitness=None, diversity=None, complexity=None,
             strategies=None, local_search=None):
        """
        Logs informational messages to the console and accumulates rows in the internal lists.
        Args:
            message (str or list): A single message or a list of messages.
        """
        if isinstance(message, list):
            message = " | ".join(message)
        # Log to console and general file
        self.logger.info(message)
        # If metrics are provided, log a row in the metrics section
        if any(v is not None for v in [generation, best_fitness, avg_fitness, diversity, complexity, strategies, local_search]):
            self.log_metrics(generation, best_fitness, avg_fitness, diversity, complexity, strategies, local_search)
        # Always log the message in the messages section
        self.log_message(message)

    def interpret_metric(self, value, thresholds):
        """Interpret a metric value based on thresholds."""
        if value <= thresholds[0]:
            return "Low"
        elif value <= thresholds[1]:
            return "Medium"
        else:
            return "High"

    def generate_summary(self, stats, best_expression, total_time, start_time, end_time, reason, success=True):
        """Generates a detailed summary of the experiment and adds this information to the internal lists."""
        strategy_usage = stats.get_strategy_usage()
        diversity_category = self.interpret_metric(stats.diversity, [0.5, 2])
        complexity_category = self.interpret_metric(stats.complexity, [5, 10])
        fitness_category = "Best" if stats.best_fitness == 0 else "Good" if stats.best_fitness < 0.5 else "Discrete" if stats.best_fitness < 1 else "Bad"

        stagnation_percentage = 0 if stats.current_generation == 0 else (stats.generations_no_improvement / stats.current_generation) * 100

        summary = (
            "\n==================== Experiment Summary ====================\n"
            "--- Experiment Configuration ---\n"
            f"Population Size: {POP_SIZE}\n"
            f"Max Depth: {MAX_DEPTH}\n"
            f"Max Generations: {N_GENERATIONS}\n"
            f"Mutation Rate: {MUTATION_RATE}\n"
            f"Crossover Rate: {CROSSOVER_RATE}\n"
            f"Elitism: {ELITISM}\n\n"
            "--- Experiment Status ---\n"
            f"Experiment started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Experiment finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Duration: {total_time:.2f} seconds\n"
            f"Status: {'Succeeded' if success else 'Failed'}\n"
            f"Reason: {reason}\n\n"
            "--- Experiment Statistics ---\n"
            f"Final Diversity: {stats.diversity:.4f} ({diversity_category})\n"
            f"Final Complexity: {stats.complexity:.4f} ({complexity_category})\n"
            f"Total Generations: {stats.current_generation}\n"
            f"Generations Without Improvement: {stats.generations_no_improvement} "
            f"({stagnation_percentage:.1f}%)\n"
            f"Best Fitness Achieved: {stats.best_fitness:.4f} ({fitness_category})\n"
            f"Best Expression: {best_expression}\n\n"
            "--- Strategy Usage ---\n"
            f"Selection Strategies: {strategy_usage['selection']}\n"
            f"Crossover Strategies: {strategy_usage['crossover']}\n"
            f"Mutation Strategies: {strategy_usage['mutation']}\n"
            f"Local Search Algorithms: {strategy_usage.get('local_search', {})}\n"
        )
        # Log summary to console and store the summary in messages
        self.logger.info(summary)
        self.log_message(summary)
        
        # Flush: writes the entire content (metrics and messages) to a single file
        self.flush_logs()

    def flush_logs(self):
        """
        Writes both tables: Metrics and Messages, to a single CSV file,
        separating them with header rows.
        """
        with open(self.log_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write the metrics table
            writer.writerow(["# Metrics"])
            writer.writerow(self.metrics_fields)
            for row in self.metrics_logs:
                writer.writerow([row[field] for field in self.metrics_fields])
            writer.writerow([])  # Empty row for separation
            # Write the messages table
            writer.writerow(["# Algorithm strategies track"])
            writer.writerow(self.messages_fields)
            for row in self.messages_logs:
                writer.writerow([row[field] for field in self.messages_fields])
