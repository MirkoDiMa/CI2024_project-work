from datetime import datetime
from pathlib import Path
from utility.utils import initialize_experiment, load_data, run_genetic_programming, save_results
from core.statistics import GPStatistics

def main():
    data_dir = '../data'
    output_file = './s333962.py'
    base_output_dir = './experiments/'
    data_files = sorted(Path(data_dir).glob('*.npz'))

    if not data_files:
        print("No data files found in the directory.")
        return

    for data_file in data_files:
        experiment_successful = True
        start_time = datetime.now()
        reason = "Max generations reached"

        stats = None
        best_individual = None

        try:
            # Initialize the experiment
            experiment_config = initialize_experiment(data_file, base_output_dir)
            logger = experiment_config["logger"]

            logger.info(f"Processing Problem {experiment_config['problem_id']}")

            # Load the data
            x, y = load_data(data_file)

            # Run GP and get the results
            best_individual, stats = run_genetic_programming(x, y, logger)

            # Save the results (formula, plots, logs)
            save_results(
                best_individual=best_individual,
                stats=stats,
                output_file=output_file,
                function_name=f"f{experiment_config['problem_id']}",  # Function name based on problem ID
                plot_dir=experiment_config["plot_dir"],
            )

        except Exception as e:
            experiment_successful = False
            reason = f"Error: {str(e)}"
            if 'logger' in locals():
                logger.info(f"Error processing Problem {experiment_config['problem_id']}: {reason}")
            print(f"Error processing Problem {experiment_config['problem_id']}: {reason}")

        finally:
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()

            # Generate a summary, even if there were errors
            if stats is not None:
                logger.generate_summary(
                    stats=stats,
                    best_expression=best_individual.tree_to_expression() if best_individual else "N/A",
                    total_time=total_time,
                    start_time=start_time,
                    end_time=end_time,
                    reason=reason,
                    success=experiment_successful
                )
            else:
                logger.log_message("No GPStatistics available. Possibly an error occurred.")

    print("All experiments completed.")


if __name__ == "__main__":
    main()
