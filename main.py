import argparse
from pathlib import Path
import json

import pandas as pd
from typing import Dict

from genetic_algo import GeneticAlgorithm
from jsp_parser import JSPParser
from models import JobShopProblem
from utils_models import GAStatisticsAnalyzer, GAResultSerializer
from visualization.plots import plot_fitness_evolution, plot_schedule
from datetime import datetime


class ExperimentRunner:
    """Handles running experiments on Job Shop Problem instances."""

    def __init__(self,
                 input_file: str,
                 output_dir: str = 'results',
                 population_size: int = 100,
                 generations: int = 100,
                 elite_size: int = 2,
                 tournament_size: int = 5,
                 max_instances: int = None,
                 mutation_rate: float = 0.1):
        """
        Initialize the experiment runner with GA parameters.

        Args:
            input_file: Path to the JSP instances file
            output_dir: Directory to save results
            population_size: Size of GA population
            generations: Number of generations to run
            elite_size: Number of elite solutions to preserve
            tournament_size: Size of tournament for selection
            mutation_rate: Probability of mutation
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_instances = max_instances

        self.ga_params = {
            'population_size': population_size,
            'generations': generations,
            'elite_size': elite_size,
            'tournament_size': tournament_size,
            'mutation_rate': mutation_rate
        }

        self.results = {}

    def run_single_instance(self, instance_name: str) -> Dict:
        """Run GA on a single instance and return results."""
        print(f"\nProcessing instance: {instance_name}")

        problem = JobShopProblem()
        problem.load_from_file(self.input_file, instance_name)

        ga = GeneticAlgorithm(**self.ga_params)
        ga.problem = problem
        best_solution, history = ga.run()

        stats = GAStatisticsAnalyzer.calculate_statistics(history, best_solution)

        instance_dir = self.output_dir / instance_name
        instance_dir.mkdir(parents=True, exist_ok=True)

        fitness_fig = plot_fitness_evolution(history)
        fitness_fig.savefig(
            instance_dir / 'fitness_evolution.png',
            dpi=300,
            bbox_inches='tight'
        )

        schedule_fig = plot_schedule(best_solution)
        schedule_fig.savefig(
            instance_dir / 'best_schedule.png',
            dpi=300,
            bbox_inches='tight'
        )

        return {
            'stats': stats,
            'history': {
                'best_fitness': history['best_fitness'],
                'diversity': history['diversity']
            },
            'problem_info': {
                'num_jobs': problem.num_jobs,
                'num_machines': problem.num_machines,
                'description': problem.instance_description
            }
        }

    def run_all_instances(self) -> Dict[str, Dict]:
        """Run genetic algorithm optimization on job shop problem instances."""
        print("\nAnalyzing input file structure...")
        with open(self.input_file, 'r') as f:
            content = f.read()

        print("\nParsing instances from file...")
        instances = JSPParser.parse_file(content)

        if not instances:
            print("\nNo valid instances were found in the file. Please check the file format.")
            return {}

        instance_names = list(instances.keys())
        if self.max_instances is not None:
            instance_names = instance_names[:self.max_instances]
            print(f"\nProcessing first {self.max_instances} instances:")
        else:
            print("\nProcessing all instances:")

        for name in instance_names:
            print(f"  - {name}")

        results = {}
        total_instances = len(instance_names)

        for idx, instance_name in enumerate(instance_names, 1):
            try:
                print(f"\nProcessing instance {idx}/{total_instances}: {instance_name}")
                results[instance_name] = self.run_single_instance(instance_name)

                # Print success message with achieved fitness
                best_fitness = results[instance_name]['stats']['best_fitness']
                print(f"✓ Successfully optimized {instance_name} - Best makespan: {best_fitness}")

            except Exception as e:
                print(f"✗ Error processing {instance_name}: {str(e)}")
                results[instance_name] = {'error': str(e)}

        successful_runs = sum(1 for r in results.values() if 'error' not in r)
        print(f"\nCompleted processing {total_instances} instances:")
        print(f"  - Successful: {successful_runs}")
        print(f"  - Failed: {total_instances - successful_runs}")

        return results

    def save_results(self, results: Dict):
        """Save experiment results to files with proper type conversion."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert results to JSON-serializable format
        serializer = GAResultSerializer()
        serializable_results = serializer.prepare_results(results)

        # Save full results as JSON
        with open(self.output_dir / f'results_{timestamp}.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # Create summary DataFrame
        summary_data = []
        for instance_name, result in results.items():
            if 'error' in result:
                continue

            summary_data.append({
                'Instance': instance_name,
                'Jobs': int(result['problem_info']['num_jobs']),  # Convert np.int64 to int
                'Machines': int(result['problem_info']['num_machines']),
                'Best Makespan': float(result['stats']['best_fitness']),  # Convert np.float64 to float
                'Improvement %': float(result['stats']['improvement_percentage']),
                'Convergence Gen': int(result['stats']['convergence_generation']) if result['stats'][
                                                                                         'convergence_generation'] is not None else None,
                'Final Diversity': float(result['stats']['final_diversity'])
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / f'summary_{timestamp}.csv', index=False)

        # Print summary
        print("\nExperiment Summary:")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("\nResults saved to:", self.output_dir)


def main():
    parser = argparse.ArgumentParser(description='Run Job Shop Problem experiments')
    parser.add_argument('input_file', help='Path to JSP instances file')
    parser.add_argument('--instance', help='Specific instance to run (optional)')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--population-size', type=int, default=100)
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--elite-size', type=int, default=2)
    parser.add_argument('--tournament-size', type=int, default=5)
    parser.add_argument('--mutation-rate', type=float, default=0.1)
    parser.add_argument('--max-instances', type=int,
                        help='Maximum number of instances to process (default: all)')

    args = parser.parse_args()

    runner = ExperimentRunner(
        input_file=args.input_file,
        output_dir=args.output_dir,
        population_size=args.population_size,
        generations=args.generations,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        mutation_rate=args.mutation_rate,
        max_instances=args.max_instances
    )

    if args.instance:
        results = {args.instance: runner.run_single_instance(args.instance)}
    else:
        results = runner.run_all_instances()

    runner.save_results(results)


if __name__ == "__main__":
    main()
