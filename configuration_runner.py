from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

from constants import OUTPUT_DIR, DEFAULT_POPULATION_SIZE, DEFAULT_GENERATIONS, DEFAULT_ELITE_SIZE, \
    DEFAULT_TOURNAMENT_SIZE, DEFAULT_MUTATION_RATE
from genetic_algo import GeneticAlgorithm
from models import JobShopProblem
from utils_models import GAStatisticsAnalyzer, GAResultSerializer
from visualization.plots import plot_fitness_evolution, plot_schedule, plot_population_diversity


class ConfigurationRunner:
    """
    A runner class that manages Job Shop Problem experiments with different genetic algorithm configurations.
    Handles multiple operator combinations and provides organized result storage and analysis.
    """

    def __init__(
            self,
            input_file: str,
            instance_name: str,
            output_dir: str = OUTPUT_DIR,
            population_size: int = DEFAULT_POPULATION_SIZE,
            generations: int = DEFAULT_GENERATIONS,
            elite_size: int = DEFAULT_ELITE_SIZE,
            tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
            mutation_rate: float = DEFAULT_MUTATION_RATE
    ):
        self.input_file = input_file
        self.instance_name = instance_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_params = {
            'population_size': population_size,
            'generations': generations,
            'elite_size': elite_size,
            'tournament_size': tournament_size,
            'mutation_rate': mutation_rate
        }

        self.operators = {
            'selection': ['tournament', 'roulette'],
            'crossover': ['ppx', 'simple'],
            'mutation': ['swap', 'inversion']
        }

        self.problem = self._initialize_problem()

    def _initialize_problem(self) -> JobShopProblem:
        problem = JobShopProblem()
        problem.load_from_file(self.input_file, self.instance_name)
        return problem

    def _create_operator_combination_name(self, sel: str, cross: str, mut: str) -> str:
        """Create a standardized name for an operator combination."""
        return f"{sel}_{cross}_{mut}"

    def _get_combination_dir(self, combination_name: str) -> Path:
        """Get the directory path for a specific combination."""
        return self.output_dir / combination_name / self.instance_name

    def run_single_configuration(self, selection: str, crossover: str, mutation: str) -> Dict:
        """
        Run the genetic algorithm with a specific combination of operators.
        Returns the results including statistics and history.
        """

        ga_params = self.base_params.copy()
        ga_params.update({
            'selection_method': selection,
            'crossover_method': crossover,
            'mutation_method': mutation
        })

        ga = GeneticAlgorithm(**ga_params)
        ga.problem = self.problem
        best_solution, history = ga.run()

        # Analyze results
        stats = GAStatisticsAnalyzer.calculate_statistics(history, best_solution)

        # Create combination directory and save visualizations
        combination_name = self._create_operator_combination_name(selection, crossover, mutation)
        combination_dir = self._get_combination_dir(combination_name)
        combination_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save plots
        plots = {
            'fitness_evolution.png': plot_fitness_evolution(history),
            'best_schedule.png': plot_schedule(best_solution),
            'population_diversity.png': plot_population_diversity(history)
        }

        for filename, fig in plots.items():
            fig.savefig(
                combination_dir / filename,
                dpi=300,
                bbox_inches='tight'
            )

        return {
            'configuration': {
                'selection': selection,
                'crossover': crossover,
                'mutation': mutation
            },
            'stats': stats,
            'history': {
                'best_fitness': history['best_fitness'],
                'diversity': history['diversity']
            },
            'problem_info': {
                'num_jobs': self.problem.num_jobs,
                'num_machines': self.problem.num_machines,
                'description': self.problem.instance_description
            }
        }

    def run_all_combinations(self) -> Dict[str, Dict]:
        """
        Run the genetic algorithm for all possible operator combinations.
        Returns a dictionary with results for each combination.
        """
        results = {}

        for selection in self.operators['selection']:
            for crossover in self.operators['crossover']:
                for mutation in self.operators['mutation']:
                    combination_name = self._create_operator_combination_name(
                        selection, crossover, mutation
                    )
                    print(f"\nTesting combination: {combination_name}")

                    results[combination_name] = self.run_single_configuration(
                        selection, crossover, mutation
                    )

        return results

    def save_results(self, results: Dict):
        """
        Save experiment results with proper formatting and organization.
        Creates both JSON and CSV summaries for easy analysis.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results as JSON
        serializer = GAResultSerializer()
        serializable_results = serializer.prepare_results(results)

        with open(self.output_dir / f'detailed_results_{timestamp}.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)

        summary_data = []
        for combo_name, result in results.items():
            summary_data.append({
                'Combination': combo_name,
                'Selection': result['configuration']['selection'],
                'Crossover': result['configuration']['crossover'],
                'Mutation': result['configuration']['mutation'],
                'Best Makespan': float(result['stats']['best_fitness']),
            })

        # Save summary as CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / f'summary_{timestamp}.csv', index=False)

        print("\nExperiment Summary:")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("\nResults saved to:", self.output_dir)
