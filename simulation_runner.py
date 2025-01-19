from pathlib import Path
import json

import pandas as pd
from typing import Dict, Optional

from constants import parameter_sets
from genetic_algo import GeneticAlgorithm
from jsp_parser import JSPParser
from models import JobShopProblem
from utils_models import GAStatisticsAnalyzer, GAResultSerializer
from visualization.plots import plot_fitness_evolution, plot_schedule, plot_population_diversity
from datetime import datetime


def _create_parameter_comparison(results: Dict, output_dir: Path):
    """
    Create and save comparative analysis of different parameter sets.
    """
    comparison_data = []

    for param_name, param_results in results["parameter_sets"].items():
        comparison_data.append({
            "Parameter Set": param_name,
            "Best Fitness": param_results["stats"]["best_fitness"],
            "Improvement %": param_results["stats"]["improvement_percentage"],
            "Convergence Gen": param_results["stats"]["convergence_generation"],
            "Final Diversity": param_results["stats"]["final_diversity"],
            "Population Size": param_results["parameters"]["population_size"],
            "Executions": param_results["stats"]["executions"],
            "Generations": param_results["parameters"]["generations"],
            "Mutation Rate": param_results["parameters"]["mutation_rate"]
        })

    # Create comparison DataFrame and save to CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / 'parameter_comparison.csv', index=False)

    # Print comparison summary
    print("\nParameter Comparison Summary:")
    print("=" * 80)
    print(comparison_df.to_string(index=False))


def plot_individual_charts(history, best_solution, instance_dir, name = None):
    if name :
        param_dir = instance_dir / name
        param_dir.mkdir(parents=True, exist_ok=True)
    else:
        param_dir = instance_dir

    # Save evolution plot
    fitness_fig = plot_fitness_evolution(history)
    fitness_fig.savefig(
        param_dir / 'fitness_evolution.png',
        dpi=300,
        bbox_inches='tight'
    )

    # Save schedule plot
    schedule_fig = plot_schedule(best_solution)
    schedule_fig.savefig(
        param_dir / 'best_schedule.png',
        dpi=300,
        bbox_inches='tight'
    )

    # Save diversity plot
    diversity_fig = plot_population_diversity(history)
    diversity_fig.savefig(
        param_dir / 'population_diversity.png',
        dpi=300,
        bbox_inches='tight'
    )


class SimulationRunner:

    def __init__(self,
                 input_file: str,
                 output_dir: str = 'results',
                 population_size: int = 100,
                 generations: int = 100,
                 elite_size: int = 2,
                 tournament_size: int = 5,
                 max_instances: int = None,
                 visualize: bool = False,
                 mutation_rate: float = 0.1):

        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_instances = max_instances
        self.visualize = visualize

        self.ga_params = {
            'population_size': population_size,
            'generations': generations,
            'elite_size': elite_size,
            'tournament_size': tournament_size,
            'mutation_rate': mutation_rate
        }

        self.results = {}

    def run(self, instance_name: Optional[str] = None, use_multiple_params: bool = False) -> Dict:
        """
        Unified method to run simulations with given configuration.

        Args:
            instance_name: Optional specific instance to run. If None, runs all instances.
            use_multiple_params: Whether to use multiple parameter sets for optimization.

        Returns:
            Dict containing the results of the simulation(s).
        """
        if instance_name:
            results = self._run_instance(instance_name, use_multiple_params)
            results_dict = {instance_name: results}
        else:
            results_dict = self._run_all_instances(use_multiple_params)

        self.save_results(results_dict)
        return results_dict

    def _run_instance(self, instance_name: str, use_multiple_params: bool) -> Dict:
        """Internal method to run a single instance."""
        if use_multiple_params:
            return self.run_multiple_parameter_sets(instance_name)
        return self.run_single_instance(instance_name)

    def _run_all_instances(self, use_multiple_params: bool) -> Dict:
        """Internal method to run all instances."""
        with open(self.input_file, 'r') as f:
            content = f.read()

        instances = JSPParser.parse_file(content)

        if self.max_instances:
            instances = dict(list(instances.items())[:self.max_instances])

        results = {}
        for instance in instances:
            try:
                results[instance] = self._run_instance(instance, use_multiple_params)
            except Exception as e:
                print(f"Error processing instance {instance}: {str(e)}")
                results[instance] = {'error': str(e)}

        return results

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

        if self.visualize:
            plot_individual_charts(history, best_solution,instance_dir, instance_name)

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

    def save_results(self, results: Dict):
        """
        Save experiment results to files with proper type conversion.
        Handles both single instance and multiple parameter results.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        serializer = GAResultSerializer()
        serializable_results = serializer.prepare_results(results)

        # Save full results as JSON
        with open(self.output_dir / f'results_{timestamp}.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)

        summary_data = []

        # Handle parameter comparison results
        if "parameter_sets" in results:
            for param_name, param_result in results["parameter_sets"].items():
                summary_data.append({
                    'Parameter Set': param_name,
                    'Best Fitness': float(param_result["stats"]["best_fitness"]),
                    'Improvement %': float(param_result["stats"]["improvement_percentage"]),
                    'Convergence Gen': int(param_result["stats"]["convergence_generation"])
                    if param_result["stats"]["convergence_generation"] is not None else None,
                    'Final Diversity': float(param_result["stats"]["final_diversity"]),
                    'Population Size': int(param_result["parameters"]["population_size"]),
                    'Generations': int(param_result["parameters"]["generations"]),
                    'Mutation Rate': float(param_result["parameters"]["mutation_rate"])
                })
        # Handle single instance results
        elif isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
            for instance_name, result in results.items():
                if 'error' in result or 'stats' not in result:
                    continue

                summary_data.append({
                    'Instance': instance_name,
                    'Jobs': int(result['problem_info']['num_jobs']),
                    'Machines': int(result['problem_info']['num_machines']),
                    'Best Makespan': float(result['stats']['best_fitness']),
                    'Improvement %': float(result['stats']['improvement_percentage']),
                    'Convergence Gen': int(result['stats']['convergence_generation'])
                    if result['stats']['convergence_generation'] is not None else None,
                    'Final Diversity': float(result['stats']['final_diversity'])
                })

        # Create summary DataFrame and save to CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.output_dir / f'summary_{timestamp}.csv', index=False)

            # Print summary
            print("\nExperiment Summary:")
            print("=" * 80)
            print(summary_df.to_string(index=False))
            print("\nResults saved to:", self.output_dir)

    def run_multiple_parameter_sets(self, instance_name: str) -> Dict:
        """
        Run GA on a single instance with multiple parameter combinations.
        Returns results for each parameter set for comparison.
        """
        print(f"\nRunning multiple parameter sets on instance: {instance_name}")


        problem = JobShopProblem()
        problem.load_from_file(self.input_file, instance_name)

        results = {
            "instance_name": instance_name,
            "problem_info": {
                "num_jobs": problem.num_jobs,
                "num_machines": problem.num_machines,
                "description": problem.instance_description
            },
            "parameter_sets": {}
        }

        # Create directory for this instance's results
        instance_dir = self.output_dir / instance_name
        instance_dir.mkdir(parents=True, exist_ok=True)

        # Run GA with each parameter set
        for params in parameter_sets:
            print(f"\nRunning with parameter set: {params['name']}")

            # Configure GA with current parameter set
            ga = GeneticAlgorithm(
                population_size=params["population_size"],
                generations=params["generations"],
                tournament_size=params["tournament_size"],
                mutation_rate=params["mutation_rate"],
                elite_size=params["elite_size"]
            )
            ga.problem = problem

            # Run optimization
            best_solution, history = ga.run()
            stats = GAStatisticsAnalyzer.calculate_statistics(history, best_solution)

            # Save parameter set results
            results["parameter_sets"][params["name"]] = {
                "parameters": params,
                "stats": stats,
                "history": {
                    "best_fitness": history["best_fitness"],
                    "diversity": history["diversity"]
                }
            }

            # Save plots
            if self.visualize:
                plot_individual_charts(history, best_solution, instance_dir, params["name"])

            print(f"Completed {params['name']} - Best makespan: {stats['best_fitness']}")

        _create_parameter_comparison(results, instance_dir)

        return results


