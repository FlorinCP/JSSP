import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from genetic_algo import GeneticAlgorithm
from models import JobShopProblem


class ParameterAnalyzer:
    """
    Handles running and analyzing genetic algorithm experiments with different parameters.
    This class helps organize multiple runs and collect meaningful metrics for comparison.
    """

    def __init__(self, input_file: str, output_dir: str = 'results'):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define parameter combinations to test
        self.parameter_combinations = [
            {
                "name": "baseline",
                "population_size": 100,
                "generations": 100,
                "tournament_size": 5,
                "mutation_rate": 0.1
            },
            {
                "name": "larger_population",
                "population_size": 200,
                "generations": 100,
                "tournament_size": 5,
                "mutation_rate": 0.15
            },
            {
                "name": "more_generations",
                "population_size": 200,
                "generations": 150,
                "tournament_size": 7,
                "mutation_rate": 0.1
            },
            {
                "name": "aggressive_mutation",
                "population_size": 300,
                "generations": 150,
                "tournament_size": 7,
                "mutation_rate": 0.2
            },
            {
                "name": "large_tournament",
                "population_size": 300,
                "generations": 200,
                "tournament_size": 10,
                "mutation_rate": 0.1
            },
            {
                "name": "balanced_large",
                "population_size": 300,
                "generations": 200,
                "tournament_size": 10,
                "mutation_rate": 0.15
            }
        ]

    def run_instance_analysis(self, instance_name: str) -> Dict:
        """
        Runs the genetic algorithm with all parameter combinations for one instance.
        Collects and organizes the results for comparison.
        """
        instance_results = {
            "instance_name": instance_name,
            "parameter_combinations": {}
        }

        for params in self.parameter_combinations:
            print(f"\nRunning {instance_name} with parameter set: {params['name']}")

            # Create and run GA with these parameters
            ga = GeneticAlgorithm(
                population_size=params["population_size"],
                generations=params["generations"],
                tournament_size=params["tournament_size"],
                mutation_rate=params["mutation_rate"]
            )

            # Load problem and run optimization
            problem = JobShopProblem()
            problem.load_from_file(self.input_file, instance_name)
            ga.problem = problem

            best_solution, history = ga.run()

            # Store results for this parameter combination
            instance_results["parameter_combinations"][params["name"]] = {
                "parameters": params,
                "stats": {
                    "best_fitness": float(best_solution.fitness),
                    "final_diversity": float(history["diversity"][-1]),
                    "total_generations": len(history["best_fitness"]),
                    "improvement_percentage": float(
                        (history["best_fitness"][0] - history["best_fitness"][-1])
                        / history["best_fitness"][0] * 100
                    ),
                    "convergence_generation": self._find_convergence_generation(
                        history["best_fitness"]
                    ),
                    "stagnant_generations": self._count_stagnant_generations(
                        history["best_fitness"]
                    )
                },
                "history": {
                    "best_fitness": [float(x) for x in history["best_fitness"]],
                    "diversity": [float(x) for x in history["diversity"]]
                }
            }

        return instance_results

    def _find_convergence_generation(self, fitness_history: List[float]) -> int:
        """
        Determines the generation where the algorithm effectively converged.
        Uses a sliding window to detect when improvements become minimal.
        """
        window_size = 20
        improvement_threshold = 0.001

        if len(fitness_history) < window_size:
            return 0

        for i in range(len(fitness_history) - window_size):
            window = fitness_history[i:i + window_size]
            improvement = (window[0] - window[-1]) / window[0]
            if improvement < improvement_threshold:
                return i

        return 0

    def _count_stagnant_generations(self, fitness_history: List[float]) -> int:
        """
        Counts the number of consecutive generations without improvement at the end.
        """
        count = 0
        final_fitness = fitness_history[-1]

        for fitness in reversed(fitness_history):
            if fitness == final_fitness:
                count += 1
            else:
                break

        return count

    def analyze_results(self, instance_results: Dict) -> Dict:
        """
        Analyzes the results across parameter combinations to identify best settings.
        """
        analysis = {
            "instance_name": instance_results["instance_name"],
            "best_parameter_set": None,
            "best_fitness": float('inf'),
            "parameter_comparison": []
        }

        for param_name, results in instance_results["parameter_combinations"].items():
            fitness = results["stats"]["best_fitness"]

            # Track best parameters
            if fitness < analysis["best_fitness"]:
                analysis["best_fitness"] = fitness
                analysis["best_parameter_set"] = param_name

            # Collect comparison data
            analysis["parameter_comparison"].append({
                "parameter_set": param_name,
                "fitness": fitness,
                "improvement": results["stats"]["improvement_percentage"],
                "convergence_gen": results["stats"]["convergence_generation"],
                "total_gens": results["stats"]["total_generations"]
            })

        return analysis

    def save_analysis(self, instance_results: Dict):
        """
        Saves the analysis results in a structured format.
        Creates both detailed JSON output and a summary CSV.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        instance_name = instance_results["instance_name"]

        # Save detailed results
        results_file = self.output_dir / f'{instance_name}_analysis_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(instance_results, f, indent=2)

        # Create and save summary
        summary_data = []
        for param_name, results in instance_results["parameter_combinations"].items():
            summary_data.append({
                "Parameter Set": param_name,
                "Best Fitness": results["stats"]["best_fitness"],
                "Improvement %": results["stats"]["improvement_percentage"],
                "Generations": results["stats"]["total_generations"],
                "Convergence Gen": results["stats"]["convergence_generation"],
                "Final Diversity": results["stats"]["final_diversity"]
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / f'{instance_name}_summary_{timestamp}.csv'
        summary_df.to_csv(summary_file, index=False)

        print(f"\nResults saved to:")
        print(f"Detailed analysis: {results_file}")
        print(f"Summary: {summary_file}")

        return summary_df