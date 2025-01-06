from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from matplotlib import pyplot as plt

from genetic_algo import GeneticAlgorithm
from models import JobShopChromosome
from visualization.plots import plot_fitness_evolution, plot_population_diversity, plot_schedule


class GAStatisticsAnalyzer:
    """Analyzer for Genetic Algorithm statistics."""

    @staticmethod
    def calculate_statistics(history: Dict, best_solution: JobShopChromosome) -> Dict:
        """Calculate detailed statistics from the GA run."""
        stats = {
            'best_fitness': best_solution.fitness,
            'final_diversity': history['diversity'][-1],
            'total_improvement': history['best_fitness'][0] - history['best_fitness'][-1],
            'improvement_percentage': ((history['best_fitness'][0] - history['best_fitness'][-1]) /
                                       history['best_fitness'][0] * 100),
            'convergence_generation': None,
            'average_improvement_rate': np.mean(np.diff(history['best_fitness'])),
            'best_generation': np.argmin(history['best_fitness']),
            'stagnant_generations': 0,
            'final_schedule_makespan': max(
                details['end'] for details in best_solution.schedule.values()
            )
        }

        # Calculate convergence information
        improvements = np.abs(np.diff(history['best_fitness']))
        threshold = np.mean(improvements) * 0.01  # 1% of average improvement
        converged_gens = np.where(improvements < threshold)[0]
        if len(converged_gens) > 0:
            stats['convergence_generation'] = converged_gens[0]

        # Calculate stagnation information
        stagnant_count = 0
        for i in range(1, len(history['best_fitness'])):
            if abs(history['best_fitness'][i] - history['best_fitness'][i - 1]) < 1e-6:
                stagnant_count += 1
            else:
                stagnant_count = 0
            stats['stagnant_generations'] = max(
                stats['stagnant_generations'],
                stagnant_count
            )

        return stats

    @staticmethod
    def print_statistics(stats: Dict):
        """Print formatted statistics."""
        print("\n" + "=" * 50)
        print("FINAL STATISTICS")
        print("=" * 50)

        print("\nPerformance Metrics:")
        print(f"Best Fitness Achieved: {stats['best_fitness']:.2f}")
        print(f"Final Schedule Makespan: {stats['final_schedule_makespan']}")
        print(f"Total Improvement: {stats['total_improvement']:.2f} " +
              f"({stats['improvement_percentage']:.1f}%)")
        print(f"Average Improvement Rate: {stats['average_improvement_rate']:.3f} per generation")

        print("\nConvergence Analysis:")
        if stats['convergence_generation'] is not None:
            print(f"Convergence reached at generation: {stats['convergence_generation']}")
        else:
            print("Algorithm did not fully converge")
        print(f"Best solution found in generation: {stats['best_generation']}")
        print(f"Maximum stagnant generations: {stats['stagnant_generations']}")

        print("\nDiversity Metrics:")
        print(f"Final Population Diversity: {stats['final_diversity']:.2f}")

        print("\n" + "=" * 50)


class JobShopSimulation:
    """Main simulation class for Job Shop Scheduling."""

    def __init__(self,
                 population_size: int = 100,
                 generations: int = 100,
                 elite_size: int = 3,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.1,
                 output_dir: str = 'results'):
        self.ga_params = {
            'population_size': population_size,
            'generations': generations,
            'elite_size': elite_size,
            'tournament_size': tournament_size,
            'mutation_rate': mutation_rate
        }
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_and_save_plots(self, history: Dict, best_solution: JobShopChromosome):
        """Create and save all plots."""

        fitness_fig = plot_fitness_evolution(history)
        fitness_fig.savefig(
            self.output_dir / 'fitness_evolution.png',
            dpi=300,
            bbox_inches='tight'
        )

        diversity_fig = plot_population_diversity(history)
        diversity_fig.savefig(
            self.output_dir / 'population_diversity.png',
            dpi=300,
            bbox_inches='tight'
        )

        schedule_fig = plot_schedule(
            best_solution,
        )
        schedule_fig.savefig(
            self.output_dir / 'best_schedule.png',
            dpi=300,
            bbox_inches='tight'
        )

        plt.show()
        

    def run(self) -> Tuple[JobShopChromosome, Dict, Dict]:
        """Run the complete simulation with visualization and analysis."""
        try:
            print("Initializing Genetic Algorithm...")
            ga = GeneticAlgorithm(**self.ga_params)

            print("\nRunning optimization...")
            best_solution, history = ga.run()

            stats = GAStatisticsAnalyzer.calculate_statistics(history, best_solution)
            print(stats)
            GAStatisticsAnalyzer.print_statistics(stats)

            self.create_and_save_plots(history, best_solution)

            return best_solution, history, stats

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
            raise
        except Exception as e:
            print(f"\nError during simulation: {str(e)}")
            raise


class GAResultSerializer:
    """Handles serialization of Genetic Algorithm results to JSON-compatible format."""

    @staticmethod
    def convert_to_serializable(obj: Any) -> Any:
        """Convert various types to JSON-serializable formats."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: GAResultSerializer.convert_to_serializable(value)
                    for key, value in obj.items()}
        elif isinstance(obj, list):
            return [GAResultSerializer.convert_to_serializable(item)
                    for item in obj]
        return obj

    @staticmethod
    def prepare_results(results: Dict) -> Dict:
        """Process the entire results dictionary to ensure JSON compatibility."""
        return GAResultSerializer.convert_to_serializable(results)