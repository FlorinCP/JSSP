from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt

from genetic_algo import GeneticAlgorithm
from visualization.plots import (plot_fitness_evolution,
                                 plot_population_diversity,
                                 plot_schedule)
from models import JobShopChromosome, GAStatisticsAnalyzer


class JobShopSimulation:
    """Main simulation class for Job Shop Scheduling."""

    def __init__(self,
                 population_size: int = 100,
                 generations: int = 100,
                 elite_size: int = 2,
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
        print("\nGenerating visualizations...")

        # Plot and save fitness evolution
        print("Creating fitness evolution plot...")
        fitness_fig = plot_fitness_evolution(history)
        fitness_fig.savefig(
            self.output_dir / 'fitness_evolution.png',
            dpi=300,
            bbox_inches='tight'
        )

        # Plot and save diversity
        print("Creating diversity plot...")
        diversity_fig = plot_population_diversity(history)
        diversity_fig.savefig(
            self.output_dir / 'population_diversity.png',
            dpi=300,
            bbox_inches='tight'
        )

        # Plot and save schedule
        print("Creating schedule plot...")
        schedule_fig = plot_schedule(
            best_solution.schedule,
            best_solution.problem.num_machines
        )
        schedule_fig.savefig(
            self.output_dir / 'best_schedule.png',
            dpi=300,
            bbox_inches='tight'
        )

        # Show all plots
        plt.show()

    def run(self) -> Tuple[JobShopChromosome, Dict, Dict]:
        """Run the complete simulation with visualization and analysis."""
        try:
            print("Initializing Genetic Algorithm...")
            ga = GeneticAlgorithm(**self.ga_params)

            print("\nRunning optimization...")
            best_solution, history = ga.run()

            stats = GAStatisticsAnalyzer.calculate_statistics(history, best_solution)
            GAStatisticsAnalyzer.print_statistics(stats)

            self.create_and_save_plots(history, best_solution)

            return best_solution, history, stats

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
            raise
        except Exception as e:
            print(f"\nError during simulation: {str(e)}")
            raise


def main():
    """Main entry point for the simulation."""
    try:
        simulation = JobShopSimulation(
            population_size=100,
            generations=100,
            elite_size=2,
            tournament_size=5,
            mutation_rate=0.1,
            output_dir='results'
        )

        best_solution, history, stats = simulation.run()

    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
        exit(1)
    except Exception as e:
        print(f"\nSimulation failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()