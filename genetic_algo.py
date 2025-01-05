from typing import Tuple, Dict

import numpy as np

from genetic_operators import smart_crossover, swap_mutation
from models import JobShopProblem, JobShopChromosome


class GeneticAlgorithm:
    def __init__(self,
                 population_size: int = 100,
                 generations: int = 100,
                 elite_size: int = 2,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.1):
        self.problem = JobShopProblem()
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.history = {
            'best_fitness': [],
            'worst_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'best_solutions': []
        }

    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        for _ in range(self.population_size):
            chromosome = JobShopChromosome(self.problem)
            self.population.append(chromosome)

    def tournament_selection(self) -> JobShopChromosome:
        """Select parent using tournament selection.

        Returns:
            JobShopChromosome: The selected parent chromosome
        """
        # Randomly select tournament_size chromosomes from the population
        tournament = np.random.choice(
            self.population,
            size=self.tournament_size,
            replace=False  # Don't select the same chromosome twice
        )

        # Return the chromosome with the best fitness from the tournament
        return min(tournament, key=lambda x: x.fitness)

    def calculate_diversity(self) -> float:
        """Calculate population diversity using average chromosome difference."""
        total_diff = 0
        comparisons = 0
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                diff = sum(a != b for a, b in zip(
                    self.population[i].chromosome,
                    self.population[j].chromosome
                ))
                total_diff += diff
                comparisons += 1
        return total_diff / comparisons if comparisons > 0 else 0

    def run(self) -> Tuple[JobShopChromosome, Dict]:
        """Run the genetic algorithm with enhanced tracking.

        Returns:
            Tuple[JobShopChromosome, Dict]: Best solution and history dictionary
        """
        print("Initializing population...")
        self.initialize_population()

        for chromosome in self.population:
            chromosome.schedule = chromosome.decode_to_schedule()

        for generation in range(self.generations):
            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness)

            # Update history
            best_fitness = self.population[0].fitness
            worst_fitness = self.population[-1].fitness
            avg_fitness = np.mean([chr.fitness for chr in self.population])
            diversity = self.calculate_diversity()

            self.history['best_fitness'].append(best_fitness)
            self.history['worst_fitness'].append(worst_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['diversity'].append(diversity)
            self.history['best_solutions'].append(self.population[0])

            if generation % 10 == 0:
                print(f"Generation {generation}: Best={best_fitness:.2f}, "
                      f"Avg={avg_fitness:.2f}, Diversity={diversity:.2f}")

            # Create new population
            new_population = []
            new_population.extend(self.population[:self.elite_size])

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child_sequence = smart_crossover(parent1.chromosome,
                                                 parent2.chromosome,
                                                 self.problem)
                child_sequence = swap_mutation(child_sequence, self.mutation_rate)

                child = JobShopChromosome(self.problem)
                child.chromosome = child_sequence
                child.schedule = child.decode_to_schedule()
                new_population.append(child)

            self.population = new_population

        self.population.sort(key=lambda x: x.fitness)
        return self.population[0], self.history