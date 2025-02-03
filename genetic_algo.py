from typing import Tuple, Dict, List
from copy import deepcopy
import numpy as np

from constants import MAX_STAGNANT_GENERATIONS_STOP, STARTING_GENERATION_TO_CHECK_STAGNATION
from genetic_operators import (
    swap_mutation, inversion_mutation,
    tournament_selection, roulette_wheel_selection, ppx_crossover, jox_crossover)
from models import JobShopProblem, JobShopChromosome


class GeneticAlgorithm:
    def __init__(self,
                 population_size: int = 100,
                 generations: int = 100,
                 elite_size: int = 2,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.1,
                 selection_method: str = 'tournament',
                 crossover_method: str = 'ppx',
                 mutation_method: str = 'swap'):

        self.problem = JobShopProblem()
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.population = []
        self.history = {
            'best_fitness': [],
            'worst_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'best_solutions': []
        }
        self.best_solution_ever = None
        self.best_fitness_ever = float('inf')

    def initialize_population(self):
        """
        Initialize population with improved reliability and error handling.
        This version ensures we can create the full population size while
        maintaining chromosome validity.
        """
        self.population = []
        attempts_per_chromosome = 5

        for _ in range(self.population_size):
            chromosome = None
            for attempt in range(attempts_per_chromosome):
                try:
                    new_chromosome = JobShopChromosome(self.problem)
                    if new_chromosome.validate_chromosome():
                        chromosome = new_chromosome
                        break
                except Exception as e:
                    continue

            self.population.append(chromosome)

    def calculate_diversity(self) -> float:
        """
        Calculate population diversity using average chromosome difference.

        This is essential in genetic algorithms to measure the diversity of the population.
        Diversity is crucial because it helps prevent premature convergence to suboptimal solutions,
        ensuring a wide range of genetic material is available for creating new solutions.
        This method calculates the average difference between all pairs of chromosomes in the population,
        providing a quantitative measure of how varied the population is.
        """

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

    def select_parent(self) -> JobShopChromosome:
        """Select parent using specified selection method."""

        if self.selection_method == 'tournament':
            return tournament_selection(self)
        elif self.selection_method == 'roulette':
            fitness_values = [chr.fitness for chr in self.population]
            return roulette_wheel_selection(self.population, fitness_values)

    def apply_mutation(self, chromosome: List[int]) -> List[int]:
        """Apply specified mutation method."""

        if self.mutation_method == 'swap':
            return swap_mutation(chromosome, self.mutation_rate)
        elif self.mutation_method == 'inversion':
            return inversion_mutation(chromosome, self.mutation_rate)

    def apply_crossover(self, parent1: JobShopChromosome, parent2: JobShopChromosome) -> List[int]:
        """Apply specified crossover method."""

        if self.crossover_method == 'ppx':
            return ppx_crossover(parent1.chromosome, parent2.chromosome, self.problem)
        elif self.crossover_method == 'jox':
            return jox_crossover(parent1.chromosome, parent2.chromosome, self.problem)

    def evaluate_population(self):
        """
        Evaluate all chromosomes in the population with improved error handling.
        """
        valid_chromosomes = []
        for chromosome in self.population:
            try:
                chromosome.schedule = chromosome.decode_to_schedule()
                valid_chromosomes.append(chromosome)
            except Exception as e:
                print(f"Warning: Failed to evaluate chromosome: {str(e)}")

        self.population = valid_chromosomes
        return [chr.fitness for chr in self.population]

    def update_history(self, generation: int):
        self.population.sort(key=lambda x: x.fitness)

        current_best = self.population[0].fitness
        current_worst = self.population[-1].fitness
        current_avg = np.mean([chr.fitness for chr in self.population])
        current_diversity = self.calculate_diversity()
        best_solution_copy = deepcopy(self.population[0])
        best_solution_copy.schedule = best_solution_copy.decode_to_schedule()

        # Update history
        self.history['best_fitness'].append(current_best)
        self.history['worst_fitness'].append(current_worst)
        self.history['avg_fitness'].append(current_avg)
        self.history['diversity'].append(current_diversity)
        self.history['best_solutions'].append(best_solution_copy)

        if generation % 10 == 0:
            print(f"\nGeneration {generation}:")
            print(f"Best Fitness: {current_best:.2f}")
            print(f"Worst Fitness: {current_worst:.2f}")
            print(f"Average Fitness: {current_avg:.2f}")
            print(f"Population Diversity: {current_diversity:.2f}")
            print(f"Best solution ever: {self.best_fitness_ever:.2f}")

    def check_early_stopping(self, generation: int) -> bool:
        """Check if the algorithm should stop early due to lack of improvement."""
        if generation > STARTING_GENERATION_TO_CHECK_STAGNATION:
            stagnant_generations = 0
            for i in range(-STARTING_GENERATION_TO_CHECK_STAGNATION, 0):
                if self.history['best_fitness'][i] == self.best_fitness_ever:
                    stagnant_generations += 1

            if stagnant_generations == MAX_STAGNANT_GENERATIONS_STOP:
                print(f"\nEarly stopping: No improvement for {MAX_STAGNANT_GENERATIONS_STOP} generations")
                return True
        return False

    def create_new_population(self, generation: int):
        new_population = []

        new_population.extend([deepcopy(chrom) for chrom in self.population[:self.elite_size]])

        while len(new_population) < self.population_size:

            parent1 = self.select_parent()
            parent2 = self.select_parent()

            child_sequence = self.apply_crossover(parent1, parent2)

            child_sequence = self.apply_mutation(child_sequence)

            child = JobShopChromosome(self.problem)
            child.chromosome = child_sequence
            child.schedule = child.decode_to_schedule()

            if child.fitness < self.best_fitness_ever:
                print(f"\nNew best solution found in generation {generation} with fitness: {child.fitness}")
                self.best_fitness_ever = child.fitness
                best_copy = deepcopy(child)
                best_copy.schedule = best_copy.decode_to_schedule()
                self.best_solution_ever = best_copy

            new_population.append(child)

        return new_population

    def run(self) -> Tuple[JobShopChromosome, Dict]:
        """Run the genetic algorithm with enhanced tracking."""

        self.initialize_population()
        initial_fitness = self.evaluate_population()
        print(f"Initial best fitness: {min(initial_fitness):.2f}")

        for generation in range(self.generations):

            self.update_history(generation)
            new_population = self.create_new_population(generation)
            self.population = new_population

            if self.check_early_stopping(generation):
                break

        return self.best_solution_ever, self.history
