from typing import Tuple, Dict, List
from copy import deepcopy
import numpy as np

from genetic_operators import (smart_crossover, order_crossover,
                               swap_mutation, inversion_mutation, scramble_mutation,
                               tournament_selection, roulette_wheel_selection)
from models import JobShopProblem, JobShopChromosome


class GeneticAlgorithm:
    def __init__(self,
                 population_size: int = 100,
                 generations: int = 100,
                 elite_size: int = 2,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.1,
                 selection_method: str = 'tournament',  # 'tournament' or 'roulette'
                 crossover_method: str = 'smart',  # 'smart' or 'order'
                 mutation_method: str = 'swap'):  # 'swap', 'inversion', or 'scramble'
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
        """Create initial random population."""
        self.population = []
        print("\nInitializing population with random chromosomes:")
        for i in range(self.population_size):
            chromosome = JobShopChromosome(self.problem)
            self.population.append(chromosome)

            # Debug info for first few chromosomes
            if i < 5:
                print(f"Chromosome {i}: {chromosome.chromosome}")

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

    def select_parent(self) -> JobShopChromosome:
        """Select parent using specified selection method."""
        if self.selection_method == 'tournament':
            return tournament_selection(self)
        elif self.selection_method == 'roulette':
            fitness_values = [chr.fitness for chr in self.population]
            return roulette_wheel_selection(self.population, fitness_values)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

    def apply_mutation(self, chromosome: List[int]) -> List[int]:
        """Apply specified mutation method."""
        if self.mutation_method == 'swap':
            return swap_mutation(chromosome, self.mutation_rate)
        elif self.mutation_method == 'inversion':
            return inversion_mutation(chromosome, self.mutation_rate)
        elif self.mutation_method == 'scramble':
            return scramble_mutation(chromosome, self.mutation_rate)
        else:
            raise ValueError(f"Unknown mutation method: {self.mutation_method}")

    def apply_crossover(self, parent1: JobShopChromosome, parent2: JobShopChromosome) -> List[int]:
        """Apply specified crossover method."""
        if self.crossover_method == 'smart':
            return smart_crossover(parent1.chromosome, parent2.chromosome, self.problem)
        elif self.crossover_method == 'order':
            return order_crossover(parent1.chromosome, parent2.chromosome, self.problem)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")

    def evaluate_population(self):
        """Evaluate all chromosomes in the population."""
        fitness_values = []
        for chromosome in self.population:
            chromosome.schedule = chromosome.decode_to_schedule()
            fitness_values.append(chromosome.fitness)

            # Update best solution ever if we found a better one
            print(f"Chromosome fitness: {chromosome.fitness:.2f}, best fitness: {self.best_fitness_ever:.2f}")
            if chromosome.fitness < self.best_fitness_ever:
                print(f"\nNew best solution found with fitness: {chromosome.fitness}")
                self.best_fitness_ever = chromosome.fitness
                # Create a deep copy of the chromosome to preserve the best solution
                self.best_solution_ever = deepcopy(chromosome)
                # Ensure the schedule is preserved in the copy
                self.best_solution_ever.schedule = self.best_solution_ever.decode_to_schedule()

        print(f"\nPopulation evaluation:")
        print(f"Min fitness: {min(fitness_values):.2f}")
        print(f"Max fitness: {max(fitness_values):.2f}")
        print(f"Avg fitness: {sum(fitness_values) / len(fitness_values):.2f}")
        return fitness_values

    def run(self) -> Tuple[JobShopChromosome, Dict]:
        """Run the genetic algorithm with enhanced tracking."""
        print("Starting genetic algorithm...")
        self.initialize_population()

        # Evaluate initial population
        print("\nEvaluating initial population...")
        initial_fitness = self.evaluate_population()
        print(f"Initial best fitness: {min(initial_fitness):.2f}")

        for generation in range(self.generations):
            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness)

            # Get current generation statistics
            current_best = self.population[0].fitness
            current_worst = self.population[-1].fitness
            current_avg = np.mean([chr.fitness for chr in self.population])
            current_diversity = self.calculate_diversity()

            # Update history
            self.history['best_fitness'].append(current_best)
            self.history['worst_fitness'].append(current_worst)
            self.history['avg_fitness'].append(current_avg)
            self.history['diversity'].append(current_diversity)

            # Store current best solution
            best_solution_copy = deepcopy(self.population[0])
            best_solution_copy.schedule = best_solution_copy.decode_to_schedule()
            self.history['best_solutions'].append(best_solution_copy)

            # Print detailed generation info
            if generation % 10 == 0:
                print(f"\nGeneration {generation}:")
                print(f"Best Fitness: {current_best:.2f}")
                print(f"Worst Fitness: {current_worst:.2f}")
                print(f"Average Fitness: {current_avg:.2f}")
                print(f"Population Diversity: {current_diversity:.2f}")
                print(f"Best solution ever: {self.best_fitness_ever:.2f}")

            # Create new population
            new_population = []

            # Elitism with deep copy
            new_population.extend([deepcopy(chrom) for chrom in self.population[:self.elite_size]])

            # Fill rest of population
            while len(new_population) < self.population_size:
                # Select parents using selected method
                parent1 = self.select_parent()
                parent2 = self.select_parent()

                # Create child using selected crossover method
                child_sequence = self.apply_crossover(parent1, parent2)

                # Apply selected mutation method
                child_sequence = self.apply_mutation(child_sequence)

                # Create and evaluate new chromosome
                child = JobShopChromosome(self.problem)
                child.chromosome = child_sequence
                child.schedule = child.decode_to_schedule()

                # Check if child is new best solution
                if child.fitness < self.best_fitness_ever:
                    print(f"\nNew best solution found in generation {generation} with fitness: {child.fitness}")
                    self.best_fitness_ever = child.fitness
                    best_copy = deepcopy(child)
                    best_copy.schedule = best_copy.decode_to_schedule()
                    self.best_solution_ever = best_copy

                # Debug info occasionally
                if len(new_population) == self.population_size // 10:
                    print(f"\nSample crossover result:")
                    print(f"Parent 1 fitness: {parent1.fitness:.2f}")
                    print(f"Parent 2 fitness: {parent2.fitness:.2f}")
                    print(f"Child fitness: {child.fitness:.2f}")

                new_population.append(child)

            self.population = new_population

            # Early stopping if no improvement for many generations
            if current_best >= self.best_fitness_ever and generation > 20:
                stagnant_generations = sum(1 for i in range(-20, 0)
                                           if self.history['best_fitness'][i] >= self.best_fitness_ever)
                if stagnant_generations == 20:
                    print("\nEarly stopping: No improvement for 20 generations")
                    break

        # Return best solution ever found
        return self.best_solution_ever, self.history
