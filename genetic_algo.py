import numpy as np
from typing import List, Tuple

from job_shop_chromosome import JobShopChromosome
from main import JobShopProblem
from utils import smart_crossover


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
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def initialize_population(self):
        """Create initial random population"""
        self.population = []
        for _ in range(self.population_size):
            chromosome = JobShopChromosome(self.problem)
            self.population.append(chromosome)

    def tournament_selection(self) -> JobShopChromosome:
        """Select parent using tournament selection"""
        tournament = np.random.choice(self.population, self.tournament_size, replace=False)
        return min(tournament, key=lambda x: x.fitness)

    def mutation(self, chromosome: List[int]) -> List[int]:
        """Apply swap mutation with given probability"""
        if np.random.random() < self.mutation_rate:
            # Select two random positions and swap them
            pos1, pos2 = np.random.choice(len(chromosome), 2, replace=False)
            chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
        return chromosome

    def run(self) -> Tuple[JobShopChromosome, List[float], List[float]]:
        """Run the genetic algorithm"""
        # Initialize population
        print("Initializing population...")
        self.initialize_population()

        # Evaluate initial population
        for chromosome in self.population:
            _ = chromosome.decode_to_schedule()  # This sets the fitness

        # Main loop
        for generation in range(self.generations):
            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness)

            # Store statistics
            best_fitness = self.population[0].fitness
            avg_fitness = np.mean([chr.fitness for chr in self.population])
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)

            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}, "
                      f"Avg Fitness = {avg_fitness:.2f}")

            # Create new population
            new_population = []

            # Elitism: Keep best solutions
            new_population.extend(self.population[:self.elite_size])

            # Create rest of new population
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()

                # Create child through crossover
                child_sequence = smart_crossover(parent1.chromosome,
                                                 parent2.chromosome,
                                                 self.problem)

                # Apply mutation
                child_sequence = self.mutation(child_sequence)

                # Create new chromosome with modified sequence
                child = JobShopChromosome(self.problem)
                child.chromosome = child_sequence
                _ = child.decode_to_schedule()  # Set fitness

                new_population.append(child)

            self.population = new_population

        # Return best solution and history
        self.population.sort(key=lambda x: x.fitness)
        return (self.population[0],
                self.best_fitness_history,
                self.avg_fitness_history)