import random
from typing import List

import numpy as np

from models import JobShopChromosome


# Selection methods
def roulette_wheel_selection(population: List, fitness_values: List[float]) -> any:
    """
    1. Handles cases where all fitness values are equal
    2. Prevents division by zero
    3. Normalizes probabilities
    """
    if not population or not fitness_values:
        raise ValueError("Empty population or fitness values")

    # Handle case where all fitness values are equal
    if len(set(fitness_values)) == 1:
        return random.choice(population)

    # Convert fitness to selection probability (lower fitness = higher probability)
    min_fitness = min(fitness_values)
    adjusted_fitness = [f - min_fitness + 1e-10 for f in fitness_values]  # Prevent division by zero
    total_fitness = sum(1 / f for f in adjusted_fitness)
    probabilities = [(1 / f) / total_fitness for f in adjusted_fitness]

    # Normalize probabilities to ensure they sum to 1
    probabilities = [p / sum(probabilities) for p in probabilities]

    # Select based on probabilities
    return random.choices(population, probabilities, k=1)[0]

def tournament_selection(self) -> JobShopChromosome:

    tournament = np.random.choice(
        self.population,
        size=len(self.population),
        replace=False
    )

    return min(tournament, key=lambda x: x.fitness)


# Mutation methods
def swap_mutation(chromosome: List[int], mutation_rate: float) -> List[int]:
    """Enhanced mutation operator with variable mutation strength."""
    if random.random() < mutation_rate:
        # Determine mutation strength (1 to 3 swaps), how many positions to swap
        num_swaps = random.randint(1, 3)

        for _ in range(num_swaps):
            idx1, idx2 = random.sample(range(len(chromosome)), 2)

            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

    return chromosome

def inversion_mutation(chromosome: List[int], mutation_rate: float) -> List[int]:
    """Inversion mutation operator - reverses a subsequence of the chromosome."""
    if random.random() < mutation_rate:
        point1, point2 = sorted(random.sample(range(len(chromosome)), 2))
        chromosome[point1:point2] = reversed(chromosome[point1:point2])
    return chromosome


# Crossover methods
def ppx_crossover(parent1: List[int], parent2: List[int], problem) -> List[int]:
    size = len(parent1)
    child = [-1] * size

    # Create binary template
    template = [random.randint(0, 1) for _ in range(size)]

    # Track used operations for each job
    used_ops = {}
    p1_idx = 0
    p2_idx = 0

    # Fill child chromosome according to template
    for i in range(size):
        current_parent = parent1 if template[i] == 1 else parent2
        current_idx = p1_idx if template[i] == 1 else p2_idx

        while current_idx < size:
            job_id = current_parent[current_idx]
            max_ops = len(problem.jobs_data[job_id])
            if used_ops.get(job_id, 0) < max_ops:
                child[i] = job_id
                used_ops[job_id] = used_ops.get(job_id, 0) + 1
                if template[i] == 1:
                    p1_idx = current_idx + 1
                else:
                    p2_idx = current_idx + 1
                break
            current_idx += 1

    return child


def simple_crossover(parent1: List[int], parent2: List[int], problem) -> List[int]:
    point = random.randint(1, len(parent1) - 1)

    child = parent1[:point] + parent2[point:]

    # Fix any invalid operation counts
    job_counts = {}
    for job in child:
        job_counts[job] = job_counts.get(job, 0) + 1

    # Replace excess operations
    for i, job in enumerate(child):
        max_ops = len(problem.jobs_data[job])
        if job_counts[job] > max_ops:
            for other_job in range(len(problem.jobs_data)):
                if job_counts.get(other_job, 0) < len(problem.jobs_data[other_job]):
                    child[i] = other_job
                    job_counts[job] -= 1
                    job_counts[other_job] = job_counts.get(other_job, 0) + 1
                    break

    return child

