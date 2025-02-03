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
    """
    Select parent using tournament selection with dynamic sizing.

    1. Ensures tournament size is never larger than population
    2. Handles edge cases gracefully
    3. Uses dynamic tournament sizing if needed

    Returns:
        JobShopChromosome: The selected parent chromosome
    """
    # Ensure tournament size is valid
    effective_tournament_size = min(
        self.tournament_size,
        len(self.population) - 1  # Leave room for at least one other competitor
    )

    if effective_tournament_size < 2:
        # If we can't do a proper tournament, just return random individual
        return np.random.choice(self.population)

    # Randomly select tournament_size chromosomes from the population
    tournament = np.random.choice(
        self.population,
        size=effective_tournament_size,
        replace=False  # Don't select the same chromosome twice
    )

    # Return the chromosome with the best fitness from the tournament
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

def jox_crossover(parent1: List[int], parent2: List[int], problem) -> List[int]:
    size = len(parent1)
    child = [-1] * size

    # Get unique jobs and randomly select subset
    unique_jobs = list(set(parent1))
    num_jobs_to_select = random.randint(1, len(unique_jobs))
    selected_jobs = random.sample(unique_jobs, num_jobs_to_select)

    # Copy selected jobs from parent1
    used_ops = {}
    for i, job_id in enumerate(parent1):
        if job_id in selected_jobs:
            child[i] = job_id
            used_ops[job_id] = used_ops.get(job_id, 0) + 1

    # Fill remaining positions from parent2, maintaining relative order
    p2_idx = 0
    for i in range(size):
        if child[i] == -1:  # Position needs to be filled
            while p2_idx < size:
                job_id = parent2[p2_idx]
                max_ops = len(problem.jobs_data[job_id])
                if used_ops.get(job_id, 0) < max_ops:
                    child[i] = job_id
                    used_ops[job_id] = used_ops.get(job_id, 0) + 1
                    p2_idx += 1
                    break
                p2_idx += 1

    return child

