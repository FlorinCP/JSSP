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


def scramble_mutation(chromosome: List[int], mutation_rate: float) -> List[int]:
    """Scramble mutation operator - randomly shuffles a subsequence."""
    if random.random() < mutation_rate:
        point1, point2 = sorted(random.sample(range(len(chromosome)), 2))
        subsequence = chromosome[point1:point2]
        random.shuffle(subsequence)
        chromosome[point1:point2] = subsequence
    return chromosome


# Crossover methods
def smart_crossover(parent1: List[int], parent2: List[int], problem) -> List[int]:
    """Enhanced crossover operator with better diversity preservation."""

    chromosome_length = len(parent1)
    child = [-1] * chromosome_length

    point1, point2 = sorted(random.sample(range(chromosome_length), 2))

    # Copy segment from first parent
    child[point1:point2] = parent1[point1:point2]

    # Track used operations for each job
    used_ops = {}
    for job_id in child[point1:point2]:
        if job_id != -1:
            used_ops[job_id] = used_ops.get(job_id, 0) + 1

    # Fill remaining positions from second parent
    parent2_idx = 0
    for i in range(chromosome_length):
        if i < point1 or i >= point2:  # Outside the copied segment
            while parent2_idx < chromosome_length:
                job_id = parent2[parent2_idx]
                max_ops = len(problem.jobs_data[job_id])
                if used_ops.get(job_id, 0) < max_ops:
                    child[i] = job_id
                    used_ops[job_id] = used_ops.get(job_id, 0) + 1
                    parent2_idx += 1
                    break
                parent2_idx += 1

    # Validate and repair if necessary
    if -1 in child or not problem.validate_solution(child):
        # If invalid, create a repair sequence
        missing_ops = {}
        for job_id, job_ops in enumerate(problem.jobs_data):
            required = len(job_ops)
            actual = sum(1 for x in child if x == job_id)
            if required > actual:
                missing_ops[job_id] = required - actual

        # Fill missing operations
        for i, gene in enumerate(child):
            if gene == -1:
                for job_id, count in missing_ops.items():
                    if count > 0:
                        child[i] = job_id
                        missing_ops[job_id] -= 1
                        break

    return child


def order_crossover(parent1: List[int], parent2: List[int], problem) -> List[int]:
    """True Order Crossover (OX) operator.

    This version:
    1. Selects a subsequence from parent1
    2. Copies that subsequence to child
    3. Orders the remaining jobs from parent2 based on their relative order
    4. Places the ordered jobs in available positions
    """
    size = len(parent1)
    # Select subsequence boundaries
    point1, point2 = sorted(random.sample(range(size), 2))

    # Initialize child
    child = [-1] * size

    # Step 1: Copy subsequence from parent1
    child[point1:point2] = parent1[point1:point2]

    # Step 2: Track what operations we've used
    used_ops = {}
    for job_id in child[point1:point2]:
        used_ops[job_id] = used_ops.get(job_id, 0) + 1

    # Step 3: Create ordered list of remaining operations from parent2
    # maintaining their relative order
    remaining_ops = []
    max_ops_per_job = {i: len(ops) for i, ops in enumerate(problem.jobs_data)}

    for job_id in parent2:
        current_count = used_ops.get(job_id, 0)
        if current_count < max_ops_per_job[job_id]:
            remaining_ops.append(job_id)
            used_ops[job_id] = used_ops.get(job_id, 0) + 1

    # Step 4: Fill the remaining positions in order
    remaining_idx = 0
    for i in range(size):
        if i < point1 or i >= point2:  # Outside the copied segment
            if remaining_idx < len(remaining_ops):
                child[i] = remaining_ops[remaining_idx]
                remaining_idx += 1

    # Validate and repair if necessary
    if -1 in child or not problem.validate_solution(child):
        missing_ops = {}
        for job_id, job_ops in enumerate(problem.jobs_data):
            required = len(job_ops)
            actual = sum(1 for x in child if x == job_id)
            if required > actual:
                missing_ops[job_id] = required - actual

        # Fill missing operations
        for i, gene in enumerate(child):
            if gene == -1:
                for job_id, count in missing_ops.items():
                    if count > 0:
                        child[i] = job_id
                        missing_ops[job_id] -= 1
                        break

    return child