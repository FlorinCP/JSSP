import random
from typing import List


def smart_crossover(parent1: List[int], parent2: List[int], problem) -> List[int]:
    """Enhanced crossover operator with better diversity preservation."""
    # Get length of chromosomes
    chromosome_length = len(parent1)

    # Choose two random crossover points
    point1, point2 = sorted(random.sample(range(chromosome_length), 2))

    # Initialize child with invalid values
    child = [-1] * chromosome_length

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


def swap_mutation(chromosome: List[int], mutation_rate: float) -> List[int]:
    """Enhanced mutation operator with variable mutation strength."""
    if random.random() < mutation_rate:
        # Determine mutation strength (1 to 3 swaps)
        num_swaps = random.randint(1, 3)

        for _ in range(num_swaps):
            # Choose positions to swap
            idx1, idx2 = random.sample(range(len(chromosome)), 2)

            # Perform swap
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

    return chromosome