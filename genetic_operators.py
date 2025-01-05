from typing import List

import numpy as np

from models import JobShopProblem


def smart_crossover(parent1: List[int], parent2: List[int], problem: JobShopProblem) -> List[int]:
    """
    Creates a new solution by combining parts of two parent solutions.
    Makes sure the new solution is valid (has correct number of operations).
    """
    # First, figure out how many operations each job needs
    # For example, this is one possible outputs: {0: 3, 1: 3, 2: 2}
    required_operations = {
        job_id: len(operations)
        for job_id, operations in enumerate(problem.jobs_data)
    }

    # Start with first half of parent1
    crossover_point = len(parent1) // 2
    child = parent1[:crossover_point].copy()

    # Keep track of operations we've used so far
    # This could be {0: 2, 1: 1, 2: 1},
    # we need to make sure that the other part of the child has the remaining operations for a valid sequence
    current_operations = {}
    for job in child:
        current_operations[job] = current_operations.get(job, 0) + 1

    # Add remaining operations, making sure we maintain valid counts
    parent2_index = crossover_point
    while len(child) < len(parent1):
        # Try to use job from parent2
        candidate_job = parent2[parent2_index]

        # Can we still add more of this job?
        if current_operations.get(candidate_job, 0) < required_operations[candidate_job]:
            child.append(candidate_job)
            current_operations[candidate_job] = current_operations.get(candidate_job, 0) + 1
        else:
            # Find a job that still needs operations
            for job_id, required in required_operations.items():
                if current_operations.get(job_id, 0) < required:
                    child.append(job_id)
                    current_operations[job_id] = current_operations.get(job_id, 0) + 1
                    break

        # Move to next position in parent2 (wrap around if needed)
        parent2_index = (parent2_index + 1) % len(parent2)

    return child

def swap_mutation(chromosome: List[int], mutation_rate: float) -> List[int]:
    """Swap mutation operator."""
    if np.random.random() < mutation_rate:
        pos1, pos2 = np.random.choice(len(chromosome), 2, replace=False)
        chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
    return chromosome