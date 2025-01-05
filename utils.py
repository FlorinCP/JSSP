from typing import List

from genetic_algo import GeneticAlgorithm
from job_shop_chromosome import JobShopChromosome
from job_shop_problem import JobShopProblem


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

def demonstrate_job_shop_scheduling():
    """
    This function demonstrates how our job shop scheduling system works.
    It creates parent solutions, combines them to make new solutions,
    and shows us the schedules and their performance.
    """
    # First, create our problem instance
    print("Creating Job Shop Problem...")
    problem = JobShopProblem()
    print(f"We have {problem.num_jobs} jobs and {problem.num_machines} machines")

    # Create and show parent solutions
    print("\nCreating parent solutions...")

    # Make first parent solution
    parent1 = [0, 1, 2, 0, 1, 2, 0, 1]
    print(f"Parent 1 sequence: {parent1}")

    # Make second parent solution
    parent2 = [1, 2, 0, 1, 2, 0, 1, 0]
    print(f"Parent 2 sequence: {parent2}")

    # Create child solution using our smart crossover
    print("\nCreating child solution through crossover...")
    child = smart_crossover(parent1, parent2, problem)
    print(f"Child sequence: {child}")

    # Now let's evaluate all three solutions
    print("\nEvaluating all solutions...")

    def evaluate_solution(chromosome_sequence, name):
        """Helper function to evaluate and print details of a solution"""
        print(f"\nEvaluating {name}...")
        chromosome = JobShopChromosome(problem)
        chromosome.chromosome = chromosome_sequence
        schedule = chromosome.decode_to_schedule()

        # Print a nice schedule visualization
        print(f"\n{name} Schedule Details:")
        print("Job Operation  Machine  Start   End   Duration")
        print("-" * 50)
        for (job_id, op_idx), details in sorted(schedule.items()):
            duration = details['end'] - details['start']
            print(
                f"{job_id:3d} {op_idx:9d} {details['machine']:8d} {details['start']:7d} {details['end']:5d} {duration:8d}")

        return chromosome.fitness

    # Evaluate all three solutions
    fitness1 = evaluate_solution(parent1, "Parent 1")
    fitness2 = evaluate_solution(parent2, "Parent 2")
    fitness3 = evaluate_solution(child, "Child")

    # Compare results
    print("\nFinal Results:")
    print(f"Parent 1 completion time: {fitness1}")
    print(f"Parent 2 completion time: {fitness2}")
    print(f"Child completion time: {fitness3}")

    # See if we got improvement
    best_parent = min(fitness1, fitness2)
    if fitness3 < best_parent:
        print("\nSuccess! Child solution is better than both parents")
        print(f"Improvement: {best_parent - fitness3} time units")
    else:
        print("\nChild solution is not better than the best parent")
        print("This is normal in genetic algorithms - not every combination produces improvement")


def run_simulation():
    """Run a complete simulation with visualization"""
    import matplotlib.pyplot as plt

    # Create and run GA
    ga = GeneticAlgorithm(
        population_size=100,
        generations=100,
        elite_size=2,
        tournament_size=5,
        mutation_rate=0.1
    )

    best_solution, best_history, avg_history = ga.run()

    # Print final solution
    print("\nBest Solution Found:")
    print(f"Chromosome: {best_solution.chromosome}")
    print(f"Fitness (Total Time): {best_solution.fitness}")

    # Plot fitness history
    plt.figure(figsize=(10, 6))
    plt.plot(best_history, label='Best Fitness')
    plt.plot(avg_history, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Total Time)')
    plt.title('Genetic Algorithm Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print detailed schedule
    print("\nDetailed Schedule:")
    schedule = best_solution.decode_to_schedule()
    print("Job Operation  Machine  Start   End   Duration")
    print("-" * 50)
    for (job_id, op_idx), details in sorted(schedule.items()):
        duration = details['end'] - details['start']
        print(f"{job_id:3d} {op_idx:9d} {details['machine']:8d} "
              f"{details['start']:7d} {details['end']:5d} {duration:8d}")