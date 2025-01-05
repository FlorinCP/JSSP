import numpy as np
from typing import List, Dict, Tuple


class JobShopProblem:
    """
    This class represents a Job Shop Scheduling Problem.
    Think of it like managing a workshop where different jobs need to use different machines
    in a specific order, like a car going through different stations in a repair shop.
    """

    def __init__(self):
        # jobs_data stores our manufacturing instructions
        # For each job, we have a list of tuples: (machine_number, time_needed)
        # Example: [(0,3), (1,2)] means:
        #   - First use machine 0 for 3 time units
        #   - Then use machine 1 for 2 time units
        self.jobs_data = [
            [(0, 3), (1, 2), (2, 2)],  # Job 0's sequence of operations
            [(0, 2), (2, 1), (1, 4)],  # Job 1's sequence of operations
            [(1, 4), (2, 3)]           # Job 2's sequence of operations
        ]

        # Count how many jobs and machines we have
        self.num_jobs = len(self.jobs_data)  # How many different jobs
        self.num_machines = 3                # How many machines available


class JobShopChromosome:
    """
    This class represents one possible solution to our scheduling problem.
    Think of it like a DNA sequence that tells us the order to process our jobs.
    """

    def __init__(self, problem: JobShopProblem):
        self.problem = problem
        self.chromosome = self.create_random_chromosome()
        self.fitness = float('inf')

    def create_random_chromosome(self) -> List[int]:
        """
        Creates a random valid sequence of jobs.
        Example: If Job 0 needs 3 operations and Job 1 needs 2 operations,
        we might create: [0,1,0,1,0]
        """
        # Start with an empty list
        operations = []

        # For each job in our problem...
        for job_id, job_operations in enumerate(self.problem.jobs_data):

            # How many operations does this job need?
            num_operations = len(job_operations)

            # Add this job's ID that many times to our list
            operations.extend([job_id] * num_operations)
            print(f"Added {num_operations} operations for Job {job_id}")
            print(f"Current operations list: {operations}")

        # Shuffle the list randomly to create our chromosome
        np.random.shuffle(operations)
        print(f"Final chromosome after shuffling: {operations}")
        return operations

    def validate_chromosome(self) -> bool:
        """
        Checks if our chromosome is valid (has correct number of operations for each job).
        Returns True if valid, False if not.
        """
        # Count how many times each job appears
        job_counts = {}
        for job_id in self.chromosome:
            job_counts[job_id] = job_counts.get(job_id, 0) + 1

        # Check against how many operations each job should have
        for job_id, job_operations in enumerate(self.problem.jobs_data):
            required_ops = len(job_operations)                              # How many operations job should have
            actual_ops = job_counts.get(job_id, 0)                          # How many it has in chromosome
            if required_ops != actual_ops:
                print(f"Invalid chromosome! Job {job_id} needs {required_ops} "
                      f"operations but has {actual_ops}")
                return False
        return True

    def decode_to_schedule(self) -> Dict:
        """
        Converts our chromosome (sequence of jobs) into an actual schedule.
        Returns a dictionary telling us when each operation starts and ends.
        """

        # First, make sure our chromosome is valid
        if not self.validate_chromosome():
            raise ValueError("Invalid chromosome!")

        # These dictionaries help us track timing:
        machine_available_time = {}                 # When will each machine be free?
        job_available_time = {}                     # When will each job be ready for its next operation?
        schedule = {}                               # Our final schedule (what we'll return)

        print("\nDecoding chromosome into schedule:")
        print(f"Chromosome sequence: {self.chromosome}")

        # Process each job in our sequence
        for position, job_id in enumerate(self.chromosome):

            # Figure out which operation this is for this job
            # (how many times have we seen this job so far?)
            job_occurrences = self.chromosome[:position + 1].count(job_id)
            operation_index = job_occurrences - 1

            # Get the details for this operation
            machine_id, processing_time = self.problem.jobs_data[job_id][operation_index]

            # When can we start this operation?
            machine_ready = machine_available_time.get(machine_id, 0)    # When machine is free
            job_ready = job_available_time.get(job_id, 0)                # When job is ready
            start_time = max(machine_ready, job_ready)                   # Start at later of these times
            end_time = start_time + processing_time                      # Calculate when we'll finish

            # Print detailed information about what we're scheduling
            print(f"\nScheduling Job {job_id}, Operation {operation_index}:")
            print(f"  Using Machine {machine_id} for {processing_time} time units")
            print(f"  Machine ready at: {machine_ready}")
            print(f"  Job ready at: {job_ready}")
            print(f"  Will start at: {start_time}")
            print(f"  Will finish at: {end_time}")

            # Update our tracking information
            machine_available_time[machine_id] = end_time  # Machine will be busy until end_time
            job_available_time[job_id] = end_time  # Job can't continue until end_time

            # Store this operation in our schedule
            schedule[(job_id, operation_index)] = {
                'machine': machine_id,
                'start': start_time,
                'end': end_time
            }

        # The fitness is the total time needed (when does the last operation finish?)
        self.fitness = max(job_available_time.values())
        print(f"\nFinal schedule completion time: {self.fitness}")
        return schedule


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


# Save all the previous code (JobShopProblem, JobShopChromosome, smart_crossover)
# in a file named job_shop.py

if __name__ == "__main__":
    print("Starting Job Shop Scheduling demonstration...\n")
    demonstrate_job_shop_scheduling()