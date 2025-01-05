from typing import List, Dict

import numpy as np


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


class GAStatisticsAnalyzer:
    """Analyzer for Genetic Algorithm statistics."""

    @staticmethod
    def calculate_statistics(history: Dict, best_solution: JobShopChromosome) -> Dict:
        """Calculate detailed statistics from the GA run."""
        stats = {
            'best_fitness': best_solution.fitness,
            'final_diversity': history['diversity'][-1],
            'total_improvement': history['best_fitness'][0] - history['best_fitness'][-1],
            'improvement_percentage': ((history['best_fitness'][0] - history['best_fitness'][-1]) /
                                       history['best_fitness'][0] * 100),
            'convergence_generation': None,
            'average_improvement_rate': np.mean(np.diff(history['best_fitness'])),
            'best_generation': np.argmin(history['best_fitness']),
            'stagnant_generations': 0,
            'final_schedule_makespan': max(
                details['end'] for details in best_solution.schedule.values()
            )
        }

        # Calculate convergence information
        improvements = np.abs(np.diff(history['best_fitness']))
        threshold = np.mean(improvements) * 0.01  # 1% of average improvement
        converged_gens = np.where(improvements < threshold)[0]
        if len(converged_gens) > 0:
            stats['convergence_generation'] = converged_gens[0]

        # Calculate stagnation information
        stagnant_count = 0
        for i in range(1, len(history['best_fitness'])):
            if abs(history['best_fitness'][i] - history['best_fitness'][i - 1]) < 1e-6:
                stagnant_count += 1
            else:
                stagnant_count = 0
            stats['stagnant_generations'] = max(
                stats['stagnant_generations'],
                stagnant_count
            )

        return stats

    @staticmethod
    def print_statistics(stats: Dict):
        """Print formatted statistics."""
        print("\n" + "=" * 50)
        print("FINAL STATISTICS")
        print("=" * 50)

        print("\nPerformance Metrics:")
        print(f"Best Fitness Achieved: {stats['best_fitness']:.2f}")
        print(f"Final Schedule Makespan: {stats['final_schedule_makespan']}")
        print(f"Total Improvement: {stats['total_improvement']:.2f} " +
              f"({stats['improvement_percentage']:.1f}%)")
        print(f"Average Improvement Rate: {stats['average_improvement_rate']:.3f} per generation")

        print("\nConvergence Analysis:")
        if stats['convergence_generation'] is not None:
            print(f"Convergence reached at generation: {stats['convergence_generation']}")
        else:
            print("Algorithm did not fully converge")
        print(f"Best solution found in generation: {stats['best_generation']}")
        print(f"Maximum stagnant generations: {stats['stagnant_generations']}")

        print("\nDiversity Metrics:")
        print(f"Final Population Diversity: {stats['final_diversity']:.2f}")

        print("\n" + "=" * 50)
