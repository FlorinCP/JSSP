from typing import List, Dict

import numpy as np


class JobShopProblem:
    """Job Shop Scheduling Problem definition."""

    def __init__(self):
        # This is a larger problem instance with 6 jobs and 6 machines
        self.jobs_data = [
            [(0, 5), (1, 4), (2, 4), (3, 3), (4, 6), (5, 3), (6, 5), (7, 4), (8, 3), (9, 4)],  # Job 0
            [(1, 6), (0, 5), (3, 4), (2, 3), (5, 5), (4, 4), (7, 5), (6, 6), (9, 3), (8, 4)],  # Job 1
            [(2, 4), (1, 5), (0, 5), (4, 4), (3, 5), (6, 4), (5, 5), (8, 5), (7, 4), (9, 5)],  # Job 2
            [(3, 5), (2, 6), (1, 4), (5, 3), (4, 5), (7, 4), (6, 5), (9, 4), (8, 5), (0, 4)],  # Job 3
            [(4, 3), (3, 4), (2, 5), (6, 4), (5, 4), (8, 3), (7, 5), (0, 5), (9, 4), (1, 5)],  # Job 4
            [(5, 4), (4, 5), (3, 4), (7, 3), (6, 5), (9, 4), (8, 4), (1, 5), (0, 4), (2, 3)],  # Job 5
            [(6, 5), (5, 4), (4, 3), (8, 4), (7, 5), (0, 5), (9, 4), (2, 3), (1, 4), (3, 5)],  # Job 6
            [(7, 4), (6, 5), (5, 4), (9, 3), (8, 4), (1, 5), (0, 4), (3, 4), (2, 5), (4, 4)],  # Job 7
            [(8, 3), (7, 4), (6, 5), (0, 4), (9, 5), (2, 4), (1, 3), (4, 5), (3, 4), (5, 5)],  # Job 8
            [(9, 5), (8, 4), (7, 3), (1, 5), (0, 4), (3, 5), (2, 4), (5, 3), (4, 4), (6, 5)]  # Job 9
        ]
        self.num_jobs = len(self.jobs_data)
        self.num_machines = 6

    def validate_solution(self, chromosome):
        """Validate if a chromosome represents a valid solution."""
        # Count operations per job
        job_counts = {}
        for job_id in chromosome:
            job_counts[job_id] = job_counts.get(job_id, 0) + 1

        # Check if each job has correct number of operations
        for job_id, job_ops in enumerate(self.jobs_data):
            if job_counts.get(job_id, 0) != len(job_ops):
                return False
        return True

    def get_operation_details(self, job_id: int, operation_idx: int) -> tuple:
        """Get machine and processing time for a specific operation."""
        return self.jobs_data[job_id][operation_idx]

    def calculate_makespan(self, schedule: dict) -> int:
        """Calculate the total makespan of a schedule."""
        return max(details['end'] for details in schedule.values())


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
        operations = []

        for job_id, job_operations in enumerate(self.problem.jobs_data):
            operations.extend([job_id] * len(job_operations))

        np.random.shuffle(operations)
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
            required_ops = len(job_operations)  # How many operations job should have
            actual_ops = job_counts.get(job_id, 0)  # How many it has in chromosome
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

        if not self.validate_chromosome():
            raise ValueError("Invalid chromosome!")

        machine_available_time = {}  # When will each machine be free?
        job_available_time = {}  # When will each job be ready for its next operation?
        schedule = {}  # Our final schedule (what we'll return)

        for position, job_id in enumerate(self.chromosome):
            # Figure out which operation this is for this job
            # (how many times have we seen this job so far?)
            job_occurrences = self.chromosome[:position + 1].count(job_id)
            operation_index = job_occurrences - 1

            # Get the details for this operation
            machine_id, processing_time = self.problem.jobs_data[job_id][operation_index]

            # When can we start this operation?
            machine_ready = machine_available_time.get(machine_id, 0)  # When machine is free
            job_ready = job_available_time.get(job_id, 0)  # When job is ready
            start_time = max(machine_ready, job_ready)  # Start at later of these times
            end_time = start_time + processing_time  # Calculate when we'll finish

            machine_available_time[machine_id] = end_time
            job_available_time[job_id] = end_time

            schedule[(job_id, operation_index)] = {
                'machine': machine_id,
                'start': start_time,
                'end': end_time
            }

        self.fitness = max(job_available_time.values())
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
