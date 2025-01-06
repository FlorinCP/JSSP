from typing import List, Dict
import numpy as np

from jsp_parser import JSPParser


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

    def load_from_file(self, file_path: str, instance_name: str = None):
        """
        Load problem instance from a file.

        Args:
            file_path: Path to the file containing JSP instances
            instance_name: Optional name of specific instance to load.
                         If None, loads the first instance found.
        """
        with open(file_path, 'r') as f:
            content = f.read()

        instances = JSPParser.parse_file(content)

        if not instances:
            raise ValueError("No valid instances found in file")

        if instance_name is None:
            # Take first instance if none specified
            instance = next(iter(instances.values()))
        else:
            if instance_name not in instances:
                raise ValueError(f"Instance '{instance_name}' not found in file")
            instance = instances[instance_name]

        # Update problem attributes
        self.jobs_data = instance.jobs_data
        self.num_jobs = instance.num_jobs
        self.num_machines = instance.num_machines
        self.instance_name = instance.name
        self.instance_description = instance.description


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

    def __str__(self):
        """String representation of the problem instance."""
        return (f"Job Shop Problem Instance: {self.instance_name}\n"
                f"Description: {self.instance_description}\n"
                f"Jobs: {self.num_jobs}, Machines: {self.num_machines}")


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


